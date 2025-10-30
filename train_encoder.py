import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Iterator, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import argparse

from qwen_feature_extract import QwenFeatureExtractor
from teacher_labels import LettuceTeacher
from encoder_model import SmallTokenEncoder


def set_seed(seed: int):
	random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


@dataclass
class Example:
	context: str
	question: str
	answer: str
	id: str


class RagTruthDataset(Dataset):
	def __init__(self, path: str):
		self.items: List[Example] = []
		with open(path, 'r', encoding='utf-8') as f:
			for line in f:
				if not line.strip():
					continue
				obj = json.loads(line)
				self.items.append(Example(
					context=obj.get('source') or '',
					question=obj.get('prompt') or '',
					answer=obj.get('answer') or '',
					id=str(obj.get('id'))
				))

	def __len__(self) -> int:
		return len(self.items)

	def __getitem__(self, idx: int) -> Example:
		return self.items[idx]


def build_answer_token_offsets(offset_mapping: List[List[int]], prefix_len: int, answer_token_indices: List[int]) -> List[List[int]]:
	ans_offsets: List[List[int]] = []
	for i in answer_token_indices:
		start, end = offset_mapping[i]
		ans_offsets.append([start, end])
	return ans_offsets


def compute_prf(pred: torch.Tensor, gold: torch.Tensor, mask: torch.Tensor) -> Tuple[int, int, int]:
	pred = (pred * mask).long()
	gold = (gold * mask).long()
	tp = int(((pred == 1) & (gold == 1)).sum().item())
	fp = int(((pred == 1) & (gold == 0)).sum().item())
	fn = int(((pred == 0) & (gold == 1)).sum().item())
	return tp, fp, fn


def make_collate(extractor: QwenFeatureExtractor, teacher: LettuceTeacher, device: str, use_bf16: bool, use_generation: bool, teacher_use_ctxq: bool):
	def collate_fn(examples: List[Example]) -> Dict[str, torch.Tensor]:
		features: List[torch.Tensor] = []
		labels_list: List[torch.Tensor] = []
		lengths: List[int] = []
		with torch.no_grad():
			for ex in examples:
				if use_generation:
					feat = extractor.generate_and_encode(ex.context, ex.question)
					ans_text = str(feat.get('generated_answer', ''))
				else:
					feat = extractor.encode_answer(ex.context, ex.question, ex.answer)
					ans_text = ex.answer
				ans_emb = feat['answer_embeddings']  # (L, D)
				offsets = feat['offset_mapping'].tolist()
				ans_idx = feat['answer_token_indices'].tolist()
				ans_offsets = build_answer_token_offsets(offsets, int(feat['prompt_prefix_len']), ans_idx)
				if teacher_use_ctxq:
					teacher_input = f"Context:\n{ex.context}\n\nQuestion:\n{ex.question}\n\nAnswer:\n{ans_text}"
				else:
					teacher_input = ans_text
				spans = teacher.predict_spans(teacher_input)
				labels = teacher.spans_to_token_labels(spans, ans_offsets)
				features.append(ans_emb)
				labels_list.append(labels)
				lengths.append(ans_emb.shape[0])
		max_len = max(lengths)
		feat_dim = features[0].shape[-1]
		B = len(examples)
		dtype = torch.bfloat16 if use_bf16 else torch.float16
		inputs = torch.zeros(B, max_len, feat_dim, dtype=dtype, device=device)
		labels = torch.zeros(B, max_len, dtype=torch.float32, device=device)
		mask = torch.zeros(B, max_len, dtype=torch.float32, device=device)
		for i, (fe, la, L) in enumerate(zip(features, labels_list, lengths)):
			inputs[i, :L, :] = fe.to(device).to(dtype)
			labels[i, :L] = la.to(device)
			mask[i, :L] = 1.0
		return {"inputs": inputs, "labels": labels, "mask": mask}
	return collate_fn


def train():
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', type=int, default=None)
	parser.add_argument('--batch_size', type=int, default=None)
	parser.add_argument('--lr', type=float, default=None)
	parser.add_argument('--max_train_samples', type=int, default=None)
	parser.add_argument('--max_val_samples', type=int, default=None)
	parser.add_argument('--threshold', type=float, default=None)
	parser.add_argument('--use_layers', type=str, default=None, help='comma-separated, e.g. -1,-2,-3')
	parser.add_argument('--use_generation', action='store_true', help='generate LLM answer and train on generated output')
	parser.add_argument('--teacher_use_ctxq', action='store_true', help='provide context+question+answer to teacher model')
	args = parser.parse_args()

	with open('config.yaml', 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)

	if args.epochs is not None:
		cfg['train']['num_epochs'] = int(args.epochs)
	if args.batch_size is not None:
		cfg['train']['batch_size'] = int(args.batch_size)
	if args.lr is not None:
		cfg['train']['learning_rate'] = float(args.lr)
	if args.max_train_samples is not None:
		cfg['train']['max_train_samples'] = int(args.max_train_samples)
	if args.max_val_samples is not None:
		cfg['train']['max_val_samples'] = int(args.max_val_samples)
	if args.threshold is not None:
		cfg['inference']['threshold'] = float(args.threshold)
	if args.use_layers is not None:
		layers = [int(x) for x in args.use_layers.split(',') if x]
		cfg['features']['use_layers'] = layers

	set_seed(int(cfg['train']['seed']))

	device = cfg['device'].get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
	use_bf16 = (cfg['device'].get('dtype') == 'bfloat16')

	processed_dir = cfg['paths']['processed_dir']
	train_path = os.path.join(processed_dir, 'train.jsonl')
	val_path = os.path.join(processed_dir, 'val.jsonl')

	train_ds = RagTruthDataset(train_path)
	val_ds = RagTruthDataset(val_path)

	if cfg['train'].get('max_train_samples'):
		train_ds.items = train_ds.items[:int(cfg['train']['max_train_samples'])]
	if cfg['train'].get('max_val_samples'):
		val_ds.items = val_ds.items[:int(cfg['train']['max_val_samples'])]

	extractor = QwenFeatureExtractor('config.yaml')
	extractor.use_layers = cfg['features']['use_layers']
	teacher = LettuceTeacher('config.yaml')

	# Determine concatenated feature dim using a tiny sample (generated or gold)
	dummy = train_ds[0] if len(train_ds) > 0 else Example('', '', 'a', '0')
	if args.use_generation:
		feat = extractor.generate_and_encode(dummy.context, dummy.question)
	else:
		feat = extractor.encode_answer(dummy.context, dummy.question, dummy.answer)
	concat_dim = feat['answer_embeddings'].shape[-1]

	encoder = SmallTokenEncoder(input_dim=concat_dim).to(device)

	batch_size = int(cfg['train']['batch_size'])
	accum = int(cfg['train']['grad_accum_steps'])
	epochs = int(cfg['train']['num_epochs'])
	lr = float(cfg['train']['learning_rate'])
	wd = float(cfg['train']['weight_decay'])
	thr = float(cfg['inference']['threshold'])

	optim = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=wd)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=max(1, len(train_ds)//max(1,batch_size)))
	crit = nn.BCEWithLogitsLoss(reduction='none')

	train_loader = DataLoader(
		train_ds,
		batch_size=batch_size,
		shuffle=True,
		collate_fn=make_collate(extractor, teacher, device, use_bf16, args.use_generation, args.teacher_use_ctxq),
		num_workers=0,
	)
	val_loader = DataLoader(
		val_ds,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=make_collate(extractor, teacher, device, use_bf16, args.use_generation, args.teacher_use_ctxq),
		num_workers=0,
	)

	def run_epoch(split: str) -> Tuple[float, float, float, float]:
		loader = train_loader if split == 'train' else val_loader
		total_loss = 0.0
		n_tokens = 0
		tp_all = fp_all = fn_all = 0
		encoder.train() if split == 'train' else encoder.eval()
		pbar = tqdm(loader, desc=f"{split}")
		optim.zero_grad(set_to_none=True)
		step = 0
		for batch in pbar:
			inputs = batch['inputs']
			labels = batch['labels']
			mask = batch['mask']
			with torch.amp.autocast('cuda', enabled=use_bf16):
				logits = encoder(inputs, mask)
				loss_mat = crit(logits, labels)
				loss = (loss_mat * mask).sum() / (mask.sum().clamp(min=1.0))
				probs = torch.sigmoid(logits)
				pred = (probs >= thr).float() * mask
			if split == 'train':
				(loss / accum).backward()
				if (step + 1) % accum == 0:
					optim.step()
					optim.zero_grad(set_to_none=True)
					scheduler.step()
			total_loss += float(loss.detach().cpu()) * int(mask.sum().item())
			n_tokens += int(mask.sum().item())
			tp, fp, fn = compute_prf(pred, labels, mask)
			tp_all += tp
			fp_all += fp
			fn_all += fn
			prec = (tp_all / (tp_all + fp_all)) if (tp_all + fp_all) > 0 else 0.0
			rec = (tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
			f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
			step += 1
			pbar.set_postfix({"loss": total_loss/max(1,n_tokens), "f1": f1})
		prec = (tp_all / (tp_all + fp_all)) if (tp_all + fp_all) > 0 else 0.0
		rec = (tp_all / (tp_all + fn_all)) if (tp_all + fn_all) > 0 else 0.0
		f1 = (2*prec*rec/(prec+rec)) if (prec+rec) > 0 else 0.0
		return (total_loss/max(1, n_tokens), prec, rec, f1)

	best_val_f1 = 0.0
	ckpt_dir = cfg['paths']['checkpoints_dir']
	os.makedirs(ckpt_dir, exist_ok=True)
	for ep in range(epochs):
		train_loss, tr_p, tr_r, tr_f1 = run_epoch('train')
		val_loss, v_p, v_r, v_f1 = run_epoch('val')
		print(f"epoch {ep+1}: train_loss={train_loss:.6f} trF1={tr_f1:.4f} val_loss={val_loss:.6f} valP={v_p:.4f} valR={v_r:.4f} valF1={v_f1:.4f}")
		if v_f1 >= best_val_f1:
			best_val_f1 = v_f1
			path = os.path.join(ckpt_dir, 'small_encoder.pt')
			torch.save({
				'state_dict': encoder.state_dict(),
				'concat_dim': concat_dim,
				'config': cfg,
			}, path)
			print("Saved best model to", path)


if __name__ == '__main__':
	train()
