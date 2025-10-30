import os
from typing import List, Dict, Any, Optional

import yaml
import torch
from transformers import AutoTokenizer

from qwen_feature_extract import QwenFeatureExtractor
from teacher_labels import LettuceTeacher
from encoder_model import SmallTokenEncoder


def load_encoder(ckpt_path: str) -> SmallTokenEncoder:
	ckpt = torch.load(ckpt_path, map_location='cpu')
	concat_dim = ckpt['concat_dim']
	model = SmallTokenEncoder(input_dim=concat_dim)
	model.load_state_dict(ckpt['state_dict'], strict=True)
	model.eval()
	return model


def tokens_to_spans(token_offsets: List[List[int]], token_scores: List[float], threshold: float) -> List[Dict[str, Any]]:
	spans: List[Dict[str, Any]] = []
	start, end = None, None
	for i, (off, score) in enumerate(zip(token_offsets, token_scores)):
		if score >= threshold and start is None:
			start, end = off[0], off[1]
		elif score >= threshold and start is not None:
			end = off[1]
		elif score < threshold and start is not None:
			spans.append({'start': start, 'end': end, 'score': 1.0, 'label': 'H'})
			start, end = None, None
	if start is not None:
		spans.append({'start': start, 'end': end, 'score': 1.0, 'label': 'H'})
	return spans


def highlight_html(text: str, spans: List[Dict[str, Any]], color: str = '#ffcccc') -> str:
	spans_sorted = sorted(spans, key=lambda s: s['start'])
	pos = 0
	html_parts: List[str] = []
	for sp in spans_sorted:
		if sp['start'] > pos:
			html_parts.append(text[pos:sp['start']])
		html_parts.append(f"<span style=\"background-color:{color}\">{text[sp['start']:sp['end']]}</span>")
		pos = sp['end']
	if pos < len(text):
		html_parts.append(text[pos:])
	return ''.join(html_parts)


def run_inference(context: str, question: str, answer: Optional[str] = None, config_path: str = 'config.yaml', teacher_use_ctxq: bool = True) -> Dict[str, Any]:
	with open(config_path, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	ckpt_path = os.path.join(cfg['paths']['checkpoints_dir'], 'small_encoder.pt')
	threshold = float(cfg['inference']['threshold'])

	extractor = QwenFeatureExtractor(config_path)
	teacher = LettuceTeacher(config_path)
	encoder = load_encoder(ckpt_path)

	preferred_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	encoder = encoder.to(preferred_device)
	enc_dtype = next(encoder.parameters()).dtype

	with torch.no_grad():
		if not answer:
			feat = extractor.generate_and_encode(context, question)
			answer_text = str(feat.get('generated_answer', ''))
		else:
			feat = extractor.encode_answer(context, question, answer)
			answer_text = answer
		ans_emb = feat['answer_embeddings']
		ans_offsets = [feat['offset_mapping'].tolist()[i] for i in feat['answer_token_indices'].tolist()]
		inputs = ans_emb.to(device=preferred_device, dtype=enc_dtype).unsqueeze(0)
		logits = encoder(inputs)
		probs = torch.sigmoid(logits).squeeze(0).cpu().tolist()

	if teacher_use_ctxq:
		teacher_input = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n{answer_text}"
	else:
		teacher_input = answer_text
	teacher_spans = teacher.predict_spans(teacher_input)
	embedded_spans = tokens_to_spans(ans_offsets, probs, threshold)

	return {
		'answer': answer_text,
		'teacher_spans': teacher_spans,
		'embedded_spans': embedded_spans,
		'answer_html_teacher': highlight_html(answer_text, teacher_spans, '#ffd1a4'),
		'answer_html_embedded': highlight_html(answer_text, embedded_spans, '#b3e6ff'),
	}


if __name__ == '__main__':
	pass
