import os
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml


class QwenFeatureExtractor:
	def __init__(self, config_path: str = 'config.yaml'):
		with open(config_path, 'r', encoding='utf-8') as f:
			cfg = yaml.safe_load(f)
		self.cfg = cfg
		model_name = cfg['models']['qwen_model_name']
		trust_remote_code = bool(cfg['models'].get('qwen_trust_remote_code', True))
		self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name,
			trust_remote_code=trust_remote_code,
			output_hidden_states=True,
			torch_dtype=(torch.bfloat16 if cfg['device'].get('dtype', 'bfloat16') == 'bfloat16' else torch.float16),
			device_map='auto'
		)
		self.use_layers: List[int] = cfg['features']['use_layers']
		self.max_answer_tokens: int = int(cfg['features'].get('max_answer_tokens', 512))
		self.max_new_tokens: int = int(cfg['inference'].get('max_new_tokens', 256))

	def _to_device(self, x: torch.Tensor) -> torch.Tensor:
		return x.to(self.model.device)

	def _build_prompt(self, context: str, question: str) -> str:
		return f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:\n"

	def _sampling_kwargs(self) -> Dict:
		inf = self.cfg['inference']
		use_thinking = bool(inf.get('enable_thinking', True))
		params = inf['thinking'] if use_thinking else inf['non_thinking']
		temperature = float(params.get('temperature', 0.7))
		top_p = float(params.get('top_p', 0.9))
		top_k = int(params.get('top_k', 20))
		# min_p is not supported in HF generate; keep for future, ignore if not used
		presence_penalty = float(params.get('presence_penalty', 0.0))
		kwargs = {
			'do_sample': True,
			'temperature': temperature,
			'top_p': top_p,
			'top_k': top_k,
		}
		# Map presence_penalty to repetition_penalty heuristically
		if presence_penalty and presence_penalty > 0:
			kwargs['repetition_penalty'] = 1.0 + min(presence_penalty, 2.0)
		return kwargs

	def encode_answer(self, context: str, question: str, answer: str) -> Dict[str, torch.Tensor]:
		prompt_prefix = self._build_prompt(context, question)
		full_text = prompt_prefix + (answer or '')
		enc = self.tokenizer(full_text, return_offsets_mapping=True, return_tensors='pt', add_special_tokens=True, truncation=True)
		offsets = enc.pop('offset_mapping')
		input_ids = self._to_device(enc['input_ids'])
		attention_mask = self._to_device(enc['attention_mask'])

		with torch.no_grad():
			out = self.model(input_ids=input_ids, attention_mask=attention_mask)
		hidden_states: Tuple[torch.Tensor, ...] = out.hidden_states

		prefix_len = len(prompt_prefix)
		offsets_np = offsets[0].tolist()
		answer_token_indices: List[int] = []
		for idx, (s, e) in enumerate(offsets_np):
			if s is None or e is None:
				continue
			if s >= prefix_len:
				answer_token_indices.append(idx)

		if len(answer_token_indices) > self.max_answer_tokens:
			answer_token_indices = answer_token_indices[:self.max_answer_tokens]

		selected_layers = []
		for rel in self.use_layers:
			layer = hidden_states[rel]
			selected_layers.append(layer)
		concat = torch.cat(selected_layers, dim=-1)
		answer_emb = concat[:, answer_token_indices, :].squeeze(0).contiguous()

		return {
			'answer_embeddings': answer_emb,
			'answer_token_indices': torch.tensor(answer_token_indices, device=answer_emb.device, dtype=torch.long),
			'input_ids': input_ids,
			'attention_mask': attention_mask,
			'full_text': full_text,
			'prompt_prefix_len': torch.tensor(prefix_len),
			'offset_mapping': torch.tensor(offsets_np),
		}

	def generate_and_encode(self, context: str, question: str) -> Dict[str, torch.Tensor]:
		prompt_prefix = self._build_prompt(context, question)
		enc_prompt = self.tokenizer(prompt_prefix, return_tensors='pt', add_special_tokens=True)
		input_ids = self._to_device(enc_prompt['input_ids'])
		attention_mask = self._to_device(enc_prompt['attention_mask'])
		gen_kwargs = dict(
			max_new_tokens=self.max_new_tokens,
			return_dict_in_generate=True,
			early_stopping=True,
		)
		gen_kwargs.update(self._sampling_kwargs())
		with torch.no_grad():
			gen = self.model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				**gen_kwargs
			)
		sequences = gen.sequences
		prompt_len = input_ids.shape[1]
		gen_ids = sequences[:, prompt_len:]
		answer_text = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
		return self.encode_answer(context, question, answer_text) | { 'generated_answer': answer_text }


def get_feature_dim(config_path: str = 'config.yaml') -> int:
	with open(config_path, 'r', encoding='utf-8') as f:
		cfg = yaml.safe_load(f)
	return -1


if __name__ == '__main__':
	pass
