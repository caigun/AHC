from typing import Dict, List, Any

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TokenClassificationPipeline
import yaml


class LettuceTeacher:
	def __init__(self, config_path: str = 'config.yaml'):
		with open(config_path, 'r', encoding='utf-8') as f:
			cfg = yaml.safe_load(f)
		self.cfg = cfg
		model_name = cfg['models']['lettucedetect_model_name']
		self.tokenizer = AutoTokenizer.from_pretrained(model_name)
		self.model = AutoModelForTokenClassification.from_pretrained(model_name)
		self.pipe = TokenClassificationPipeline(model=self.model, tokenizer=self.tokenizer, aggregation_strategy='simple', device=0 if torch.cuda.is_available() else -1)

	def predict_spans(self, text: str) -> List[Dict[str, Any]]:
		# Returns list of spans with start, end, score, label
		ents = self.pipe(text)
		spans: List[Dict[str, Any]] = []
		for e in ents:
			spans.append({
				'start': int(e['start']),
				'end': int(e['end']),
				'score': float(e.get('score', 1.0)),
				'label': str(e.get('entity_group', e.get('entity', 'HALLUCINATION')))
			})
		return spans

	@staticmethod
	def spans_to_token_labels(spans: List[Dict[str, Any]], offsets: List[List[int]], threshold: float = 0.5) -> torch.Tensor:
		# offsets: list of [start, end] per token
		# returns tensor of shape (seq_len,) with 1 for hallucinated tokens within any span
		seq_len = len(offsets)
		labels = torch.zeros(seq_len, dtype=torch.float32)
		for i, (s_tok, e_tok) in enumerate(offsets):
			for sp in spans:
				if sp['start'] <= s_tok and e_tok <= sp['end']:
					labels[i] = 1.0
					break
		return labels


if __name__ == '__main__':
	pass
