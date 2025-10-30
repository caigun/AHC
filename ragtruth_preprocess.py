import os
import json
import random
import argparse
from typing import Dict, List, Any, Tuple

import tqdm


def load_response_jsonl(path: str) -> List[Dict[str, Any]]:
	items: List[Dict[str, Any]] = []
	with open(path, 'r', encoding='utf-8') as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			items.append(json.loads(line))
	return items


def normalize_example(raw: Dict[str, Any]) -> Dict[str, Any]:
	# Map RAGTruth fields to a consistent schema
	# Expected fields in response.jsonl entries include:
	# id, source_id, model, temperature, labels (list of dict spans),
	# task_type, source, source_info, prompt, output/response
	labels = raw.get('labels', []) or []
	# Normalize label spans (start, end, label_type)
	norm_labels: List[Dict[str, Any]] = []
	for lab in labels:
		start = lab.get('start')
		end = lab.get('end')
		text = lab.get('text')
		label_type = lab.get('label_type')
		if start is None or end is None:
			continue
		norm_labels.append({
			'start': int(start),
			'end': int(end),
			'text': text,
			'label_type': label_type,
		})

	return {
		'id': raw.get('id'),
		'source_id': raw.get('source_id'),
		'model': raw.get('model'),
		'temperature': raw.get('temperature'),
		'task_type': raw.get('task_type'),
		'source': raw.get('source'),
		'source_info': raw.get('source_info'),
		'prompt': raw.get('prompt'),
		'answer': raw.get('output') or raw.get('response') or raw.get('answer'),
		'labels': norm_labels,
	}


def train_val_test_split(items: List[Dict[str, Any]],
						   seed: int = 42,
						   ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1)) -> Tuple[List, List, List]:
	random.Random(seed).shuffle(items)
	train_r, val_r, test_r = ratios
	n = len(items)
	train_n = int(n * train_r)
	val_n = int(n * val_r)
	train_set = items[:train_n]
	val_set = items[train_n:train_n + val_n]
	test_set = items[train_n + val_n:]
	return train_set, val_set, test_set


def ensure_dir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def save_jsonl(path: str, items: List[Dict[str, Any]]) -> None:
	with open(path, 'w', encoding='utf-8') as f:
		for ex in items:
			f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--ragtruth_dir', type=str, default='./RAGTruth')
	parser.add_argument('--output_dir', type=str, default='./data/processed')
	parser.add_argument('--response_file', type=str, default='dataset/response.jsonl')
	parser.add_argument('--seed', type=int, default=42)
	args = parser.parse_args()

	response_path = os.path.join(args.ragtruth_dir, args.response_file)
	if not os.path.exists(response_path):
		raise FileNotFoundError(f"Could not find response.jsonl at {response_path}")

	print(f"Loading RAGTruth from {response_path} ...")
	raw_items = load_response_jsonl(response_path)

	normalized: List[Dict[str, Any]] = []
	for raw in tqdm.tqdm(raw_items, desc='Normalize'):
		ex = normalize_example(raw)
		# Filter bad examples with missing answer text
		if not ex.get('answer'):
			continue
		normalized.append(ex)

	print(f"Total normalized examples: {len(normalized)}")
	ensure_dir(args.output_dir)

	train_set, val_set, test_set = train_val_test_split(normalized, seed=args.seed)
	save_jsonl(os.path.join(args.output_dir, 'train.jsonl'), train_set)
	save_jsonl(os.path.join(args.output_dir, 'val.jsonl'), val_set)
	save_jsonl(os.path.join(args.output_dir, 'test.jsonl'), test_set)
	print("Saved splits to", args.output_dir)


if __name__ == '__main__':
	main()
