# Qwen3-8B Embedded Hallucination Detector

A lightweight, embedded hallucination detector that uses the last 2–3 layers of Qwen3-8B to predict hallucination spans, trained to mimic LettuceDetect (teacher) on the RAGTruth dataset.

- Dataset: `RAGTruth` (place at `./RAGTruth`) — see `ParticleMedia/RAGTruth` [GitHub](https://github.com/ParticleMedia/RAGTruth)
- Teacher model: `lettucedetect-large-en` — see `KRLabsOrg/LettuceDetect` [GitHub](https://github.com/KRLabsOrg/LettuceDetect?tab=readme-ov-file)

## Environment

Assumes a conda env named `dl-torch` and that Qwen3-8B is cached in Hugging Face.

```bash
conda activate dl-torch
pip install transformers gradio pyyaml tqdm torch
```

## Configuration

Edit `config.yaml` for model names and paths. Defaults:
- Qwen: `Qwen/Qwen3-8B-Instruct`
- LettuceDetect: `KRLabs/lettucedetect-large-en`

## Data Preprocess

RAGTruth already downloaded under `./RAGTruth`. If needed, regenerate splits:

```bash
python ragtruth_preprocess.py --ragtruth_dir ./RAGTruth --output_dir ./data/processed
```

## Train (Distill to Embedded Encoder)

```bash
python train_encoder.py
```
This trains a small BiLSTM token encoder on Qwen embeddings, with LettuceDetect spans converted to per-token labels.

Checkpoints: `./checkpoints/small_encoder.pt`

## Inference & Gradio UI

- Programmatic compare:
```bash
python -c "from infer_and_compare import run_inference; print(run_inference('France...', 'What is...', 'The capital...'))"
```

- Gradio app:
```bash
python gradio_ui.py
```
Opens a local web app to input context, question, answer, and view highlighted spans from both LettuceDetect and the embedded encoder.

## Citations
- RAGTruth dataset: `ParticleMedia/RAGTruth` [GitHub](https://github.com/ParticleMedia/RAGTruth)
- LettuceDetect: `KRLabsOrg/LettuceDetect` [GitHub](https://github.com/KRLabsOrg/LettuceDetect?tab=readme-ov-file)
