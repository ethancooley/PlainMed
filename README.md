---
title: PlainMed
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# Plain-Med — Plain-Language Medical Rewriter

Fine-tunes a small LLM to translate clinical and biomedical text into plain
language a patient can understand. Built for the AIPI generative-modeling
hackathon.

- **Base model:** Qwen2.5-1.5B-Instruct
- **Adaptation strategy:** LoRA (PEFT) — only a ~40 MB low-rank adapter is
  trained; the base weights stay frozen.
- **Task:** biomedical sentence in → plain-language rewrite out, targeting a
  6th-grade reading level while preserving medical facts.
- **Live app:** https://huggingface.co/spaces/ecooley/PlainMed
- **Repo:** (https://github.com/ethancooley/Plain-Med)

## What the model learned (before / after)

The learned capability is measured with Flesch-Kincaid reading grade level on a
held-out test set (`scripts/evaluate.py`). Lower is easier to read.

| | Avg. reading grade level |
|---|---|
| Source biomedical text | ~15–17 (graduate) |
| Base model rewrite (before) | varies; often still clinical |
| **Fine-tuned rewrite (after)** | **~6–8 (target)** |

Run it yourself: `python scripts/evaluate.py` writes a per-sentence report to
`data/outputs/eval_report.json`. The deployed app shows the same before/after
live — every rewrite displays the source, the fine-tuned output, and the base
output, each with its grade level.

> Readability is a necessary but not sufficient signal. A rewrite can hit a
> 6th-grade score and still be medically wrong. See the ethics note below.

## Quickstart

### Train (GPU — free Colab T4 is enough)
```bash
pip install -r requirements.txt
pip install trl datasets scikit-learn   # training-only deps
python setup.py                         # data -> features -> LoRA train
```
This produces `models/plainmed-lora/`.

### Serve (CPU)
```bash
python main.py            # http://localhost:7860
```

### Try one rewrite from the CLI
```bash
python scripts/model.py predict "Administer the anticoagulant with caution."
python scripts/model.py predict "..." --base   # base model, for comparison
```

## Deployment

Deployed as a **Gradio Space on Hugging Face ZeroGPU** — the free tier for
personal accounts in good standing (up to 2 ZeroGPU Spaces free). ZeroGPU
attaches an H200 GPU per request via the `@spaces.GPU` decorator in `app.py`
and releases it after, which keeps the Space free within the daily GPU quota.
Because a GPU is available per call, both the fine-tuned and base models load
and the before/after comparison runs live.

(Render's free tier — 512 MB — cannot hold a 1.5B model, and HF Docker/CPU-Basic
Spaces are no longer free, which is why ZeroGPU is the path here.)

To deploy:
1. Create a new **Gradio** Space under your HF account. On a free account the
   runtime will be ZeroGPU automatically.
2. Push this repo to the Space. `app.py` is the entry point; the adapter in
   `models/plainmed-lora/` loads at import and is moved to the GPU per call.
   Track the adapter with Git LFS if it exceeds 10 MB (see below).
3. First request warms the models; later ones are fast on the GPU.

`app.py` guards the `spaces` import, so the same file also runs locally
(`python app.py`) on CPU or a local GPU. A FastAPI variant is kept in
`app/server.py` for non-Gradio hosting.

## Data

Primary source: **PLABA** (Plain Language Adaptation of Biomedical Abstracts,
U.S. National Library of Medicine) — professionally written source→plain
sentence pairs. `scripts/make_dataset.py` pulls it and falls back to a small
hand-written seed set if it is unavailable, so the pipeline always runs.
Verify PLABA's current access terms before relying on it.

## Project structure
```
README.md
requirements.txt
setup.py                 <- data -> features -> train
app.py                   <- Gradio app on ZeroGPU (runs inference)
main.py                  <- launch the FastAPI variant locally
Dockerfile               <- optional: FastAPI/Docker hosting
app/
  server.py              <- FastAPI app + HTML UI (alternative host)
scripts/
  make_dataset.py        <- acquire PLABA (seed fallback)
  build_features.py      <- format into chat examples, train/val split
  model.py               <- LoRA train + Rewriter (predict)
  evaluate.py            <- before/after readability report
models/
  plainmed-lora/         <- trained adapter (committed or LFS-tracked)
data/
  raw/ processed/ outputs/
notebooks/               <- exploration only (not graded)
```

## Git LFS (only if the adapter > 10 MB)
```bash
git lfs install
git lfs track "models/plainmed-lora/adapter_model.safetensors"
git add .gitattributes                 # commit BEFORE adding the model file
git add models/plainmed-lora/adapter_model.safetensors
```
LFS tracking must be set up **before** `git add` of the large file.

## Ethics, risks, and evaluation challenges

- **Factual drift is the core risk.** A fluent, easy-to-read rewrite that
  quietly changes a dose, a contraindication, or a hedge ("may" → "will") is
  more dangerous than the original jargon, because it reads as authoritative
  and clear. The system prompt forbids adding information, advice, or
  reassurance, but prompt constraints are not guarantees.
- **False reassurance.** Simplification can strip clinically important
  qualifiers. The model is instructed to preserve them; this needs human
  verification, not just automated scoring.
- **Readability metrics are gameable.** Flesch-Kincaid rewards short words and
  sentences. Optimizing for it alone can produce text that is easy to read and
  wrong. Grade level is reported as one axis; faithfulness needs clinical
  review and ideally a fact-consistency metric.
- **Training data provenance.** If the seed/synthetic fallback is used instead
  of PLABA, the model learns from a tiny, non-representative sample — outputs
  will be unreliable and this must be disclosed.
- **Scope.** Educational demo only. Not a medical device, not medical advice.
  The UI states this on every response.

## Git workflow (branches + PRs)

Development followed a feature-branch + PR workflow (required by the rubric,
including for solo work):

| Branch | PR |
|---|---|
| `feat/data-pipeline` | make_dataset + build_features |
| `feat/training` | LoRA training in model.py |
| `feat/inference-app` | FastAPI app + UI |
| `feat/evaluation` | before/after readability eval |
| `feat/deploy` | Dockerfile + HF Spaces config |

Each branch was opened as a PR into `main`, self-reviewed with a summary and
review comments, then squash-merged.

## Attribution
- Base model: Qwen2.5-1.5B-Instruct (Alibaba / Qwen), via Hugging Face.
- LoRA fine-tuning uses PEFT + the transformers Trainer:
  https://huggingface.co/docs/peft
- PLABA dataset: U.S. National Library of Medicine.
- Readability: `textstat`.
