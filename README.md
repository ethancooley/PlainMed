# PlainMed — Plain-Language Medical Rewriter

Fine-tunes a small LLM to translate clinical and biomedical text into plain
language a patient can understand. Built for the AIPI generative-modeling
hackathon.

- **Base model:** Qwen2.5-1.5B-Instruct
- **Adaptation strategy:** LoRA (PEFT) — only a ~40 MB low-rank adapter is
  trained; the base weights stay frozen.
- **Task:** biomedical sentence in → plain-language rewrite out, targeting a
  6th-grade reading level while preserving medical facts.
- **Live app:** _add your deployed URL here_
- **Repo:** _add your GitHub URL here_

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

Deployed on **Hugging Face Spaces** (Docker SDK, free CPU tier — 16 GB RAM,
which comfortably holds the merged 1.5B model; Render's 512 MB free/Starter
tiers do not). The `Dockerfile` serves the FastAPI app on port 7860.

To deploy:
1. Create a Space (Docker SDK) under your HF account.
2. Push this repo to the Space. The adapter in `models/plainmed-lora/` is
   loaded at startup; track it with Git LFS if it exceeds 10 MB (see below).
3. First request is slow on CPU (model load + generation); later ones are
   faster.

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
main.py                  <- launch the app
Dockerfile               <- HF Spaces deployment
app/
  server.py              <- FastAPI app + HTML UI (runs inference)
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
- LoRA/SFT training follows the TRL docs: https://huggingface.co/docs/trl
- PLABA dataset: U.S. National Library of Medicine.
- Readability: `textstat`.
