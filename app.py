"""PlainMed Gradio app for Hugging Face ZeroGPU.

Paste clinical text, get a plain-language rewrite. Shows the fine-tuned output
and the base-model output side by side, each with its Flesch-Kincaid reading
grade level, so the learned capability is visible live.

Deployment: Gradio SDK Space on ZeroGPU (the free tier for personal accounts).
GPU is requested per call via the @spaces.GPU decorator and released after,
which is what keeps the Space free within the daily quota.

The `spaces` import and decorator are guarded so this same file also runs
locally or on a plain CPU host (the decorator becomes a no-op there).
"""

from pathlib import Path
from typing import Tuple

import torch
import textstat
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

try:  # available only on HF Spaces; no-op elsewhere so the file still runs
    import spaces

    gpu_decorator = spaces.GPU(duration=120)
except Exception:  # noqa: BLE001

    def gpu_decorator(func):
        """Fallback decorator when the spaces module is unavailable."""
        return func


BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parent / "models" / "plainmed-lora"

SYSTEM_PROMPT = (
    "You are a medical translator. Rewrite the user's clinical or biomedical "
    "text in plain language a patient with no medical background can "
    "understand. Keep every medical fact accurate. Do not add information, "
    "advice, reassurance, or diagnoses that are not in the original text. Aim "
    "for a sixth-grade reading level. Return only the rewritten text."
)

# Load tokenizer and both models once at import, on CPU. They are moved to the
# GPU inside the decorated function, where ZeroGPU has attached one.
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)

_tuned = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=torch.float16)
if ADAPTER_DIR.exists():
    from peft import PeftModel

    _tuned = PeftModel.from_pretrained(_tuned, str(ADAPTER_DIR)).merge_and_unload()
tuned_model = _tuned

base_model.eval()
tuned_model.eval()


def _grade(text: str) -> float:
    """Flesch-Kincaid grade level, guarded against empty strings."""
    return round(textstat.flesch_kincaid_grade(text), 1) if text.strip() else 0.0


def _generate(model, text: str, device: str, max_new_tokens: int = 256) -> str:
    """Run one greedy generation for the given model on the given device."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@gpu_decorator
def rewrite(text: str) -> Tuple[str, str, str]:
    """Return (fine-tuned rewrite, base rewrite, grade-level summary)."""
    if not text.strip():
        return "", "", "Enter some clinical text above."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tuned_model.to(device)
    base_model.to(device)

    tuned_out = _generate(tuned_model, text, device)
    base_out = _generate(base_model, text, device)

    summary = (
        f"Reading grade level — original: {_grade(text)}  |  "
        f"fine-tuned: {_grade(tuned_out)}  |  base: {_grade(base_out)}  "
        f"(lower is easier to read)"
    )
    return tuned_out, base_out, summary


EXAMPLES = [
    "Serial troponin measurements were obtained to rule out myocardial "
    "infarction in the setting of atypical chest pain.",
    "The patient exhibited postprandial hyperglycemia refractory to oral "
    "hypoglycemic agents.",
    "Prophylactic anticoagulation is indicated post-operatively to mitigate "
    "thromboembolic risk.",
]

with gr.Blocks(title="PlainMed", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# 🩺 PlainMed\n"
        "Rewrites clinical text into plain language. Fine-tuned "
        "Qwen2.5-1.5B (LoRA). The fine-tuned and base outputs are shown "
        "side by side so you can see what the model learned."
    )
    inp = gr.Textbox(
        label="Clinical or biomedical text",
        placeholder="Paste a sentence from a note, discharge summary, or "
                    "research abstract...",
        lines=3,
    )
    btn = gr.Button("Rewrite", variant="primary")
    grades = gr.Markdown()
    with gr.Row():
        tuned_box = gr.Textbox(label="Plain language (fine-tuned)", lines=5)
        base_box = gr.Textbox(label="Base model (before fine-tuning)", lines=5)
    gr.Examples(examples=EXAMPLES, inputs=inp)
    gr.Markdown(
        "_Educational demo only. Not medical advice. Outputs may be "
        "inaccurate and must not be used for clinical decisions. Reading "
        "grade is Flesch-Kincaid, which measures word and sentence length, "
        "not medical correctness._"
    )

    btn.click(fn=rewrite, inputs=inp, outputs=[tuned_box, base_box, grades])


if __name__ == "__main__":
    demo.launch()
