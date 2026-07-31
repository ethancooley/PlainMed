"""Train a LoRA adapter for plain-language medical rewriting, and predict.

Strategy: parameter-efficient fine-tuning (LoRA) on Qwen2.5-1.5B-Instruct.
Only the low-rank adapter matrices are trained; the base weights are frozen.
The resulting adapter is a few tens of MB, which keeps deployment light.

Training is designed for a single free Colab T4 GPU. Inference (predict) runs
on CPU so it can be served on a free Hugging Face Space.

External libraries: transformers, peft, datasets (standard HF stack).
LoRA usage follows the PEFT documentation: https://huggingface.co/docs/peft
"""

import argparse
from pathlib import Path
from typing import List, Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = Path(__file__).resolve().parents[1] / "models" / "plainmed-lora"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SYSTEM_PROMPT = (
    "You are a medical translator. Rewrite the user's clinical or biomedical "
    "text in plain language a patient with no medical background can "
    "understand. Keep every medical fact accurate. Do not add information, "
    "advice, reassurance, or diagnoses that are not in the original text. Aim "
    "for a sixth-grade reading level. Return only the rewritten text."
)


def _device() -> str:
    """Return the best available torch device string."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def _supports_bf16() -> bool:
    """True only on GPUs with real bf16 support (Ampere / sm_80+).

    Turing cards (e.g. the free Colab T4, sm_75) do NOT support bf16 and will
    error if it is requested, so training must fall back to fp16 there.
    """
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 8


def _train_dtype() -> torch.dtype:
    """Pick the training dtype the current device actually supports."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if _supports_bf16() else torch.float16


def train(epochs: int = 3, lr: float = 2e-4, rank: int = 16,
          max_length: int = 1024) -> Path:
    """Fine-tune a LoRA adapter and save it to models/plainmed-lora.

    Uses the standard transformers Trainer with PEFT. Precision is chosen from
    the actual GPU: bf16 on Ampere+ (sm_80+), fp16 on Turing (e.g. the free
    Colab T4), fp32 on CPU. This avoids the bf16-on-T4 crash.

    Args:
        epochs: passes over the training set. Default 3 suits ~100 pairs; a
            larger dataset can use fewer, a tiny one more.
        lr: learning rate for the adapter parameters.
        rank: LoRA rank (capacity of the adapter).
        max_length: token truncation length for training examples.

    Returns:
        Path to the saved adapter directory.
    """
    # Local imports keep CPU-only inference environments from needing peft here.
    from peft import LoraConfig, get_peft_model
    from transformers import (
        DataCollatorForLanguageModeling, Trainer, TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, dtype=_train_dtype(), device_map=_device()
    )
    model = get_peft_model(model, LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    ))

    def _tokenize(example: Dict) -> Dict:
        """Render the chat template and tokenize one example."""
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False
        )
        return tokenizer(text, truncation=True, max_length=max_length)

    train_ds = load_dataset(
        "json", data_files=str(PROC_DIR / "train.jsonl"), split="train"
    )
    train_ds = train_ds.map(_tokenize, remove_columns=train_ds.column_names)

    use_bf16 = _supports_bf16()
    args = TrainingArguments(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=5,
        save_strategy="no",
        bf16=use_bf16,
        fp16=torch.cuda.is_available() and not use_bf16,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    trainer.train()
    model.save_pretrained(str(ADAPTER_DIR))
    tokenizer.save_pretrained(str(ADAPTER_DIR))
    print(f"[model] Adapter saved -> {ADAPTER_DIR}")
    return ADAPTER_DIR


class Rewriter:
    """Loads the base model (optionally with the LoRA adapter) and rewrites.

    Set use_adapter=False to get the BEFORE (base model) behavior and
    use_adapter=True to get the AFTER (fine-tuned) behavior. This single class
    powers both the app and the before/after evaluation.
    """

    def __init__(self, use_adapter: bool = True) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.float32, device_map=_device()
        )

        if use_adapter and ADAPTER_DIR.exists():
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(ADAPTER_DIR))
            self.model = self.model.merge_and_unload()
        self.model.eval()

    def rewrite(self, text: str, max_new_tokens: int = 256) -> str:
        """Return a plain-language rewrite of the input medical text."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated = output[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def main() -> None:
    """CLI: `python scripts/model.py train` or `... predict "<text>"`."""
    parser = argparse.ArgumentParser(description="PlainMed LoRA train/predict")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    predict_parser = sub.add_parser("predict")
    predict_parser.add_argument("text", type=str)
    predict_parser.add_argument("--base", action="store_true",
                                help="use base model only (before)")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        rewriter = Rewriter(use_adapter=not args.base)
        print(rewriter.rewrite(args.text))


if __name__ == "__main__":
    main()
