"""Train a LoRA adapter for plain-language medical rewriting, and predict.

Strategy: parameter-efficient fine-tuning (LoRA) on Qwen2.5-1.5B-Instruct.
Only the low-rank adapter matrices are trained; the base weights are frozen.
The resulting adapter is a few tens of MB, which keeps deployment light.

Training is designed for a single free Colab T4 GPU. Inference (predict) runs
on CPU so it can be served on a free Hugging Face Space.

External libraries: transformers, peft, trl, datasets (all standard HF stack).
LoRA/SFT usage follows the TRL documentation: https://huggingface.co/docs/trl
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


def train(epochs: int = 8, lr: float = 2e-4, rank: int = 16) -> Path:
    """Fine-tune a LoRA adapter and save it to models/plainmed-lora.

    Args:
        epochs: number of passes over the training set.
        lr: learning rate for the adapter parameters.
        rank: LoRA rank (capacity of the adapter).

    Returns:
        Path to the saved adapter directory.
    """
    # Local imports keep CPU-only inference environments from needing trl/peft.
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16 if _device() == "cuda" else torch.float32,
        device_map=_device(),
    )

    train_ds = load_dataset(
        "json", data_files=str(PROC_DIR / "train.jsonl"), split="train"
    )
    val_ds = load_dataset(
        "json", data_files=str(PROC_DIR / "val.jsonl"), split="train"
    )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=str(ADAPTER_DIR),
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        bf16=_device() == "cuda",
        report_to="none",
        assistant_only_loss=True,  # train only on the assistant (rewrite) turn
        max_length=1024,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(ADAPTER_DIR))
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
            BASE_MODEL, torch_dtype=torch.float32, device_map=_device()
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
