"""Turn raw source->plain pairs into instruction-formatted training examples.

Each pair becomes a chat-format example using the base model's chat template:
a system instruction defining the plain-language task, a user turn containing
the biomedical sentence, and an assistant turn containing the plain rewrite.
Only the assistant turn contributes to the loss (handled by the trainer's
completion-only collator in scripts/model.py).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "plaba.jsonl"
PROC_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

SYSTEM_PROMPT = (
    "You are a medical translator. Rewrite the user's clinical or biomedical "
    "text in plain language a patient with no medical background can "
    "understand. Keep every medical fact accurate. Do not add information, "
    "advice, reassurance, or diagnoses that are not in the original text. Aim "
    "for a sixth-grade reading level. Return only the rewritten text."
)


def load_pairs(path: Path) -> List[Dict[str, str]]:
    """Read a JSONL file of {source, target} pairs."""
    pairs: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def to_chat_example(pair: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """Convert one pair into a 'messages' chat example."""
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": pair["source"]},
            {"role": "assistant", "content": pair["target"]},
        ]
    }


def build(test_size: float = 0.15, seed: int = 42) -> Tuple[Path, Path]:
    """Build train/val JSONL files of chat examples. Returns their paths."""
    pairs = load_pairs(RAW_PATH)
    examples = [to_chat_example(p) for p in pairs]

    if len(examples) < 4:
        # Too few to split meaningfully; duplicate into both for a smoke test.
        train_ex, val_ex = examples, examples
    else:
        train_ex, val_ex = train_test_split(
            examples, test_size=test_size, random_state=seed
        )

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    train_path = PROC_DIR / "train.jsonl"
    val_path = PROC_DIR / "val.jsonl"

    for path, data in ((train_path, train_ex), (val_path, val_ex)):
        with open(path, "w", encoding="utf-8") as handle:
            for ex in data:
                handle.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"[build_features] train={len(train_ex)} val={len(val_ex)}")
    return train_path, val_path


def main() -> None:
    """Entry point for the feature build step."""
    build()


if __name__ == "__main__":
    main()
