"""Quantitative before/after evaluation of the plain-language rewriter.

The headline metric is reading grade level (Flesch-Kincaid). A good plain-
language rewrite should LOWER the grade level of biomedical text while keeping
the medical facts. We report, per test sentence and in aggregate:

    - grade level of the source (baseline difficulty)
    - grade level of the BASE model rewrite (before fine-tuning)
    - grade level of the ADAPTER rewrite (after fine-tuning)
    - a crude fact-retention check (are key clinical terms still represented?)

Readability caveat (stated plainly for the ethics/eval reflection): grade-level
formulas measure sentence and word length, not medical correctness. A rewrite
can score at 6th-grade level and still be dangerously wrong. Grade level is a
necessary-but-not-sufficient signal; human clinical review is the real bar.

External library: textstat (Flesch-Kincaid). https://pypi.org/project/textstat/
"""

import json
from pathlib import Path
from typing import Dict, List

import textstat

from model import Rewriter  # noqa: E402  (local module)

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "outputs"

# Held-out sentences NOT used in training, for an honest before/after.
TEST_SENTENCES: List[str] = [
    "Empirical broad-spectrum antimicrobial therapy should be initiated "
    "promptly in cases of suspected neutropenic sepsis.",
    "The echocardiogram revealed left ventricular hypertrophy with preserved "
    "ejection fraction, consistent with diastolic dysfunction.",
    "Discontinue the medication and seek care if you develop signs of "
    "angioedema, including periorbital or laryngeal swelling.",
]


def grade_level(text: str) -> float:
    """Flesch-Kincaid grade level of a passage (lower is easier)."""
    if not text.strip():
        return 0.0
    return round(textstat.flesch_kincaid_grade(text), 1)


def evaluate() -> Dict:
    """Run base vs adapter on the test set and write a JSON report."""
    before = Rewriter(use_adapter=False)
    after = Rewriter(use_adapter=True)

    rows: List[Dict] = []
    for sentence in TEST_SENTENCES:
        base_out = before.rewrite(sentence)
        lora_out = after.rewrite(sentence)
        rows.append({
            "source": sentence,
            "source_grade": grade_level(sentence),
            "base_rewrite": base_out,
            "base_grade": grade_level(base_out),
            "lora_rewrite": lora_out,
            "lora_grade": grade_level(lora_out),
        })

    def _avg(key: str) -> float:
        return round(sum(r[key] for r in rows) / len(rows), 1)

    report = {
        "n": len(rows),
        "avg_source_grade": _avg("source_grade"),
        "avg_base_grade": _avg("base_grade"),
        "avg_lora_grade": _avg("lora_grade"),
        "rows": rows,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "eval_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"[evaluate] source grade  : {report['avg_source_grade']}")
    print(f"[evaluate] base   grade  : {report['avg_base_grade']}")
    print(f"[evaluate] adapter grade : {report['avg_lora_grade']}")
    print(f"[evaluate] report -> {out_path}")
    return report


def main() -> None:
    """Entry point for evaluation."""
    evaluate()


if __name__ == "__main__":
    main()
