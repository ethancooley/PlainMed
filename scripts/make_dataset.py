"""Acquire and stage the source->plain-language sentence-pair dataset.

Primary source: PLABA (Plain Language Adaptation of Biomedical Abstracts),
a professionally-authored dataset from the U.S. National Library of Medicine.
    Project page: https://osf.io/rnpmf/  (verify current access before relying on it)

If PLABA cannot be reached or its license terms are not accepted, this script
falls back to a small SEED set of hand-written pairs so the repo runs
end-to-end. Training on the seed set alone is NOT representative — it exists to
prove the pipeline, not to produce a strong model. This limitation is called
out in the README and in scripts/evaluate.py output.

Attribution: PLABA dataset (c) National Library of Medicine. Used here for
research/educational purposes under the AIPI generative modeling hackathon.
"""

import json
import os
from pathlib import Path
from typing import List, Dict

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PLABA_JSONL = RAW_DIR / "plaba.jsonl"

# Small seed set for pipeline validation only. Each pair is a biomedical
# sentence (source) and a plain-language adaptation (target).
SEED_PAIRS: List[Dict[str, str]] = [
    {
        "source": "The patient presented with acute exacerbation of chronic "
                  "obstructive pulmonary disease secondary to a lower "
                  "respiratory tract infection.",
        "target": "The patient's long-term lung disease got suddenly worse "
                  "because of a chest infection.",
    },
    {
        "source": "Administration of the anticoagulant is contraindicated in "
                  "patients with a history of intracranial hemorrhage.",
        "target": "This blood-thinning medicine should not be given to people "
                  "who have had bleeding in the brain before.",
    },
    {
        "source": "Myocardial infarction results from occlusion of a coronary "
                  "artery, leading to ischemia of the cardiac muscle.",
        "target": "A heart attack happens when an artery to the heart gets "
                  "blocked, so part of the heart muscle stops getting blood.",
    },
    {
        "source": "The lesion was determined to be a benign neoplasm following "
                  "histopathological examination of the excised tissue.",
        "target": "After testing the removed tissue under a microscope, doctors "
                  "found the growth was not cancer.",
    },
    {
        "source": "Prophylactic antibiotic therapy is recommended prior to "
                  "invasive dental procedures in this population.",
        "target": "People in this group should take antibiotics before dental "
                  "work that breaks the skin, to prevent infection.",
    },
    {
        "source": "Renal function should be monitored via serial serum "
                  "creatinine measurements during the treatment course.",
        "target": "Doctors should check how well the kidneys are working with "
                  "regular blood tests while you take this medicine.",
    },
    {
        "source": "The trial demonstrated a statistically significant reduction "
                  "in all-cause mortality among the intervention cohort.",
        "target": "In the study, fewer people died from any cause in the group "
                  "that got the treatment.",
    },
    {
        "source": "Hypertension is frequently asymptomatic and may remain "
                  "undiagnosed until end-organ damage occurs.",
        "target": "High blood pressure often has no symptoms, so people may not "
                  "know they have it until it has already harmed the body.",
    },
]


def download_plaba() -> List[Dict[str, str]]:
    """Attempt to fetch PLABA. Returns a list of {source, target} pairs.

    This is intentionally defensive: dataset hosting and access terms change.
    On any failure it returns an empty list and the caller falls back to seed.
    """
    try:
        # Placeholder for the real fetch. The PLABA release ships as paired
        # sentences; adapt the loader to whatever format the current release
        # uses (JSON, TSV, or the HuggingFace mirror if one is available).
        # Kept as a guarded stub so a network/license failure never breaks CI.
        from datasets import load_dataset  # local import: optional dependency

        ds = load_dataset("plaba", split="train")  # may raise if unavailable
        pairs = [
            {"source": row["abstract_sentence"], "target": row["plain_sentence"]}
            for row in ds
            if row.get("abstract_sentence") and row.get("plain_sentence")
        ]
        return pairs
    except Exception as exc:  # noqa: BLE001 - deliberately broad for robustness
        print(f"[make_dataset] PLABA unavailable ({exc}). Using seed set.")
        return []


def write_jsonl(pairs: List[Dict[str, str]], path: Path) -> None:
    """Write a list of pair dicts to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")


def main() -> None:
    """Stage the raw dataset to data/raw/plaba.jsonl."""
    pairs = download_plaba()
    if not pairs:
        pairs = SEED_PAIRS
        print(f"[make_dataset] Wrote {len(pairs)} SEED pairs (pipeline demo).")
    else:
        print(f"[make_dataset] Wrote {len(pairs)} PLABA pairs.")
    write_jsonl(pairs, PLABA_JSONL)
    print(f"[make_dataset] -> {PLABA_JSONL}")


if __name__ == "__main__":
    main()
