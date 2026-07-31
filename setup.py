"""One-shot project setup: get data, build features, train the adapter.

Run on a GPU machine (Colab T4 is enough):
    python setup.py

Steps:
    1. make_dataset  -> data/raw/plaba.jsonl
    2. build_features -> data/processed/{train,val}.jsonl
    3. model.train   -> models/plainmed-lora/

Inference-only environments (the deployed app) do NOT run this; they load the
already-trained adapter committed to models/plainmed-lora/.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent / "scripts"))


def main() -> None:
    """Run data preparation and training end to end."""
    import make_dataset
    import build_features
    import model

    print("== 1/3 make_dataset ==")
    make_dataset.main()
    print("== 2/3 build_features ==")
    build_features.main()
    print("== 3/3 train ==")
    model.train()
    print("Done. Adapter in models/plainmed-lora/")


if __name__ == "__main__":
    main()
