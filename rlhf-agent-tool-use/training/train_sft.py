"""Supervised Fine-Tuning (SFT) for the ToolRouter.

Trains distilbert-base-uncased on queries.csv to classify which tool
to call for a given customer support query.

Run from the project root:
    python training/train_sft.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

# Make `router` importable when script is run from any directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from router.tool_router import LABEL2ID, TOOLS, ToolRouter  # noqa: E402

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
DATA_PATH = PROJECT_ROOT / "data" / "queries.csv"
MODEL_SAVE_PATH = PROJECT_ROOT / "models" / "sft_router"
CHECKPOINT_DIR = PROJECT_ROOT / "models" / "checkpoints"
LOG_DIR = PROJECT_ROOT / "models" / "logs"

EPOCHS = 3
BATCH_SIZE = 8
MAX_LEN = 128
TEST_SIZE = 0.2


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #
class QueryDataset(Dataset):
    """Tokenised query dataset compatible with HuggingFace Trainer."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        tokenizer,
        max_len: int,
    ) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_len,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ------------------------------------------------------------------ #
# Metrics                                                              #
# ------------------------------------------------------------------ #
def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": float(accuracy_score(labels, preds))}


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 60)
    print("  SFT Training — ToolRouter (distilbert-base-uncased)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────
    print(f"\n[1/5] Loading data from '{DATA_PATH}' ...")
    df = pd.read_csv(DATA_PATH)
    print(f"      Total samples : {len(df)}")
    print("      Label distribution:")
    for tool, count in df["correct_tool"].value_counts().items():
        print(f"        {tool:<22} {count}")

    df["label"] = df["correct_tool"].map(LABEL2ID)
    if df["label"].isna().any():
        unknown = df[df["label"].isna()]["correct_tool"].unique().tolist()
        raise ValueError(f"Unknown tool labels in CSV: {unknown}")

    # ── 2. Train / test split ─────────────────────────────────────
    print(f"\n[2/5] Splitting data ({int((1-TEST_SIZE)*100)}% train / "
          f"{int(TEST_SIZE*100)}% test, stratified) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["query"].tolist(),
        df["label"].tolist(),
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["label"].tolist(),
    )
    print(f"      Train : {len(X_train)} samples")
    print(f"      Test  : {len(X_test)} samples")

    # ── 3. Initialise model ───────────────────────────────────────
    print("\n[3/5] Initialising model ...")
    router = ToolRouter().load_base()

    # ── 4. Tokenise ───────────────────────────────────────────────
    print("\n[4/5] Tokenising datasets ...")
    train_dataset = QueryDataset(X_train, y_train, router.tokenizer, MAX_LEN)
    test_dataset = QueryDataset(X_test, y_test, router.tokenizer, MAX_LEN)
    print(f"      Train dataset : {len(train_dataset)} samples")
    print(f"      Test dataset  : {len(test_dataset)} samples")

    # ── 5. Train ──────────────────────────────────────────────────
    print(f"\n[5/5] Fine-tuning for {EPOCHS} epoch(s) on {router.device} ...")
    print("-" * 60)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_dir=str(LOG_DIR),
        logging_steps=5,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=router.model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()

    # ── Results ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Training complete.")
    print("=" * 60)
    print(f"  Final training loss : {train_result.training_loss:.4f}")

    # Best model is already loaded back by load_best_model_at_end
    router.model = trainer.model

    test_metrics = trainer.evaluate(test_dataset)
    train_metrics = trainer.evaluate(train_dataset)
    print(f"  Train accuracy      : {train_metrics['eval_accuracy']:.4f} "
          f"({train_metrics['eval_accuracy']*100:.1f}%)")
    print(f"  Test accuracy       : {test_metrics['eval_accuracy']:.4f} "
          f"({test_metrics['eval_accuracy']*100:.1f}%)")

    # ── Save ──────────────────────────────────────────────────────
    print()
    router.save(str(MODEL_SAVE_PATH))
    print("\nNext step: run `python evaluation/compare_baseline_vs_dpo.py`")


if __name__ == "__main__":
    main()
