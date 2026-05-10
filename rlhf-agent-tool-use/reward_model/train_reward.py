"""Train the RewardModel on human feedback and preference pairs.

Data sources
────────────
1. data/feedback.csv — human thumbs_up / thumbs_down on model predictions
     thumbs_up   → label 1.0  (model chose correctly for this query)
     thumbs_down → label 0.0  (model chose poorly for this query)

2. data/preference_pairs.jsonl — {prompt, chosen, rejected} triples
     chosen  tool → label 1.0
     rejected tool → label 0.0

3. Synthetic balanced samples — generated at runtime (no external file needed)
     10 clear positive (query, correct_tool) pairs per tool  → label 1.0
     10 clear negative (query, wrong_tool)   pairs per tool  → label 0.0
     Total: 80 synthetic samples, perfectly balanced (40 pos / 40 neg)

Real feedback data is often small and imbalanced; synthetic samples provide a
stable base signal so the reward model always has enough data to train from.
Real data is merged on top and takes precedence (via deduplication last-wins).

Training uses MSELoss (regression), not CrossEntropyLoss — the model outputs
a continuous quality score, not a class probability distribution.

Run from the project root:
    python reward_model/train_reward.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from reward_model.reward_scorer import RewardModel, TOOLS  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
FEEDBACK_PATH = PROJECT_ROOT / "data" / "feedback.csv"
PAIRS_PATH = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
SAVE_PATH = PROJECT_ROOT / "models" / "reward_model"
STATS_PATH = PROJECT_ROOT / "data" / "reward_model_stats.json"

LR = 5e-6
EPOCHS = 15
BATCH_SIZE = 8
MAX_LEN = 128

EXAMPLE_QUERIES = [
    "Where is my order #1234?",
    "What is your return policy?",
    "I need to open a complaint ticket",
    "Can I get a refund for my damaged item?",
]

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ── PyTorch Dataset ───────────────────────────────────────────────────────────

class RewardDataset(Dataset):
    """Tokenises (query, tool) pairs for the regression training loop."""

    def __init__(self, samples: list[dict], tokenizer, max_len: int = MAX_LEN) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = f"{sample['query']} [SEP] {sample['tool']}"
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float32),
        }


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_feedback_samples() -> list[dict]:
    """Convert feedback.csv rows into {query, tool, label} dicts."""
    samples: list[dict] = []
    try:
        df = pd.read_csv(FEEDBACK_PATH)
        required = {"query", "selected_tool", "rating"}
        if not required.issubset(df.columns):
            print(f"[train_reward] feedback.csv missing columns: {required - set(df.columns)}")
            return samples
        for _, row in df.iterrows():
            label = 1.0 if str(row["rating"]).strip() == "thumbs_up" else 0.0
            samples.append({
                "query": str(row["query"]).strip(),
                "tool": str(row["selected_tool"]).strip(),
                "label": label,
            })
    except FileNotFoundError:
        print("[train_reward] data/feedback.csv not found — skipping.")
    except pd.errors.EmptyDataError:
        print("[train_reward] data/feedback.csv is empty — skipping.")
    return samples


def _load_preference_samples() -> list[dict]:
    """Convert preference_pairs.jsonl into positive + negative {query, tool, label} dicts."""
    samples: list[dict] = []
    try:
        with open(PAIRS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                query = str(pair.get("prompt", "")).strip()
                chosen = str(pair.get("chosen", "")).strip()
                rejected = str(pair.get("rejected", "")).strip()
                if not query or chosen not in TOOLS or rejected not in TOOLS:
                    continue
                samples.append({"query": query, "tool": chosen, "label": 1.0})
                samples.append({"query": query, "tool": rejected, "label": 0.0})
    except FileNotFoundError:
        print("[train_reward] data/preference_pairs.jsonl not found — skipping.")
    return samples


def _deduplicate(samples: list[dict]) -> list[dict]:
    """Remove duplicate (query, tool) keys. When duplicates exist, last label wins."""
    seen: dict[tuple[str, str], dict] = {}
    for s in samples:
        seen[(s["query"], s["tool"])] = s
    return list(seen.values())


# ── Synthetic balanced data ───────────────────────────────────────────────────

# 10 representative queries per tool — chosen to be unambiguous so the model
# learns a clear signal even before any real feedback is collected.
_POSITIVE_QUERIES: dict[str, list[str]] = {
    "faq_search": [
        "What are your business hours?",
        "Do you offer free shipping on orders?",
        "What payment methods do you accept?",
        "How long does standard shipping take?",
        "What is your privacy policy?",
        "Do you ship internationally?",
        "What are your customer support hours?",
        "How do I apply a discount code at checkout?",
        "What products and categories do you carry?",
        "Is there a loyalty or rewards program?",
    ],
    "order_status": [
        "Where is my order #1234?",
        "Can you track my package ORD-5678?",
        "My order #9012 hasn't arrived — can you check?",
        "What is the delivery status of order 3456?",
        "Has order #7890 shipped yet?",
        "When will order #2345 be delivered?",
        "I want to check on my most recent order",
        "My tracking number TRK-123 shows no update",
        "Order ORD-6789 still says processing — when does it ship?",
        "Can I get an update on order number 1111?",
    ],
    "refund_policy": [
        "What is your return policy?",
        "Can I get a refund for my purchase?",
        "How do I return an item I bought?",
        "What is the deadline to request a refund?",
        "Do you accept returns after 30 days?",
        "Can I exchange a product for a different size?",
        "I want to return something I bought last week",
        "How long does a refund take to appear on my card?",
        "Do you offer store credit instead of a cash refund?",
        "Can I return a sale item for a full refund?",
    ],
    "raise_ticket": [
        "My package arrived completely damaged",
        "I received the wrong item in my shipment",
        "I need to file an urgent complaint about my order",
        "My product stopped working after just one day",
        "I have an urgent issue that needs immediate help",
        "The item I received is broken and unusable",
        "I need to escalate a problem to your support team",
        "I have been waiting three weeks and nothing has arrived",
        "I was charged twice for the same order — please fix this",
        "Please open a support ticket for my ongoing problem",
    ],
}

# Negative pairings: queries clearly about tool A, wrongly paired with tool B.
# We rotate through the other three tools' query lists (3–4 queries each) to
# reach 10 negatives per tool while keeping variety.
def _build_synthetic_samples() -> list[dict]:
    """Generate 10 positives + 10 negatives for each of the four tools.

    Positives: (query_about_T, T)           → label 1.0
    Negatives: (query_about_other_tool, T)  → label 0.0

    Returns 80 samples total: 40 positive, 40 negative.
    """
    samples: list[dict] = []
    tool_list = list(_POSITIVE_QUERIES.keys())

    for tool in tool_list:
        # ── Positives ──────────────────────────────────────────────
        for query in _POSITIVE_QUERIES[tool]:
            samples.append({"query": query, "tool": tool, "label": 1.0})

        # ── Negatives — use queries from the other three tools ─────
        other_tools = [t for t in tool_list if t != tool]
        neg_queries: list[str] = []
        # Take 3–4 queries from each other tool to reach exactly 10
        per_other = 10 // len(other_tools)       # 3
        remainder = 10 % len(other_tools)        # 1
        for i, other in enumerate(other_tools):
            take = per_other + (1 if i < remainder else 0)
            neg_queries.extend(_POSITIVE_QUERIES[other][:take])

        for query in neg_queries:
            samples.append({"query": query, "tool": tool, "label": 0.0})

    return samples


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model: RewardModel, samples: list[dict]) -> dict:
    """Compute MSE, Pearson r, and discrimination gap on a held-out set."""
    preds, labels = [], []
    for s in samples:
        preds.append(model.score(s["query"], s["tool"]))
        labels.append(s["label"])

    n = len(labels)
    mse = sum((p - l) ** 2 for p, l in zip(preds, labels)) / n

    # Pearson r — skip if all labels identical (undefined correlation)
    label_set = set(labels)
    if len(label_set) > 1:
        mean_l = sum(labels) / n
        mean_p = sum(preds) / n
        cov = sum((l - mean_l) * (p - mean_p) for l, p in zip(labels, preds)) / n
        std_l = (sum((l - mean_l) ** 2 for l in labels) / n) ** 0.5
        std_p = (sum((p - mean_p) ** 2 for p in preds) / n) ** 0.5
        pearson_r = cov / (std_l * std_p) if std_l * std_p > 0 else float("nan")
    else:
        pearson_r = float("nan")

    correct_scores = [p for p, l in zip(preds, labels) if l == 1.0]
    wrong_scores = [p for p, l in zip(preds, labels) if l == 0.0]
    avg_correct = sum(correct_scores) / len(correct_scores) if correct_scores else float("nan")
    avg_wrong = sum(wrong_scores) / len(wrong_scores) if wrong_scores else float("nan")

    def _is_nan(v: float) -> bool:
        return v != v  # NaN check without math import

    gap = (avg_correct - avg_wrong) if not (_is_nan(avg_correct) or _is_nan(avg_wrong)) else float("nan")

    return {
        "mse": round(mse, 4),
        "pearson_r": round(pearson_r, 4) if not _is_nan(pearson_r) else pearson_r,
        "avg_correct": round(avg_correct, 4) if not _is_nan(avg_correct) else avg_correct,
        "avg_wrong": round(avg_wrong, 4) if not _is_nan(avg_wrong) else avg_wrong,
        "gap": round(gap, 4) if not _is_nan(gap) else gap,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n" + "=" * 60)
    print("  Reward Model Training")
    print("=" * 60)

    # ── 1. Load and merge data ────────────────────────────────────────────
    print("\n[1/5] Loading training data ...")

    # Synthetic samples are always generated first as the base layer.
    # Real data is appended on top; deduplication ensures real labels win
    # on any key collision (real feedback overrides synthetic labels).
    synthetic_samples = _build_synthetic_samples()
    feedback_samples = _load_feedback_samples()
    preference_samples = _load_preference_samples()
    real_count = len(feedback_samples) + len(preference_samples)

    syn_pos = sum(1 for s in synthetic_samples if s["label"] == 1.0)
    syn_neg = len(synthetic_samples) - syn_pos
    print(f"      Synthetic samples       : {len(synthetic_samples)}  ({syn_pos} pos / {syn_neg} neg)")
    print(f"      feedback.csv samples    : {len(feedback_samples)}")
    print(f"      preference_pairs samples: {len(preference_samples)}")
    print(f"      Real samples (total)    : {real_count}")

    # Synthetic first so real data wins dedup conflicts
    all_samples = _deduplicate(synthetic_samples + feedback_samples + preference_samples)
    random.shuffle(all_samples)

    # Synthetic data guarantees ≥ 80 samples — this guard remains as a safety net
    if len(all_samples) < 4:
        print("\n[ERROR] Unexpected: fewer than 4 samples after merge. Aborting.\n")
        return

    pos = sum(1 for s in all_samples if s["label"] == 1.0)
    neg = len(all_samples) - pos
    print(f"\n      Total (after dedup)     : {len(all_samples)}")
    print(f"      Positive (label = 1.0)  : {pos}")
    print(f"      Negative (label = 0.0)  : {neg}")

    # ── 2. Split 80 / 20 ─────────────────────────────────────────────────
    print("\n[2/5] Splitting train / test (80/20) ...")
    train_samples, test_samples = train_test_split(
        all_samples, test_size=0.2, random_state=RANDOM_SEED
    )
    print(f"      Train: {len(train_samples)}  |  Test: {len(test_samples)}")

    # ── 3. Initialise model ───────────────────────────────────────────────
    print("\n[3/5] Initialising reward model ...")
    reward_model = RewardModel().load_base()
    tokenizer = reward_model.tokenizer
    net = reward_model.model

    train_dataset = RewardDataset(train_samples, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(net.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, total_steps // 10),
        num_training_steps=total_steps,
    )
    # MSELoss — not CrossEntropyLoss — because this is regression (score in [0,1])
    loss_fn = nn.MSELoss()

    # ── 4. Training loop ──────────────────────────────────────────────────
    print("\n[4/5] Training ...")
    net.train()
    for epoch in range(1, EPOCHS + 1):
        epoch_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(reward_model.device)
            attention_mask = batch["attention_mask"].to(reward_model.device)
            labels = batch["label"].to(reward_model.device)

            preds = net(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"      Epoch {epoch}/{EPOCHS}  —  Train MSE loss: {avg_loss:.4f}")

    net.eval()

    # ── 5. Save ───────────────────────────────────────────────────────────
    print(f"\n[5/5] Saving model to '{SAVE_PATH}' ...")
    reward_model.save(str(SAVE_PATH))

    # ── Final evaluation ──────────────────────────────────────────────────
    print("\n  Evaluating on test set ...")
    metrics = _evaluate(reward_model, test_samples)

    print("\n" + "=" * 60)
    print("  Reward Model Training Complete")
    print("=" * 60)
    print(f"  Samples used       : {len(all_samples)}  ({real_count} real + {len(synthetic_samples)} synthetic)")
    print(f"  Train/Test split   : {len(train_samples)} / {len(test_samples)}")
    print()
    print(f"  Test MSE Loss      : {metrics['mse']}")
    print(f"  Pearson Correlation: {metrics['pearson_r']}")
    print()
    print("  Discrimination Test:")
    print(f"  Correct tool avg score  : {metrics['avg_correct']}")
    print(f"  Wrong tool avg score    : {metrics['avg_wrong']}")
    print(f"  Gap                     : {metrics['gap']}  ← bigger is better")

    # ── Save stats JSON for /api/reward-model/stats endpoint ──────────────
    _nan_to_none = lambda v: None if (isinstance(v, float) and v != v) else v
    gap = metrics.get("gap", float("nan"))
    if isinstance(gap, float) and gap != gap:
        note = "Insufficient data to compute discrimination gap."
    elif gap < 0.1:
        note = "Reward model loaded. Gap < 0.1 indicates limited discrimination — collect more real feedback."
    elif gap < 0.3:
        note = "Moderate discrimination. Continue collecting feedback to improve."
    else:
        note = "Good discrimination. Reward model is performing well."

    reward_stats = {
        "trained": True,
        "pearson_correlation": _nan_to_none(metrics["pearson_r"]),
        "discrimination_gap": _nan_to_none(gap),
        "correct_tool_avg_score": _nan_to_none(metrics["avg_correct"]),
        "wrong_tool_avg_score": _nan_to_none(metrics["avg_wrong"]),
        "samples_used": len(all_samples),
        "real_samples": real_count,
        "synthetic_samples": len(synthetic_samples),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": note,
    }
    STATS_PATH.write_text(json.dumps(reward_stats, indent=2), encoding="utf-8")
    print(f"\n  Stats saved → '{STATS_PATH.name}'")

    # ── Sample scores for a handful of queries ─────────────────────────
    print("\n  Sample scores:")
    for query in EXAMPLE_QUERIES:
        scores = reward_model.score_all_tools(query)
        best_tool = next(iter(scores))  # highest score first (dict is sorted)
        print(f'\n  "{query}"')
        for tool, sc in scores.items():
            mark = "✅" if tool == best_tool else "❌"
            print(f"    {tool:<20} → {sc:.2f} {mark}")

    print()


if __name__ == "__main__":
    main()
