"""Evaluate and compare the SFT baseline router against the DPO fine-tuned router.

Prints overall accuracy, per-tool accuracy, and a confusion matrix for each
available model. If the DPO model does not exist yet, only the SFT results
are shown.

Run from the project root:
    python evaluation/compare_baseline_vs_dpo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from router.tool_router import TOOLS, ToolRouter  # noqa: E402

# ------------------------------------------------------------------ #
# Config — must match training/train_sft.py exactly                   #
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
TEST_SIZE = 0.2
DATA_PATH = PROJECT_ROOT / "data" / "queries.csv"
SFT_MODEL_PATH = PROJECT_ROOT / "models" / "sft_router"
DPO_MODEL_PATH = PROJECT_ROOT / "models" / "dpo_router"

# Short labels for confusion matrix columns
SHORT = {
    "faq_search": "faq",
    "order_status": "order",
    "refund_policy": "refund",
    "raise_ticket": "ticket",
}


# ------------------------------------------------------------------ #
# Data helpers                                                         #
# ------------------------------------------------------------------ #
def load_test_set() -> tuple[list[str], list[str]]:
    """Reproduce the exact test split used during SFT training.

    Returns:
        (queries, true_tool_names) — both as plain Python lists.
    """
    df = pd.read_csv(DATA_PATH)
    _, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=df["correct_tool"],
    )
    return test_df["query"].tolist(), test_df["correct_tool"].tolist()


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #
def run_predictions(
    router: ToolRouter, queries: list[str]
) -> list[str]:
    predictions = []
    for i, query in enumerate(queries, 1):
        pred = router.predict(query)
        predictions.append(pred)
        if i % 4 == 0 or i == len(queries):
            print(f"    Evaluated {i}/{len(queries)} queries ...", end="\r")
    print()
    return predictions


def per_tool_accuracy(y_true: list[str], y_pred: list[str]) -> dict[str, float]:
    result = {}
    for tool in TOOLS:
        indices = [i for i, y in enumerate(y_true) if y == tool]
        if not indices:
            result[tool] = 0.0
            continue
        t = [y_true[i] for i in indices]
        p = [y_pred[i] for i in indices]
        result[tool] = accuracy_score(t, p)
    return result


# ------------------------------------------------------------------ #
# Printing                                                             #
# ------------------------------------------------------------------ #
def _bar(value: float, width: int = 20) -> str:
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def print_report(
    model_name: str,
    y_true: list[str],
    y_pred: list[str],
) -> dict:
    overall = accuracy_score(y_true, y_pred)
    pt_acc = per_tool_accuracy(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=TOOLS)

    print(f"\n{'═' * 62}")
    print(f"  {model_name}")
    print(f"{'═' * 62}")
    print(f"  Overall accuracy : {overall * 100:.1f}%  ({sum(p == t for p, t in zip(y_pred, y_true))}/{len(y_true)} correct)\n")

    print("  Per-tool accuracy:")
    for tool in TOOLS:
        acc = pt_acc[tool]
        print(f"    {tool:<22} {_bar(acc)}  {acc * 100:.0f}%")

    # Confusion matrix
    col_w = 9
    print("\n  Confusion matrix  (rows = true label, cols = predicted):")
    header = f"    {'':22}" + "".join(f"{SHORT[t]:>{col_w}}" for t in TOOLS)
    print(header)
    print(f"    {'':22}" + "-" * (col_w * len(TOOLS)))
    for i, tool in enumerate(TOOLS):
        row_cells = "".join(f"{cm[i][j]:>{col_w}}" for j in range(len(TOOLS)))
        print(f"    {tool:<22}{row_cells}")

    # sklearn summary
    print("\n  Classification report:")
    report = classification_report(
        y_true, y_pred, labels=TOOLS, target_names=TOOLS, zero_division=0
    )
    for line in report.splitlines():
        print(f"    {line}")

    return {"overall": overall, "per_tool": pt_acc}


def print_delta(sft: dict, dpo: dict) -> None:
    print(f"\n{'═' * 62}")
    print("  Improvement: SFT baseline  →  DPO fine-tuned")
    print(f"{'═' * 62}")

    delta = dpo["overall"] - sft["overall"]
    sign = "+" if delta >= 0 else ""
    arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    print(f"  Overall  {sft['overall']*100:.1f}%  →  {dpo['overall']*100:.1f}%  "
          f"{arrow} {sign}{delta*100:.1f}%\n")

    print("  Per-tool:")
    for tool in TOOLS:
        s = sft["per_tool"][tool]
        d = dpo["per_tool"][tool]
        diff = d - s
        sign = "+" if diff >= 0 else ""
        arrow = "▲" if diff > 0.0001 else ("▼" if diff < -0.0001 else "─")
        print(f"    {tool:<22} {s*100:.0f}%  →  {d*100:.0f}%  {arrow} {sign}{diff*100:.0f}%")


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #
def main() -> None:
    print("=" * 62)
    print("  ToolRouter Evaluation: SFT Baseline vs DPO Fine-tuned")
    print("=" * 62)

    print(f"\nLoading test split from '{DATA_PATH.name}' ...")
    queries, y_true = load_test_set()
    print(f"Test set size : {len(queries)} samples")
    print("Label counts  :")
    for tool in TOOLS:
        n = y_true.count(tool)
        print(f"  {tool:<22} {n}")

    sft_metrics = None
    dpo_metrics = None

    # ── SFT baseline ──────────────────────────────────────────────
    print("\n--- Evaluating: SFT Baseline ---")
    try:
        sft_router = ToolRouter().load(str(SFT_MODEL_PATH))
        y_pred_sft = run_predictions(sft_router, queries)
        sft_metrics = print_report("SFT Baseline", y_true, y_pred_sft)
    except FileNotFoundError as exc:
        print(f"  [SKIP] {exc}")

    # ── DPO model ─────────────────────────────────────────────────
    print("\n--- Evaluating: DPO Fine-tuned ---")
    try:
        dpo_router = ToolRouter().load(str(DPO_MODEL_PATH))
        y_pred_dpo = run_predictions(dpo_router, queries)
        dpo_metrics = print_report("DPO Fine-tuned", y_true, y_pred_dpo)
    except FileNotFoundError:
        print("  [SKIP] DPO model not found.")
        print("         Run `python training/run_dpo.py` to generate it.")

    # ── Delta ─────────────────────────────────────────────────────
    if sft_metrics and dpo_metrics:
        print_delta(sft_metrics, dpo_metrics)

    print(f"\n{'=' * 62}")
    print("  Evaluation complete.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    main()
