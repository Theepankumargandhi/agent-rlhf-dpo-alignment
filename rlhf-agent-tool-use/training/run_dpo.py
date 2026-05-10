"""DPO Fine-Tuning for the ToolRouter.

What DPO does
─────────────
Direct Preference Optimization trains a *policy* model to assign higher
probability to the "chosen" label than the "rejected" label for each query,
*relative to a frozen reference model* (the SFT checkpoint).

The loss is:

    L_DPO = -log sigmoid( β · [(log π(c|x) - log π_ref(c|x))
                               - (log π(r|x) - log π_ref(r|x))] )

where c = chosen tool, r = rejected tool, x = query,
π = policy model, π_ref = frozen reference model, β = KL coefficient.

This encourages the policy to prefer chosen over rejected more than the
reference model does — without needing an explicit reward model.

Note on TRL
───────────
TRL's DPOTrainer is designed for causal language models (text generation).
Our ToolRouter uses a sequence classifier (DistilBERT), so we implement the
DPO objective directly on classification logits. We use DPOConfig for
hyperparameter management and a custom PyTorch training loop for the loss.

Run from the project root:
    python training/run_dpo.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from router.tool_router import LABEL2ID, TOOLS, ToolRouter  # noqa: E402

try:
    from trl import DPOConfig
except ImportError:
    raise ImportError(
        "TRL >= 0.8.0 is required for DPOConfig.\n"
        "Install with:  pip install 'trl>=0.8.0'"
    )

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #
RANDOM_SEED = 42
MIN_PAIRS = 3

PAIRS_PATH = PROJECT_ROOT / "data" / "preference_pairs.jsonl"
QUERIES_PATH = PROJECT_ROOT / "data" / "queries.csv"
SFT_MODEL_PATH = PROJECT_ROOT / "models" / "sft_router"
DPO_SAVE_PATH = PROJECT_ROOT / "models" / "dpo_router"
EVAL_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation_results.json"

# β controls how strictly we stay close to the reference model.
# Low β = allow large divergence; high β = stay near SFT weights.
DPO_BETA = 0.1
DPO_LR = 1e-5           # Small LR to avoid catastrophic forgetting
DPO_EPOCHS = 3
BATCH_SIZE = 1
GRAD_ACCUM = 4           # Effective batch size = BATCH_SIZE × GRAD_ACCUM = 4
MAX_LEN = 128

# Prompt wrapper so the model sees a classification-style instruction.
# This is also what gets stored in preference_pairs_formatted if needed.
PROMPT_TEMPLATE = (
    "Classify this customer query into one of these tools: "
    "[faq_search, order_status, refund_policy, raise_ticket]\n\n"
    "Query: {query}\n\n"
    "Respond with only the tool name."
)


# ------------------------------------------------------------------ #
# Dataset                                                              #
# ------------------------------------------------------------------ #

class PreferencePairDataset(Dataset):
    """Tokenised preference pairs for DPO classification training.

    Each sample contains a tokenised query and the integer label indices
    for the chosen (correct) and rejected (wrong) tools.
    """

    def __init__(
        self,
        pairs: list[dict],
        tokenizer,
        max_len: int,
    ) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        pair = self.pairs[idx]
        enc = self.tokenizer(
            pair["prompt"],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "chosen_label": torch.tensor(LABEL2ID[pair["chosen"]], dtype=torch.long),
            "rejected_label": torch.tensor(LABEL2ID[pair["rejected"]], dtype=torch.long),
        }


# ------------------------------------------------------------------ #
# DPO loss for sequence classification                                 #
# ------------------------------------------------------------------ #

def compute_dpo_loss(
    policy_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    chosen_labels: torch.Tensor,
    rejected_labels: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    """DPO objective adapted for a sequence classifier.

    Standard DPO works on token log-probabilities. Here we apply the same
    principle to class log-probabilities from the 4-class classification head.

    For each preference pair:
      1. Get log-prob of chosen and rejected labels from both policy and reference.
      2. Compute implicit reward: (log π - log π_ref) for chosen and rejected.
      3. DPO loss = -log σ(β · (reward_chosen - reward_rejected))

    Minimising this loss pushes the policy to widen the probability gap between
    chosen and rejected *beyond* what the reference model already assigns.

    Args:
        policy_logits:   Raw classification logits from policy model  [B, 4]
        ref_logits:      Raw classification logits from reference model [B, 4]
        chosen_labels:   Integer indices of the correct tools  [B]
        rejected_labels: Integer indices of the wrong tools    [B]
        beta:            KL regularisation coefficient

    Returns:
        Scalar mean DPO loss.
    """
    policy_log_probs = F.log_softmax(policy_logits, dim=-1)   # [B, 4]
    ref_log_probs = F.log_softmax(ref_logits, dim=-1)          # [B, 4]

    # Extract per-sample log-probs for the specific chosen / rejected labels
    log_p_chosen = policy_log_probs.gather(1, chosen_labels.unsqueeze(1)).squeeze(1)
    log_p_rejected = policy_log_probs.gather(1, rejected_labels.unsqueeze(1)).squeeze(1)
    log_ref_chosen = ref_log_probs.gather(1, chosen_labels.unsqueeze(1)).squeeze(1)
    log_ref_rejected = ref_log_probs.gather(1, rejected_labels.unsqueeze(1)).squeeze(1)

    # Implicit reward = how much the policy shifted each label vs the reference
    # Positive → policy assigns more probability than reference (improvement)
    reward_chosen = log_p_chosen - log_ref_chosen
    reward_rejected = log_p_rejected - log_ref_rejected

    # DPO: maximise the preference of chosen reward over rejected reward
    advantage = beta * (reward_chosen - reward_rejected)
    loss = -F.logsigmoid(advantage)
    return loss.mean()


# ------------------------------------------------------------------ #
# Evaluation helpers (reproduce train_sft.py split exactly)           #
# ------------------------------------------------------------------ #

def _load_test_set() -> tuple[list[str], list[str]]:
    df = pd.read_csv(QUERIES_PATH)
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=df["correct_tool"],
    )
    return test_df["query"].tolist(), test_df["correct_tool"].tolist()


def _evaluate_router(
    router: ToolRouter,
    queries: list[str],
    y_true: list[str],
) -> dict:
    y_pred = [router.predict(q) for q in queries]
    overall = float(accuracy_score(y_true, y_pred))

    per_tool: dict[str, float | None] = {}
    for tool in TOOLS:
        idx = [i for i, y in enumerate(y_true) if y == tool]
        if idx:
            t = [y_true[i] for i in idx]
            p = [y_pred[i] for i in idx]
            per_tool[tool] = round(float(accuracy_score(t, p)), 4)
        else:
            per_tool[tool] = None

    return {"overall_accuracy": round(overall, 4), "per_tool": per_tool}


def _print_comparison_table(
    n_pairs: int,
    sft: dict,
    dpo: dict,
) -> None:
    W = 60
    print("\n" + "=" * W)
    print("  DPO Training Complete — Results")
    print("=" * W)
    print(f"  Preference pairs used : {n_pairs}")
    print()
    print("  Tool Accuracy Comparison:")
    print(f"  {'Tool':<20} {'SFT Baseline':>14} {'DPO Model':>11} {'Change':>12}")
    print("  " + "─" * (W - 2))

    def _row(label: str, sft_val: float | None, dpo_val: float | None) -> None:
        sft_s = f"{sft_val * 100:.0f}%" if sft_val is not None else "—"
        dpo_s = f"{dpo_val * 100:.0f}%" if dpo_val is not None else "—"

        if sft_val is not None and dpo_val is not None:
            d = (dpo_val - sft_val) * 100
            sign = "+" if d >= 0 else ""
            icon = " ✅" if d > 0.5 else (" ❌" if d < -0.5 else "")
            change_s = f"{sign}{d:.0f}%{icon}"
        else:
            change_s = "—"

        print(f"  {label:<20} {sft_s:>14} {dpo_s:>11} {change_s:>12}")

    for tool in TOOLS:
        _row(tool, sft["per_tool"].get(tool), dpo["per_tool"].get(tool))

    print("  " + "─" * (W - 2))
    _row("Overall", sft["overall_accuracy"], dpo["overall_accuracy"])
    print()


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    print("=" * 60)
    print("  DPO Fine-Tuning — ToolRouter")
    print("=" * 60)

    # ── 1. Load preference pairs ──────────────────────────────────
    print(f"\n[1/5] Loading preference pairs from '{PAIRS_PATH.name}' ...")

    pairs: list[dict] = []
    if PAIRS_PATH.is_file():
        for lineno, raw in enumerate(
            PAIRS_PATH.read_text(encoding="utf-8").splitlines(), start=1
        ):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"      ⚠ Line {lineno}: invalid JSON ({exc}) — skipped")
                continue

            missing = [k for k in ("prompt", "chosen", "rejected") if k not in obj]
            if missing:
                print(f"      ⚠ Line {lineno}: missing keys {missing} — skipped")
                continue

            if obj["chosen"] not in LABEL2ID:
                print(f"      ⚠ Line {lineno}: unknown tool '{obj['chosen']}' — skipped")
                continue
            if obj["rejected"] not in LABEL2ID:
                print(f"      ⚠ Line {lineno}: unknown tool '{obj['rejected']}' — skipped")
                continue

            pairs.append(obj)
    else:
        print(f"      File not found: {PAIRS_PATH}")

    print(f"      Valid pairs loaded: {len(pairs)}")

    if len(pairs) < MIN_PAIRS:
        print(f"\n  ⚠  Only {len(pairs)} valid preference pair(s) found.")
        print(f"     Need at least {MIN_PAIRS} to start DPO training.\n")
        print("  How to collect more preference pairs:")
        print("    1. Start the backend:  uvicorn api.main:app --reload")
        print("    2. Open the UI at      http://localhost:3000")
        print("    3. Submit queries and click 👎 when the wrong tool is chosen.")
        print("    4. Re-run this script once you have enough pairs.\n")
        return

    # ── 2. Convert to DPO prompt format ──────────────────────────
    print(f"\n[2/5] Converting {len(pairs)} pair(s) to DPO prompt format ...")

    dpo_pairs: list[dict] = []
    for pair in pairs:
        dpo_pairs.append({
            "prompt": PROMPT_TEMPLATE.format(query=pair["prompt"]),
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
        })

    print("      Sample entry:")
    print(f"        prompt   : {dpo_pairs[0]['prompt'][:72]}...")
    print(f"        chosen   : {dpo_pairs[0]['chosen']}")
    print(f"        rejected : {dpo_pairs[0]['rejected']}")

    # ── Overwrite guard ───────────────────────────────────────────
    if (DPO_SAVE_PATH / "config.json").is_file():
        print(f"\n  ⚠  A saved DPO model already exists at '{DPO_SAVE_PATH}'.")
        try:
            answer = input("     Overwrite it? [y/N]  ").strip().lower()
        except EOFError:
            answer = "n"
        if answer not in ("y", "yes"):
            print("     Aborted — existing model preserved.")
            return

    # ── 3. Load policy model + frozen reference model ─────────────
    print(f"\n[3/5] Loading models from '{SFT_MODEL_PATH}' ...")

    # Policy model: will be updated by DPO
    policy_router = ToolRouter().load(str(SFT_MODEL_PATH))
    policy_model = policy_router.model
    tokenizer = policy_router.tokenizer
    device = policy_router.device

    # Reference model: frozen copy of SFT, used to compute the KL baseline.
    # DPO measures how much the policy shifts each label's probability
    # *relative to this anchor*, preventing catastrophic forgetting.
    print("      Loading frozen reference model (same SFT weights, no grad) ...")
    ref_router = ToolRouter().load(str(SFT_MODEL_PATH))
    ref_model = ref_router.model
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    policy_model.train()
    print(f"      Policy model  : trainable  ({device})")
    print(f"      Reference model: frozen    ({device})")

    # ── 4. DPO training ───────────────────────────────────────────
    print(f"\n[4/5] DPO training ...")

    # DPOConfig holds all hyperparameters (subclass of TrainingArguments).
    # We use it as a settings container; the actual training loop is custom
    # because DPOTrainer expects a causal LM, not a sequence classifier.
    try:
        dpo_cfg = DPOConfig(
            beta=DPO_BETA,
            learning_rate=DPO_LR,
            num_train_epochs=DPO_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            max_length=MAX_LEN,
            output_dir=str(DPO_SAVE_PATH),
            logging_steps=1,
            save_strategy="epoch",
            remove_unused_columns=False,
            report_to="none",
        )
    except TypeError:
        # max_length was added in a later TRL release; fall back gracefully
        dpo_cfg = DPOConfig(
            beta=DPO_BETA,
            learning_rate=DPO_LR,
            num_train_epochs=DPO_EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            output_dir=str(DPO_SAVE_PATH),
            logging_steps=1,
            save_strategy="epoch",
            remove_unused_columns=False,
            report_to="none",
        )

    dataset = PreferencePairDataset(dpo_pairs, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    effective_batch = BATCH_SIZE * GRAD_ACCUM
    steps_per_epoch = max(len(dataloader) // GRAD_ACCUM, 1)
    total_updates = steps_per_epoch * DPO_EPOCHS

    print(f"      Pairs            : {len(dpo_pairs)}")
    print(f"      Effective batch  : {effective_batch}")
    print(f"      Optimizer steps  : {total_updates}")
    print(f"      β (KL penalty)   : {DPO_BETA}")
    print(f"      Learning rate    : {DPO_LR}")
    print(f"      Device           : {device}")
    print(f"-" * 60)

    optimizer = AdamW(
        policy_model.parameters(),
        lr=dpo_cfg.learning_rate,
        weight_decay=0.01,
    )
    # Linear decay with no warmup (dataset is too small for warmup)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=max(total_updates, 1),
    )

    DPO_SAVE_PATH.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(1, DPO_EPOCHS + 1):
        epoch_loss_sum = 0.0
        epoch_local_steps = 0

        for local_step, batch in enumerate(dataloader, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            chosen_labels = batch["chosen_label"].to(device)
            rejected_labels = batch["rejected_label"].to(device)

            # ── Policy forward pass (gradients flow here) ──────
            policy_logits = policy_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

            # ── Reference forward pass (no gradients — frozen) ──
            # The reference captures the SFT baseline distribution so
            # the DPO loss measures improvement *relative* to it.
            with torch.no_grad():
                ref_logits = ref_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            # Normalise loss by accumulation steps so gradient magnitude
            # is independent of GRAD_ACCUM
            loss = compute_dpo_loss(
                policy_logits, ref_logits,
                chosen_labels, rejected_labels,
                DPO_BETA,
            ) / GRAD_ACCUM

            loss.backward()
            epoch_loss_sum += loss.item() * GRAD_ACCUM   # log un-normalised
            epoch_local_steps += 1

            # Flush accumulated gradients
            is_last_in_epoch = (local_step == len(dataloader))
            if local_step % GRAD_ACCUM == 0 or is_last_in_epoch:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg = epoch_loss_sum / epoch_local_steps
                print(
                    f"  Epoch {epoch}/{DPO_EPOCHS}  "
                    f"Step {global_step:>3}  "
                    f"Loss: {avg:.4f}"
                )

        # Per-epoch checkpoint
        ckpt_dir = DPO_SAVE_PATH / f"checkpoint-epoch-{epoch}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        policy_model.save_pretrained(str(ckpt_dir))
        tokenizer.save_pretrained(str(ckpt_dir))
        avg_epoch = epoch_loss_sum / max(epoch_local_steps, 1)
        print(f"\n  ── Epoch {epoch} done.  Avg loss: {avg_epoch:.4f}  "
              f"Checkpoint → {ckpt_dir.name}\n")

    # Save final model (best = last epoch for DPO with small datasets)
    print("Saving final DPO model ...")
    policy_router.model = policy_model
    policy_router.save(str(DPO_SAVE_PATH))

    # ── 5. Post-training evaluation ───────────────────────────────
    print(f"\n[5/5] Post-training evaluation on held-out test set ...")

    queries, y_true = _load_test_set()
    print(f"      Test samples: {len(queries)}")

    print("      Evaluating SFT baseline ...")
    sft_router = ToolRouter().load(str(SFT_MODEL_PATH))
    sft_metrics = _evaluate_router(sft_router, queries, y_true)

    print("      Evaluating DPO model ...")
    dpo_router = ToolRouter().load(str(DPO_SAVE_PATH))
    dpo_metrics = _evaluate_router(dpo_router, queries, y_true)

    _print_comparison_table(len(pairs), sft_metrics, dpo_metrics)

    # ── Save evaluation_results.json for fast API serving ─────────
    results = {
        "sft_baseline": sft_metrics,
        "dpo_model": dpo_metrics,
        "training_info": {
            "preference_pairs_used": len(pairs),
            "dpo_beta": DPO_BETA,
            "epochs": DPO_EPOCHS,
            "learning_rate": DPO_LR,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }
    EVAL_RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Evaluation results → '{EVAL_RESULTS_PATH.name}'")
    print("\n  Next step: run `python evaluation/compare_baseline_vs_dpo.py`\n")


if __name__ == "__main__":
    main()
