"""Service layer — loaded once at startup, shared across all requests.

Owns:
- The singleton ToolRouter (SFT model, and optionally DPO model)
- The singleton RewardModel (optional — loaded if models/reward_model/ exists)
- The compiled LangGraph agent
- All business logic for feedback persistence and evaluation
"""

from __future__ import annotations

import json
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent.langgraph_agent import init_agent, run_agent  # noqa: E402
from reward_model.reward_scorer import RewardModel  # noqa: E402
from router.tool_router import TOOLS, ToolRouter  # noqa: E402

# ------------------------------------------------------------------ #
# Singleton state                                                      #
# ------------------------------------------------------------------ #

@dataclass
class _ServiceState:
    sft_router: Optional[ToolRouter] = None
    dpo_router: Optional[ToolRouter] = None
    # Reward model scores (query, tool) pairs → scalar in [0, 1].
    # Distinct from ToolRouter: the router classifies; the reward model evaluates quality.
    reward_model: Optional[RewardModel] = None
    data_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data")
    sft_model_path: str = ""
    dpo_model_path: str = ""
    _write_lock: threading.Lock = field(default_factory=threading.Lock)


_state = _ServiceState()

# ------------------------------------------------------------------ #
# Initialisation (called from FastAPI lifespan)                        #
# ------------------------------------------------------------------ #

def initialize(
    sft_model_path: str,
    dpo_model_path: str,
    data_dir: str,
) -> None:
    """Load models and wire up the LangGraph agent.  Called once at startup.

    Args:
        sft_model_path: Path to the SFT-trained ToolRouter checkpoint.
        dpo_model_path: Path to the DPO-trained ToolRouter checkpoint.
                        Silently skipped if the directory does not exist.
        data_dir:       Directory containing feedback.csv / preference_pairs.jsonl.
    """
    _state.sft_model_path = sft_model_path
    _state.dpo_model_path = dpo_model_path
    _state.data_dir = Path(data_dir)

    # SFT router (required — raises FileNotFoundError if missing)
    _state.sft_router = ToolRouter().load(sft_model_path)

    # Wire LangGraph agent to the SFT router
    init_agent(_state.sft_router)

    # DPO router (optional — skip gracefully if not yet trained)
    dpo_config = Path(dpo_model_path) / "config.json"
    if dpo_config.is_file():
        try:
            _state.dpo_router = ToolRouter().load(dpo_model_path)
        except Exception as exc:
            print(f"[AgentService] DPO model could not be loaded: {exc}")

    # Reward model (optional — skip gracefully if not yet trained)
    reward_model_path = PROJECT_ROOT / "models" / "reward_model"
    if (reward_model_path / "config.json").is_file():
        try:
            _state.reward_model = RewardModel().load(str(reward_model_path))
        except Exception as exc:
            print(f"[AgentService] Reward model could not be loaded: {exc}")

    print("[AgentService] Initialisation complete.")


def is_ready() -> bool:
    return _state.sft_router is not None


# ------------------------------------------------------------------ #
# Query execution                                                      #
# ------------------------------------------------------------------ #

def run_query(query: str) -> dict:
    """Route *query* through the LangGraph agent and return structured output.

    Returns:
        dict with keys: query, selected_tool, confidence, response,
        reward_score (float | None), all_tool_scores (dict | None).

    Raises:
        RuntimeError: If the service has not been initialised.
    """
    if not is_ready():
        raise RuntimeError("AgentService is not initialised. Call initialize() first.")

    result = run_agent(query)

    # Enrich with reward scores when the reward model is available.
    # The reward model evaluates *quality* of the tool choice (0-1 scalar),
    # while the router *predicts* the tool — they serve different purposes.
    if _state.reward_model is not None:
        try:
            selected_tool = result.get("selected_tool", "")
            result["reward_score"] = _state.reward_model.score(query, selected_tool)
            result["all_tool_scores"] = _state.reward_model.score_all_tools(query)
        except Exception as exc:
            print(f"[AgentService] Reward scoring failed: {exc}")
            result["reward_score"] = None
            result["all_tool_scores"] = None
    else:
        result["reward_score"] = None
        result["all_tool_scores"] = None

    return result


# ------------------------------------------------------------------ #
# Feedback persistence                                                 #
# ------------------------------------------------------------------ #

def save_feedback(
    session_id: str,
    query: str,
    selected_tool: str,
    correct_tool: str,
    rating: str,
) -> bool:
    """Persist one feedback record and optionally create a DPO preference pair.

    A preference pair is created when the human signals the model was wrong:
    rating == "thumbs_down" AND correct_tool != selected_tool.

    Args:
        session_id:    UUID linking the feedback to the original query response.
        query:         The original customer query text.
        selected_tool: The tool the model predicted.
        correct_tool:  The tool the human considers correct.
        rating:        ``"thumbs_up"`` or ``"thumbs_down"``.

    Returns:
        True if a preference pair was created, False otherwise.
    """
    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).isoformat()
    pair_created = False

    with _state._write_lock:
        # Append to feedback.csv
        feedback_path = _state.data_dir / "feedback.csv"
        row = f'"{query}",{selected_tool},{correct_tool},{rating},{timestamp}\n'
        with open(feedback_path, "a", encoding="utf-8") as fh:
            fh.write(row)

        # Create DPO preference pair if the model chose wrong
        if rating == "thumbs_down" and correct_tool != selected_tool:
            pair = {
                "prompt": query,
                "chosen": correct_tool,
                "rejected": selected_tool,
            }
            pairs_path = _state.data_dir / "preference_pairs.jsonl"
            with open(pairs_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(pair) + "\n")
            pair_created = True

    return pair_created


# ------------------------------------------------------------------ #
# Feedback statistics                                                  #
# ------------------------------------------------------------------ #

def get_feedback_stats() -> dict:
    """Read feedback.csv and preference_pairs.jsonl and return aggregate stats.

    Returns:
        dict with keys: total_feedback, thumbs_up, thumbs_down,
        preference_pairs_collected, accuracy_from_feedback.
    """
    feedback_path = _state.data_dir / "feedback.csv"
    pairs_path = _state.data_dir / "preference_pairs.jsonl"

    thumbs_up = 0
    thumbs_down = 0

    try:
        df = pd.read_csv(feedback_path)
        if not df.empty and "rating" in df.columns:
            thumbs_up = int((df["rating"] == "thumbs_up").sum())
            thumbs_down = int((df["rating"] == "thumbs_down").sum())
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pass

    pairs_count = 0
    try:
        with open(pairs_path, "r", encoding="utf-8") as fh:
            pairs_count = sum(1 for line in fh if line.strip())
    except FileNotFoundError:
        pass

    total = thumbs_up + thumbs_down
    accuracy = round(thumbs_up / total, 4) if total > 0 else None

    return {
        "total_feedback": total,
        "thumbs_up": thumbs_up,
        "thumbs_down": thumbs_down,
        "preference_pairs_collected": pairs_count,
        "accuracy_from_feedback": accuracy,
    }


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #

def _load_test_set() -> tuple[list[str], list[str]]:
    """Reproduce the exact 80/20 split used during SFT training."""
    df = pd.read_csv(_state.data_dir / "queries.csv")
    _, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["correct_tool"],
    )
    return test_df["query"].tolist(), test_df["correct_tool"].tolist()


def _evaluate_router(
    router: ToolRouter,
    queries: list[str],
    y_true: list[str],
) -> dict:
    """Run predictions and return overall + per-tool accuracy."""
    y_pred = [router.predict(q) for q in queries]
    overall = round(float(accuracy_score(y_true, y_pred)), 4)

    per_tool: dict[str, Optional[float]] = {}
    for tool in TOOLS:
        indices = [i for i, y in enumerate(y_true) if y == tool]
        if not indices:
            per_tool[tool] = None
            continue
        t = [y_true[i] for i in indices]
        p = [y_pred[i] for i in indices]
        per_tool[tool] = round(float(accuracy_score(t, p)), 4)

    return {"overall_accuracy": overall, "per_tool": per_tool}


def get_evaluation() -> dict:
    """Evaluate SFT and (optionally) DPO routers on the held-out test set.

    Fast path: if training/run_dpo.py has already written
    ``data/evaluation_results.json``, read and return it immediately
    (no model inference needed).

    Slow path: if the cache file is absent, run live predictions on both
    models — takes a few seconds on CPU.

    Returns:
        dict with keys ``sft_baseline`` and ``dpo_model`` (null if not available).

    Raises:
        RuntimeError: If the SFT model is not loaded.
    """
    if not is_ready():
        raise RuntimeError("AgentService is not initialised.")

    # ── Fast path — return pre-computed results from DPO training ──
    results_path = _state.data_dir / "evaluation_results.json"
    if results_path.is_file():
        try:
            with open(results_path, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            return {
                "sft_baseline": cached.get("sft_baseline"),
                "dpo_model": cached.get("dpo_model"),
                "training_info": cached.get("training_info"),
            }
        except (json.JSONDecodeError, KeyError) as exc:
            print(
                f"[AgentService] evaluation_results.json could not be parsed "
                f"({exc}). Falling back to live evaluation."
            )

    # ── Slow path — run inference on both models ────────────────────
    queries, y_true = _load_test_set()

    sft_result = _evaluate_router(_state.sft_router, queries, y_true)

    dpo_result = None
    if _state.dpo_router is not None:
        dpo_result = _evaluate_router(_state.dpo_router, queries, y_true)

    return {
        "sft_baseline": sft_result,
        "dpo_model": dpo_result,
        "training_info": None,
    }
