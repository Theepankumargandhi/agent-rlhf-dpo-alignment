"""FastAPI application — customer support agent with RLHF feedback loop.

Run from the project root:
    uvicorn api.main:app --reload --port 8000
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

import os
from api import agent_service  # noqa: E402

# ------------------------------------------------------------------ #
# Logging                                                              #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Settings (from .env with fallbacks)                                  #
# ------------------------------------------------------------------ #

_MODEL_PATH = os.getenv("MODEL_PATH", str(PROJECT_ROOT / "models" / "sft_router"))
_DPO_PATH = os.getenv("DPO_MODEL_PATH", str(PROJECT_ROOT / "models" / "dpo_router"))
_DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data"))

# Resolve relative paths against project root
if not Path(_MODEL_PATH).is_absolute():
    _MODEL_PATH = str(PROJECT_ROOT / _MODEL_PATH)
if not Path(_DPO_PATH).is_absolute():
    _DPO_PATH = str(PROJECT_ROOT / _DPO_PATH)
if not Path(_DATA_DIR).is_absolute():
    _DATA_DIR = str(PROJECT_ROOT / _DATA_DIR)

# ------------------------------------------------------------------ #
# Lifespan                                                             #
# ------------------------------------------------------------------ #

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — loading models ...")
    try:
        agent_service.initialize(
            sft_model_path=_MODEL_PATH,
            dpo_model_path=_DPO_PATH,
            data_dir=_DATA_DIR,
        )
        logger.info("Models loaded. API is ready.")
    except FileNotFoundError as exc:
        logger.error(f"Model not found: {exc}")
        logger.error("Run `python training/train_sft.py` first.")
    yield
    logger.info("Shutting down.")


# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #

app = FastAPI(
    title="RLHF Agent Tool Use API",
    description="Customer support agent with human preference feedback loop.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------------ #
# Request / response logging middleware                                #
# ------------------------------------------------------------------ #

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    logger.info(
        f"{request.method:<6} {request.url.path:<35} "
        f"→ {response.status_code}  ({elapsed_ms:.1f} ms)"
    )
    return response


# ------------------------------------------------------------------ #
# Pydantic schemas                                                     #
# ------------------------------------------------------------------ #

class QueryRequest(BaseModel):
    query: str

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be empty")
        return v


class QueryResponse(BaseModel):
    query: str
    selected_tool: str
    confidence: float
    response: str
    session_id: str
    # Reward model fields — None when reward model has not been trained yet.
    # reward_score: quality of the selected tool choice (0 = bad, 1 = good)
    # all_tool_scores: quality scores for every tool, for comparison in the UI
    reward_score: Optional[float] = None
    all_tool_scores: Optional[dict[str, float]] = None


class FeedbackRequest(BaseModel):
    session_id: str
    query: str
    selected_tool: str
    correct_tool: str
    rating: Literal["thumbs_up", "thumbs_down"]

    @field_validator("correct_tool", "selected_tool")
    @classmethod
    def valid_tool(cls, v: str) -> str:
        allowed = {"faq_search", "order_status", "refund_policy", "raise_ticket"}
        if v not in allowed:
            raise ValueError(f"tool must be one of {allowed}")
        return v


class FeedbackResponse(BaseModel):
    status: str
    preference_pair_created: bool


class FeedbackStatsResponse(BaseModel):
    total_feedback: int
    thumbs_up: int
    thumbs_down: int
    preference_pairs_collected: int
    accuracy_from_feedback: Optional[float]


class ToolAccuracy(BaseModel):
    overall_accuracy: float
    per_tool: dict[str, Optional[float]]


class TrainingInfo(BaseModel):
    preference_pairs_used: int
    dpo_beta: float
    epochs: int
    learning_rate: float
    gradient_accumulation_steps: int
    trained_at: str


class EvaluationResponse(BaseModel):
    sft_baseline: Optional[ToolAccuracy]
    dpo_model: Optional[ToolAccuracy]
    training_info: Optional[TrainingInfo]


class RewardModelStatsResponse(BaseModel):
    trained: bool
    pearson_correlation: Optional[float]
    discrimination_gap: Optional[float]
    correct_tool_avg_score: Optional[float]
    wrong_tool_avg_score: Optional[float]
    samples_used: int
    real_samples: int
    synthetic_samples: int
    trained_at: Optional[str]
    note: Optional[str]


# ------------------------------------------------------------------ #
# Health                                                               #
# ------------------------------------------------------------------ #

@app.get("/api/health", tags=["System"])
def health():
    """Liveness check — returns model load status."""
    return {"status": "ok", "model_loaded": agent_service.is_ready()}


# ------------------------------------------------------------------ #
# Query                                                                #
# ------------------------------------------------------------------ #

@app.post(
    "/api/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    tags=["Agent"],
)
def handle_query(body: QueryRequest):
    """Route the customer query through the LangGraph agent and return the response."""
    if not agent_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Run training/train_sft.py first.",
        )
    try:
        result = agent_service.run_query(body.query)
    except Exception as exc:
        logger.exception("Query execution failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return QueryResponse(
        query=result["query"],
        selected_tool=result["selected_tool"],
        confidence=result["confidence"],
        response=result["response"],
        session_id=str(uuid.uuid4()),
        reward_score=result.get("reward_score"),
        all_tool_scores=result.get("all_tool_scores"),
    )


# ------------------------------------------------------------------ #
# Feedback                                                             #
# ------------------------------------------------------------------ #

@app.post(
    "/api/feedback",
    response_model=FeedbackResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Feedback"],
)
def submit_feedback(body: FeedbackRequest):
    """Save human rating and optionally create a DPO preference pair."""
    try:
        pair_created = agent_service.save_feedback(
            session_id=body.session_id,
            query=body.query,
            selected_tool=body.selected_tool,
            correct_tool=body.correct_tool,
            rating=body.rating,
        )
    except OSError as exc:
        logger.exception("Failed to write feedback")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not persist feedback: {exc}",
        )

    return FeedbackResponse(status="saved", preference_pair_created=pair_created)


@app.get(
    "/api/feedback/stats",
    response_model=FeedbackStatsResponse,
    tags=["Feedback"],
)
def feedback_stats():
    """Return aggregate statistics from collected human feedback."""
    try:
        return agent_service.get_feedback_stats()
    except Exception as exc:
        logger.exception("Failed to read feedback stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )


# ------------------------------------------------------------------ #
# Evaluation                                                           #
# ------------------------------------------------------------------ #

@app.get(
    "/api/evaluation",
    response_model=EvaluationResponse,
    tags=["Evaluation"],
)
def evaluation():
    """Run both routers on the held-out test set and return accuracy metrics."""
    if not agent_service.is_ready():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded.",
        )
    try:
        result = agent_service.get_evaluation()
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )

    return EvaluationResponse(
        sft_baseline=ToolAccuracy(**result["sft_baseline"]) if result["sft_baseline"] else None,
        dpo_model=ToolAccuracy(**result["dpo_model"]) if result["dpo_model"] else None,
        training_info=TrainingInfo(**result["training_info"]) if result.get("training_info") else None,
    )


# ------------------------------------------------------------------ #
# Reward Model                                                         #
# ------------------------------------------------------------------ #

@app.get(
    "/api/reward-model/stats",
    response_model=RewardModelStatsResponse,
    tags=["Reward Model"],
)
def reward_model_stats():
    """Return training statistics for the reward model, read from data/reward_model_stats.json."""
    stats_path = Path(_DATA_DIR) / "reward_model_stats.json"
    if not stats_path.is_file():
        return RewardModelStatsResponse(
            trained=False,
            pearson_correlation=None,
            discrimination_gap=None,
            correct_tool_avg_score=None,
            wrong_tool_avg_score=None,
            samples_used=0,
            real_samples=0,
            synthetic_samples=0,
            trained_at=None,
            note="Reward model not yet trained. Run `python reward_model/train_reward.py` first.",
        )
    try:
        with open(stats_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return RewardModelStatsResponse(**data)
    except Exception as exc:
        logger.exception("Failed to read reward model stats")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
