"""Reward Model — scores how appropriate a tool choice is for a given query.

Unlike the ToolRouter (which classifies query → one predicted tool), the Reward
Model takes a (query, tool) pair as input and outputs a scalar in [0, 1]:

  score ≈ 1.0  →  the tool is a great choice for this query
  score ≈ 0.0  →  the tool is a poor choice for this query

This is a regression problem, not classification — MSELoss is used during
training, and the output is passed through Sigmoid so it lives in [0, 1].

Architecture:
    DistilBERT encoder on "{query} [SEP] {tool_name}"
    → [CLS] hidden state (768-dim)
    → Linear(768, 256) → ReLU → Dropout(0.1) → Linear(256, 1) → Sigmoid
"""

from __future__ import annotations

import json
import os
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"
TOOLS = ["faq_search", "order_status", "refund_policy", "raise_ticket"]
MAX_LEN = 128


# ── Neural network modules ──────────────────────────────────────────────────

class _RegressionHead(nn.Module):
    """Three-layer MLP that maps a 768-dim [CLS] vector to a scalar in [0, 1]."""

    def __init__(self, hidden_size: int = 768) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, cls_hidden: torch.Tensor) -> torch.Tensor:
        # cls_hidden: (batch, 768) → output: (batch,)
        return self.net(cls_hidden).squeeze(-1)


class _RewardNet(nn.Module):
    """DistilBERT encoder + regression head."""

    def __init__(self, base_model_name: str) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model_name)
        self.head = _RegressionHead(self.encoder.config.hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [CLS] token representation
        return self.head(cls)


# ── Public API ───────────────────────────────────────────────────────────────

class RewardModel:
    """Scores (query, tool) pairs with a scalar in [0, 1].

    Higher score means the tool is more appropriate for the query.

    Typical usage::

        # Train a new model (see reward_model/train_reward.py)
        model = RewardModel().load_base()

        # Load a saved checkpoint
        model = RewardModel().load("models/reward_model")

        score = model.score("Where is my order?", "order_status")   # → 0.91
        all   = model.score_all_tools("Where is my order?")          # sorted dict
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[_RewardNet] = None

    # ── Loading ─────────────────────────────────────────────────────────────

    def load_base(self) -> "RewardModel":
        """Initialise a fresh DistilBERT encoder with a new regression head.

        Returns:
            self, for chaining: ``RewardModel().load_base()``.
        """
        print(f"[RewardModel] Loading base model '{BASE_MODEL}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = _RewardNet(BASE_MODEL).to(self.device)
        param_count = sum(p.numel() for p in self.model.parameters())
        print(f"[RewardModel] Ready on {self.device}.  Parameters: {param_count:,}")
        return self

    def load(self, path: str) -> "RewardModel":
        """Restore a saved RewardModel from *path*.

        Args:
            path: Directory created by :meth:`save`.

        Returns:
            self, for chaining.

        Raises:
            FileNotFoundError: If config.json is absent — model not trained yet.
        """
        config_file = os.path.join(path, "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"No saved reward model at '{path}' (config.json missing).\n"
                "Run `python reward_model/train_reward.py` first to train and save."
            )
        print(f"[RewardModel] Loading from '{path}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = _RewardNet(BASE_MODEL).to(self.device)
        weights_path = os.path.join(path, "reward_model.pt")
        self.model.load_state_dict(
            torch.load(weights_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print(f"[RewardModel] Ready on {self.device}.")
        return self

    def save(self, path: str) -> None:
        """Persist the tokenizer and model weights to *path*.

        Args:
            path: Target directory; created automatically if absent.
        """
        self._require_loaded()
        os.makedirs(path, exist_ok=True)
        self.tokenizer.save_pretrained(path)
        torch.save(self.model.state_dict(), os.path.join(path, "reward_model.pt"))
        with open(os.path.join(path, "config.json"), "w", encoding="utf-8") as fh:
            json.dump({"base_model": BASE_MODEL, "type": "reward_regression"}, fh)
        print(f"[RewardModel] Saved to '{path}'.")

    # ── Inference ────────────────────────────────────────────────────────────

    def score(self, query: str, tool: str) -> float:
        """Return the reward score for a (query, tool) pair.

        Args:
            query: Customer query in natural language.
            tool:  One of the four tool names.

        Returns:
            Float in [0, 1].  Higher = better tool choice.
        """
        self._require_loaded()
        self.model.eval()
        text = f"{query} [SEP] {tool}"
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        inputs.pop("token_type_ids", None)  # DistilBERT has no token_type_ids
        with torch.no_grad():
            raw = self.model(**inputs).item()
        return round(float(raw), 4)

    def score_all_tools(self, query: str) -> dict[str, float]:
        """Score all four tools for *query* and return sorted best-to-worst.

        Args:
            query: Customer query in natural language.

        Returns:
            Dict mapping tool name → score, sorted descending by score.
        """
        scores = {tool: self.score(query, tool) for tool in TOOLS}
        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    # ── Internal ─────────────────────────────────────────────────────────────

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "[RewardModel] Model not loaded.\n"
                "  • Call load_base() to initialise a fresh model for training.\n"
                "  • Call load(path) to restore a saved checkpoint."
            )
