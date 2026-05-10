from __future__ import annotations

import os
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE_MODEL = "distilbert-base-uncased"
TOOLS = ["faq_search", "order_status", "refund_policy", "raise_ticket"]
ID2LABEL = {i: label for i, label in enumerate(TOOLS)}
LABEL2ID = {label: i for i, label in enumerate(TOOLS)}


class ToolRouter:
    """Classifies a customer query into one of four tool calls.

    Wraps a DistilBERT sequence-classification model with a clean API so
    training, inference, and evaluation scripts can all share the same logic.

    Typical usage::

        # New model for fine-tuning
        router = ToolRouter().load_base()

        # Previously saved fine-tuned model
        router = ToolRouter().load("models/sft_router")

        tool = router.predict("Where is my order #123?")
        result = router.predict_with_confidence("What is your refund policy?")
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None

    # ------------------------------------------------------------------ #
    # Loading                                                              #
    # ------------------------------------------------------------------ #

    def load_base(self) -> "ToolRouter":
        """Download and initialise the base DistilBERT model with a fresh 4-class head.

        Returns:
            self, allowing chained calls: ``ToolRouter().load_base()``.
        """
        print(f"[ToolRouter] Loading base model '{BASE_MODEL}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            BASE_MODEL,
            num_labels=len(TOOLS),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.model.to(self.device)
        print(f"[ToolRouter] Ready on {self.device}. Parameters: "
              f"{sum(p.numel() for p in self.model.parameters()):,}")
        return self

    def load(self, path: str) -> "ToolRouter":
        """Load a previously fine-tuned router from a local directory.

        Args:
            path: Directory created by :meth:`save`.

        Returns:
            self, allowing chained calls.

        Raises:
            FileNotFoundError: If the directory or model config is absent.
                               Run training/train_sft.py first to create the model.
        """
        config_file = os.path.join(path, "config.json")
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"No saved model found at '{path}' (config.json missing).\n"
                "Run `python training/train_sft.py` first to train and save the model."
            )
        print(f"[ToolRouter] Loading fine-tuned model from '{path}' ...")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)
        print(f"[ToolRouter] Ready on {self.device}.")
        return self

    def save(self, path: str) -> None:
        """Persist the model and tokenizer to *path*.

        Args:
            path: Target directory; created automatically if it does not exist.
        """
        self._require_loaded()
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"[ToolRouter] Saved to '{path}'.")

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, query: str) -> str:
        """Return the name of the highest-scoring tool for *query*.

        Args:
            query: Customer support question in natural language.

        Returns:
            One of: ``'faq_search'``, ``'order_status'``,
            ``'refund_policy'``, ``'raise_ticket'``.

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        return self.predict_with_confidence(query)["tool"]

    def predict_with_confidence(self, query: str) -> dict:
        """Return the predicted tool together with per-class softmax scores.

        Args:
            query: Customer support question in natural language.

        Returns:
            A dict with three keys:

            * ``'tool'``        — predicted tool name (str)
            * ``'confidence'``  — softmax probability of the top class (float 0-1)
            * ``'all_scores'``  — ``{tool_name: probability}`` for all four tools

        Raises:
            RuntimeError: If the model has not been loaded yet.
        """
        self._require_loaded()
        self.model.eval()

        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        probs = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
        top_idx = int(torch.argmax(logits, dim=-1).item())

        return {
            "tool": TOOLS[top_idx],
            "confidence": round(probs[top_idx], 4),
            "all_scores": {TOOLS[i]: round(probs[i], 4) for i in range(len(TOOLS))},
        }

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "[ToolRouter] Model is not loaded.\n"
                "  • Call load_base() to initialise a fresh model for fine-tuning.\n"
                "  • Call load(path) to restore a saved fine-tuned checkpoint."
            )
