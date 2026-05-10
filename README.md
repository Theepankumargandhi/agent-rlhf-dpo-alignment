# RLHF Agent Tool Use

> An end-to-end **Reinforcement Learning from Human Feedback** system that fine-tunes a DistilBERT tool-selector using human preference data, orchestrated by a LangGraph agent, served through a FastAPI backend, and monitored via a Next.js dashboard.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         RLHF Feedback Loop                              │
│                                                                         │
│  User Query                                                             │
│      │                                                                  │
│      ▼                                                                  │
│  ┌──────────────────────────────────────────┐                           │
│  │           LangGraph Agent                │                           │
│  │                                          │                           │
│  │  route_query ──► execute_tool ──► format_response                    │
│  │       │               │                  │                           │
│  │  ToolRouter       Mock Tools         Response                        │
│  │  (DistilBERT)   (faq / order /    (+ session_id)                    │
│  │                  refund / ticket)        │                           │
│  └──────────────────────────────────────────┘                           │
│                                             │                           │
│                                             ▼                           │
│                                     Human Feedback                      │
│                                       👍  or  👎                        │
│                                     (+ correct tool)                    │
│                                             │                           │
│              ┌──────────────────────────────┘                           │
│              │                                                          │
│              ▼                                                          │
│   feedback.csv          preference_pairs.jsonl                          │
│   (all ratings)         (thumbs_down + wrong tool)                      │
│              │                    │                                     │
│              └────────┬───────────┘                                     │
│                       ▼                                                 │
│              DPO Fine-Tuning (run_dpo.py)                               │
│              Policy ← SFT weights (frozen ref)                          │
│              Loss = -log σ(β·[Δ_chosen − Δ_rejected])                  │
│                       │                                                 │
│                       ▼                                                 │
│              dpo_router/ (fine-tuned checkpoint)                        │
│                       │                                                 │
│                       └──► evaluation_results.json                      │
│                            (SFT vs DPO accuracy)                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Results

15 human preference pairs collected → 3 epochs of DPO → **perfect test accuracy**.

| Tool | SFT Baseline | DPO Model | Improvement |
|------|:------------:|:---------:|:-----------:|
| `faq_search` | 66.7% | 100.0% | +33.3% ✅ |
| `order_status` | 100.0% | 100.0% | — |
| `refund_policy` | 100.0% | 100.0% | — |
| `raise_ticket` | 100.0% | 100.0% | — |
| **Overall** | **91.7%** | **100.0%** | **+8.3% ✅** |

Training config: β = 0.1 · LR = 1e-5 · 3 epochs · gradient accumulation steps = 4

### Reward Model (V2)

A separate **regression model** trained to score how appropriate a tool choice is for a given query, independent of the classifier that selects the tool.

| Metric | Value |
|--------|-------|
| Architecture | DistilBERT + 3-layer regression head (768 → 512 → 128 → 1) |
| Training loss | MSELoss (regression, not classification) |
| Training data | 118 samples (54 real + 80 synthetic) |
| Pearson correlation | 0.35 |
| Discrimination gap | 0.01 (correct avg − wrong avg) |
| LR / Epochs | 5e-6 / 15 with cosine warmup |

#### Lessons Learned

- **Reward models need scale.** Meaningful discrimination requires 10K+ clean preference pairs. With only 54 real samples, the model converges to near-random discrimination between correct and incorrect tools.
- **Label noise is amplified.** 54 human ratings contain natural inconsistencies that overwhelm the signal on a small dataset, driving the gap toward zero.
- **Synthetic data helps but does not replace real feedback.** The 80 synthetic samples provided a stable training base; however, synthetic positives and negatives are too clean compared to real user queries, limiting generalization.
- **Production path: RLAIF.** Using a frontier LLM as a synthetic annotator (Constitutional AI style) can generate tens of thousands of consistent preference pairs at scale — this is the approach taken at Anthropic/OpenAI to bootstrap reward models before human data is available.
- **Gap is the key metric, not MSE.** A low MSE with a near-zero gap means the model assigns similar scores to all tools — it has learned the marginal distribution but not the conditional preference. Monitor the gap during training, not just loss.

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Model | DistilBERT + 4-class head | Tool classification from query text |
| SFT Training | HuggingFace Trainer | Initial supervised fine-tuning on 60 labeled queries |
| DPO Training | Custom PyTorch loop + TRL DPOConfig | Preference-aligned fine-tuning from human feedback |
| Agent | LangGraph (stateful graph) | Orchestrates route → execute → respond pipeline |
| Backend | FastAPI + Pydantic | REST API with lifespan model loading and thread-safe feedback writes |
| Frontend | Next.js 14 (App Router) | Chat interface, feedback UI, metrics dashboard |
| Embeddings | HuggingFace Transformers | DistilBERT tokenizer + `AutoModelForSequenceClassification` |
| Data | pandas · scikit-learn | CSV handling, stratified train/test split, accuracy scoring |

---

## Project Structure

```
rlhf-agent-tool-use/
│
├── router/
│   └── tool_router.py          # ToolRouter class — load, predict, save; shared TOOLS/LABEL2ID constants
│
├── tools/
│   ├── faq_search.py           # Mock FAQ lookup (keyword matching)
│   ├── order_status.py         # Mock order tracking (regex order_id extraction)
│   ├── refund_policy.py        # Mock refund policy retrieval
│   └── raise_ticket.py         # Mock ticket creation with confirmation number
│
├── agent/
│   └── langgraph_agent.py      # 3-node LangGraph: route_query → execute_tool → format_response
│
├── api/
│   ├── main.py                 # FastAPI app — 5 endpoints, CORS, logging middleware, lifespan
│   └── agent_service.py        # Singleton service layer — owns routers, feedback persistence, evaluation
│
├── training/
│   ├── train_sft.py            # Stage 1–5 SFT pipeline: CSV → split → tokenise → Trainer → save
│   └── run_dpo.py              # Stage 1–5 DPO pipeline: validate pairs → format → custom loss loop → eval
│
├── evaluation/
│   └── compare_baseline_vs_dpo.py  # Offline confusion matrix + per-tool accuracy comparison
│
├── data/
│   ├── queries.csv                  # 60 labelled queries (15 per tool, stratified)
│   ├── feedback.csv                 # Human ratings collected at runtime (append-only)
│   ├── preference_pairs.jsonl       # DPO training pairs: {prompt, chosen, rejected}
│   └── evaluation_results.json      # Pre-computed SFT vs DPO metrics (written by run_dpo.py)
│
├── models/
│   ├── sft_router/             # HuggingFace checkpoint after SFT training
│   └── dpo_router/             # HuggingFace checkpoint after DPO fine-tuning
│
├── requirements.txt
└── README.md
```

---

## Setup & Run

### Prerequisites

- Python 3.10+
- Node.js 18+ (for the frontend)
- ~2 GB disk (DistilBERT weights)

### 1 — Install Python dependencies

```bash
cd rlhf-agent-tool-use
pip install -r requirements.txt
```

### 2 — Train the SFT model (~3 min on CPU)

```bash
python training/train_sft.py
# → saves models/sft_router/
# → prints per-epoch accuracy on the held-out test set
```

### 3 — Start the FastAPI backend

```bash
uvicorn api.main:app --reload --port 8000
# API docs → http://localhost:8000/docs
```

### 4 — Start the Next.js frontend

```bash
cd ../rlhf-agent-frontend
npm install
npm run dev
# → http://localhost:3000
```

### 5 — Collect feedback, then run DPO (≥ 3 pairs required)

Use the chat interface at `http://localhost:3000` to send queries and rate the agent's tool selections with thumbs-up / thumbs-down. Thumbs-down votes where you specify the correct tool are automatically written to `data/preference_pairs.jsonl`.

Once you have enough pairs:

```bash
cd rlhf-agent-tool-use
python training/run_dpo.py
# → trains models/dpo_router/
# → writes data/evaluation_results.json
# → prints SFT vs DPO comparison table
```

### 6 — (Optional) Offline evaluation with confusion matrices

```bash
python evaluation/compare_baseline_vs_dpo.py
# → prints confusion matrices and per-tool accuracy for both models
```

---

## Key Concepts

### What is RLHF?

Reinforcement Learning from Human Feedback replaces a hand-crafted reward function with direct human signal. Humans rate model outputs; those ratings become a training signal that steers the model toward preferred behavior — in this project, selecting the right customer-support tool.

### DPO vs PPO

| Aspect | PPO (traditional RLHF) | DPO (this project) |
|--------|----------------------|-------------------|
| Reward model | Explicit neural reward model | Implicit — derived from preference pairs |
| Training loop | 4 models simultaneously | 2 models (policy + frozen reference) |
| Stability | Can be unstable (reward hacking) | More stable — direct on log-prob ratios |
| Compute | High (RL rollouts) | Low (supervised-style backward pass) |

### DPO Loss (adapted for classifiers)

```
L_DPO = -log σ( β · [(log π(chosen|x) − log π_ref(chosen|x))
                     − (log π(rejected|x) − log π_ref(rejected|x))] )
```

For a sequence classifier, `log π(tool|x)` is the log-softmax score for that tool's class index. We implement this directly on DistilBERT logits instead of using TRL's `DPOTrainer` (which targets causal language models only).

### Why Tool Selection Matters

In agentic systems, routing to the wrong tool causes compounding errors — a refund question sent to `order_status` returns irrelevant data, producing a wrong response. Even a single misrouted class (`faq_search` at 67% SFT accuracy) cascades into poor user experience across ~33% of FAQ queries. RLHF closes the loop between user dissatisfaction and model improvement without requiring manual re-labeling of the full dataset.

---

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness check — returns model load status |
| `POST` | `/api/query` | Route query through LangGraph agent |
| `POST` | `/api/feedback` | Save human rating and create DPO pair if wrong |
| `GET` | `/api/feedback/stats` | Aggregate feedback statistics |
| `GET` | `/api/evaluation` | SFT vs DPO accuracy on held-out test set |

---

## Future Improvements

- **Larger base model** — Replace DistilBERT with Mistral-7B or Llama-3 for open-domain tool descriptions and richer representations
- **RLAIF** — Replace human raters with a critic LLM (Constitutional AI style) to scale feedback collection without human bottlenecks
- **Explicit reward model** — Train a separate `RewardModel` on preference pairs; use PPO against it for finer-grained optimization
- **More tools** — Expand to 10+ tools and evaluate whether DPO maintains accuracy under a larger action space
- **Online learning** — Stream preference pairs into a continual DPO loop rather than batch re-training after collection periods
- **SageMaker deployment** — Package SFT + DPO training as SageMaker Training Jobs; serve via SageMaker endpoint with A/B routing between SFT and DPO models
- **Retrieval-augmented tools** — Connect `faq_search` to a real vector store (FAISS / Pinecone) with actual knowledge-base documents
