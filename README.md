# AI Research Intelligence System

A multi-agent, fully open-source RAG pipeline that researches a topic end-to-end - searching the web, ingesting and chunking content, answering questions with a fine-tuned local LLM using advanced retrieval techniques, and generating a structured research report - all orchestrated with LangGraph and observable through LangSmith and MLflow.

This is the second half of a two-part portfolio project. Part 1 fine-tuned **Qwen3-4B** (via LoRA) on AI research paper Q&A and pushed the adapter to HuggingFace Hub. This system loads that adapter and uses it as the RAG answering brain.

---

## Architecture

```
                              ┌─────────────────────────┐
                              │   Streamlit UI (app.py) │
                              │  topic + questions input│
                              │  advanced RAG toggles   │
                              └────────────┬────────────┘
                                           │
                              ┌────────────▼─────────────┐
                              │   LangGraph Orchestrator │
                              │  (conditional routing +  │
                              │   retry + checkpointing) │
                              └────────────┬─────────────┘
                                           │
        ┌──────────────────────────────────┼──────────────────────────────────┐
        │                                  │                                  │
        ▼                                  ▼                                  ▼
┌─────────────────┐               ┌─────────────────────┐              ┌─────────────────────┐
│ Agent 1         │               │ Agent 2             │              │ Agent 3             │
│ Web             │──documents──▶ │ Ingestion           │──chunks────▶│ RAG Analysis         │
│ Intelligence    │               │                     │              │                     │
│                 │               │ • Markdown/         │              │ 1. Query rewrite    │
│ • Tavily search │               │   RecursiveChar     │              │ 2. Query expand x3  │
│ • Tavily crawl  │               │   splitter          │              │ 3. Retrieve x4      │
│ • Format detect │               │ • Hierarchical      │              │ 4. RRF merge        │
│ • MCP save      │               │   parent/child      │              │ 5. Cross-encoder    │
│                 │               │   chunks            │              │    rerank           │
│                 │               │ • bge-small-en-v1.5 │              │ 6. Parent chunk     │
│                 │               │   embeddings        │              │    lookup           │
│                 │               │ • Qdrant storage    │              │ 7. Qwen3-4B (LoRA)  │
│                 │               │                     │              │    answer           │
│                 │               │                     │              │ 8. Eval (Ragas via  │
│                 │               │                     │              │    Groq gpt-oss)    │
└───────┬─────────┘               └──────────┬──────────┘              └──────────┬──────────┘
        │                                    │                                    │
        │              retry on failure (max 3, conditional edge)                 │
        └────────────────────────────────────┘                                    │
                                                                                    ▼
                                                                        ┌─────────────────────┐
                                                                        │ Agent 4             │
                                                                        │ Report Generation   │
                                                                        │                     │
                                                                        │ • Groq gpt-oss-120b │
                                                                        │   narrative synth   │
                                                                        │ • Eval metrics table│
                                                                        │ • MCP save (.md)    │
                                                                        │ • fpdf2 export (.pdf)│
                                                                        └──────────┬──────────┘
                                                                                   │
                              ┌────────────────────────────────────────────────────┘
                              ▼
                    ┌─────────────────────────┐        ┌────────────────────┐
                    │  Observability          │        │  Custom MCP Server │
                    │  • LangSmith (full      │        │  (stdio, 2 tools)  │
                    │    trace: rewrite→      │        │  • save_research_  │
                    │    expand→retrieve→     │        │    document        │
                    │    RRF→rerank→generate) │        │  • save_report     │
                    │  • MLflow (naive vs.    │        └────────────────────┘
                    │    advanced retrieval,  │
                    │    Ragas generation     │
                    │    scores, over time)   │
                    └─────────────────────────┘
```

---

## Tech Stack

| Component | Technology                                         | Version |
|---|----------------------------------------------------|---------|
| Language / Package Manager | Python + `uv`                                      | 3.11    |
| Orchestration | LangGraph                                          | 1.2+    |
| LLM Framework | LangChain                                          | 1.3+    |
| RAG Answering Model | Qwen3-4B-Instruct-2507 + LoRA adapter (PEFT)       | -       |
| Quantization | bitsandbytes (4-bit NF4)                           | 0.50+   |
| Vector Database | Qdrant (self-hosted, Docker)                       | 1.19+   |
| Embeddings | BAAI/bge-small-en-v1.5 (open-source, CPU)          | -       |
| Reranker | cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU)         | -       |
| Web Search / Crawl | Tavily                                             | -       |
| Report / Judge LLM | Groq - `openai/gpt-oss-120b` (open-weight, hosted) | -       |
| Generation Evaluation | Ragas                                              | 0.3.9   |
| Tool Protocol | MCP (Model Context Protocol)                       | 1.28+   |
| Tracing | LangSmith                                          | -       |
| Experiment Tracking | MLflow                                             | 3.15+   |
| UI | Streamlit                                          | 1.61+   |
| PDF Export | fpdf2                                              | -       |
| Containerization | Docker (CUDA 12.1 base image)                      | -       |

**Fully open-weight / self-hostable by design:** the only hosted API calls are Tavily (search), HuggingFace Hub (model download), and Groq (fast inference of an open-weight model used as report writer / eval judge - not the RAG brain).

---

## Advanced RAG Techniques

**Query Rewriting** - The user's raw question is rewritten by the fine-tuned model into a concise, keyword-dense search query before retrieval, stripping filler words search engines and embedding models respond poorly to.

**Query Expansion** - 3 additional query variants are generated, each covering a different angle (more technical/mechanistic phrasing, a comparison angle, a broader conceptual angle), generated via **separate single-shot LLM calls** rather than one multi-line completion - small models are unreliable at multi-item list formatting, so this trades a bit of latency for much higher reliability.

**Hierarchical Chunking** - Documents are split into 1500-character parent chunks and 400-character child chunks, linked by `parent_id`. Retrieval and reranking operate on child chunks (precise, embedding-friendly), but the parent chunk is what's actually passed to the LLM as context - giving the generator more surrounding information than the exact matched snippet.

**Reciprocal Rank Fusion (RRF)** - Results from all 4 queries (1 rewritten + 3 expanded) are merged using RRF: `score = Σ 1 / (k + rank)` across each query's ranked list, with `k=60`. This rewards chunks that rank well across multiple query phrasings, rather than trusting any single query's ranking.

**Cross-Encoder Reranking** - The top 20 RRF-merged chunks are re-scored by `cross-encoder/ms-marco-MiniLM-L-6-v2`, which jointly encodes (query, chunk) pairs for much higher-precision relevance scoring than embedding similarity alone. Runs on CPU so the GPU stays free for the LLM.

All 3 techniques are individually toggleable in the Streamlit UI for live before/after comparison.

---

## Evaluation Results

Retrieval metrics require ground-truth relevance labels, which don't exist for arbitrary live questions - so these are measured against a small hand-built benchmark set (3 questions with reference answers and relevance keywords), run via `test_benchmark.py` / `benchmark_runner.py`. This mirrors real-world RAG evaluation practice, where retrieval quality is benchmarked offline, not scored live.

### Retrieval: Naive vs. Advanced RAG

| Metric | Naive (single query) | Advanced (rewrite+expand+RRF+rerank) | Improvement |
|---|---|---|---|
| MRR@10 | 0.6667 | 0.8333 | +25.0% |
| nDCG@10 | 0.5615 | 0.7852 | +39.8% |
| Precision@5 | 0.5333 | 0.6667 | +25.0% |
| Recall@10 | 0.3667 | 0.7088 | +93.3% |

Advanced RAG techniques improved every retrieval metric, with the largest gain in Recall - expanding one query into four distinct phrasings surfaces relevant chunks that a single query's wording misses entirely.

### Generation: Ragas Metrics (judged by Groq `gpt-oss-120b`)

| Question | Faithfulness | Context Precision | Answer Relevancy |
|---|---|---|---|
| What is retrieval augmented generation? | ~1.0 | ~1.0 | ~0.69–0.88 |
| What are common challenges with RAG systems? | ~0.75–1.0 | ~0.5–1.0 | ~0.77–0.79 |
| How does chunking affect RAG performance? | ~1.0 | ~0.0 | ~0.78 |

*Ranges reflect run-to-run variance from the hosted judge LLM, which is not perfectly deterministic even at `temperature=0` (see Limitations). The Context Precision = 0.0 result for the chunking question is a genuine finding, not a bug: the model's answer to that question was accurate but noticeably terser than the reference answer, so Ragas scored the retrieved context as under-utilized - see "Why a fine-tuned model instead of GPT-4" below.*

*Live pipeline runs (arbitrary topics/questions typed into the UI) intentionally do not display a per-question evaluation table in the generated report - those metrics require ground-truth relevance labels or reference answers that don't exist for an arbitrary question. Evaluation only runs against the benchmark set above, via `test_benchmark.py`, with results tracked in MLflow.*

---

The RAG-answering step uses the project's own LoRA-fine-tuned Qwen3-4B rather than a frontier API model, for three reasons: (1) it demonstrates the full ML lifecycle - fine-tuning *and* deploying that model in a real application - rather than just prompting someone else's model; (2) a 4-bit quantized 4B model runs entirely on a 6GB consumer GPU, at zero marginal cost per query, which matters for a project meant to be runnable by anyone cloning the repo; (3) it surfaces real, interesting trade-offs (documented in Limitations below) that are genuinely instructive - e.g., the fine-tuned model's strong bias toward terse answers, likely inherited from concise Q&A training data, which persists even when the system prompt explicitly asks for more detail. That's a legitimate fine-tuning characteristic worth understanding, not something a frontier model would ever expose.

## Why LangGraph Instead of a Simple Chain

The pipeline needs real branching behavior a linear chain can't express cleanly: Agent 1 can fail (Tavily returns nothing), and the system needs to retry up to 3 times before giving up - a conditional loop, not a straight line. LangGraph's explicit state graph makes this routing visible and testable (`route_after_web_intelligence`), and its checkpointing (`MemorySaver`) means the pipeline is resumable rather than an all-or-nothing script. A simple chain would need custom control-flow code to replicate what LangGraph gives natively.

## Why Hierarchical Chunking Over Flat Chunking

Small chunks (400 chars) are better for retrieval - they embed more precisely and reduce noise in similarity search - but they're often too little context for the LLM to answer well from. Large chunks (1500 chars) are better context but dilute embedding precision, causing worse retrieval. Hierarchical chunking gets both: search matches happen on the precise child chunks, but the LLM is given the parent chunk's fuller context - measurably improving Recall in the benchmark above, without sacrificing answer quality.

---

## Setup Instructions (uv)

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker Desktop (for Qdrant, and optionally for the full containerized app)
- NVIDIA GPU with 6GB+ VRAM (tested on RTX 2060), CUDA-capable driver
- Accounts (all free tier): [Tavily](https://tavily.com), [HuggingFace](https://huggingface.co/settings/tokens), [LangSmith](https://smith.langchain.com), [Groq](https://console.groq.com/keys)

### Install
```bash
git clone <this-repo>
cd ai-research-intelligence-system
uv sync
```

### Environment variables
Copy `.env.example` to `.env` and fill in:
```
TAVILY_API_KEY=
HF_TOKEN=
HF_MODEL_ID=<your-username>/<your-lora-adapter-repo>
LANGSMITH_API_KEY=
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=ai-research-intelligence-system
GROQ_API_KEY=
QDRANT_URL=http://localhost:6333
```

### Start Qdrant
```bash
docker run -d --name qdrant-db -p 6333:6333 -p 6334:6334 \
  -v ${PWD}/qdrant_storage:/qdrant/storage \
  --restart unless-stopped \
  qdrant/qdrant
```

### Loading the fine-tuned model (4-bit, RTX 2060 6GB)
The model is a **LoRA adapter**, not a full checkpoint - it's loaded on top of the base `Qwen/Qwen3-4B-Instruct-2507`, quantized to 4-bit NF4 via `bitsandbytes`:
```python
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-4B-Instruct-2507", quantization_config=quant_config, device_map="auto"
)
model = PeftModel.from_pretrained(base_model, HF_MODEL_ID)
```
Full weights + adapter fit comfortably in 6GB VRAM with headroom to spare.

### Run the full pipeline (UI)
```bash
uv run streamlit run src/app.py
```
Open `http://localhost:8501`, enter a research topic and questions, and run.

### Run the retrieval + generation benchmark
```bash
uv run python test_benchmark.py
```

### View traces / experiments
- LangSmith: https://smith.langchain.com → project `ai-research-intelligence-system`
- MLflow: `uv run mlflow ui --port 5001` → `http://localhost:5001`

### Docker (optional, full containerized app)
```bash
docker build -t ai-research-intelligence-system .
docker run --rm --gpus all -p 8501:8501 --env-file .env ai-research-intelligence-system
```
> **Known issue:** the container currently fails to start due to a chain of Linux-specific dependency conflicts in the `transformers`/`torch`/`torchvision` stack. Root-caused through several iterations: (1) `transformers`' newer FP8/DeepGEMM/MoE integrations trigger a `torch._dynamo` duplicate-registration bug on Linux that never surfaces on Windows; (2) pinning `transformers` down to avoid it hits a `huggingface-hub` version floor conflict with other dependencies; (3) manually patching or reinstalling `torch` inside the image breaks `torchvision`'s binary compatibility; (4) `uv run`'s automatic re-sync at container start was also silently undoing build-time fixes, which took real effort to diagnose. Each root cause was real and reproducible - this is a genuine current fragility in the `transformers`/`torch` Linux dependency graph, not a project design flaw. Deferred as future work; see below.

---

## Screenshots

*Streamlit UI - pipeline input and live agent progress:*
![screenshot 1](https://github.com/Chris-Cruz08/Fine_tuned-Agentic_RAG-Research-Assistant-Complete/blob/929239af61e9e3ee599a1b30b417e8b1f74eb274/images/sc2.png)

*Streamlit UI - RAG answers and evaluation tabs:*
![screenshot 2](https://github.com/Chris-Cruz08/Fine_tuned-Agentic_RAG-Research-Assistant-Complete/blob/929239af61e9e3ee599a1b30b417e8b1f74eb274/images/sc2.1.png)

*LangSmith trace - full nested pipeline (rewrite → expand → retrieve → RRF → rerank → generate):*
![](https://github.com/Chris-Cruz08/Fine_tuned-Agentic_RAG-Research-Assistant-Complete/blob/929239af61e9e3ee599a1b30b417e8b1f74eb274/images/sc3.png)
![](https://github.com/Chris-Cruz08/Fine_tuned-Agentic_RAG-Research-Assistant-Complete/blob/929239af61e9e3ee599a1b30b417e8b1f74eb274/images/sc3.1.png)

*MLflow - naive vs. advanced retrieval comparison:*
![](https://github.com/Chris-Cruz08/Fine_tuned-Agentic_RAG-Research-Assistant-Complete/blob/929239af61e9e3ee599a1b30b417e8b1f74eb274/images/sc4.png)

---

## Limitations and Future Directions

- **Repeated-list generation failure**: on at least one tested question, the model's answer degenerated into the same list of items repeated multiple times, likely due to greedy decoding (`do_sample=False`, used for reproducible benchmarking) on a longer generation. A `repetition_penalty` parameter would likely fix this and is a planned tune.
- **Fine-tuned model's terseness bias**: the model tends toward short, factual answers even when explicitly prompted for more detail - likely inherited from concise Q&A pairs in the fine-tuning dataset. This occasionally under-utilizes available context (visible in the Context Precision = 0.0 result above). Worth revisiting with more varied-length training examples in Part 1.
- **Hosted judge non-determinism**: Groq's `gpt-oss-120b`, even at `temperature=0`, doesn't guarantee fully deterministic scoring across runs (a known characteristic of batched inference on most hosted LLM APIs, including OpenAI). Generation metric numbers above are representative, not exact-reproducible to the decimal.
- **Retrieval metrics require a hand-built benchmark**: MRR/nDCG/Precision/Recall can't be computed on arbitrary live questions without ground-truth relevance labels. A future version could use LLM-as-judge relevance labeling to generate a larger, less manually-curated benchmark set automatically.
- **Docker + Linux triton/dynamo bug**: documented above; needs one more round of debugging (likely an exact `transformers` version pin) before the containerized deployment path is fully solid.
- **What I'd improve with more compute and time**: a larger, GPU-hosted judge model instead of relying on Groq's free tier limits; a proper held-out benchmark set (20-50+ questions) instead of 3 hand-labeled ones; hybrid retrieval (dense + BM25) on top of the current pure-vector approach; disk-backed LangGraph checkpointing (currently in-memory, resets between process restarts) for true pipeline resumability across sessions; and a second LoRA fine-tuning pass specifically targeting the terseness/verbosity trade-off surfaced during evaluation.
