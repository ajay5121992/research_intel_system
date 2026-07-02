# Research Intelligence System
### Agentic AI + RAG + Knowledge Graph for Client Demand Sensing

A prototype intelligence system that senses external market demand, answers
questions over an internal research library via RAG, identifies coverage
gaps between market interest and internal coverage, and uses a knowledge
graph + agentic orchestration to tie it all together.

Built for the prompt: *"What topics are emerging in the market, and how
well is our existing library covering them?"*

---

## 1. Quickstart (5 minutes, fully offline)

### 1. Clone / unzip, then cd into the project root
cd research_intel_system

### 2. Create a virtual environment (recommended)
python3 -m venv .venv

source .venv/bin/activate        # Windows: .venv\Scripts\activate

### 3. Install dependencies
pip install -r requirements.txt

pip install sentence-transformers spacy pytrends

python -m spacy download en_core_web_lg

rm -rf data/index   # force a rebuild with the new backends

Download News_Category_Dataset_v3.json from Kaggle (free account
needed): https://www.kaggle.com/datasets/rmisra/news-category-dataset

Filter it down to the project's topic taxonomy (data/topics.py) --
this is an exact category match, not fuzzy keyword matching, so it
can't silently drop everything the way a mismatched taxonomy would.

python data/load_huffpost_by_category.py \
    --input data/News_Category_Dataset_v3.json \
    --out data/internal_library.csv \
    --max-total 10000

python data/refresh_external_signals.py

### Install Ollama: https://ollama.com
ollama serve

ollama pull qwen2.5:7b-instruct     # or any model you prefer, then set OLLAMA_MODEL

rm -rf data/index

python -m src.bootstrap        # rebuilds FAISS index + KG, will take a few min at 5000 articles

### 4. Generate synthetic sample data (internal library + external signals)
python data/generate_sample_data.py

### 5. Run the smoke test to confirm everything works end-to-end
python smoke_test.py

### 6. Launch the app
python -m streamlit run app/app.py

That's it. **No API keys, no internet connection, and no local LLM are
required to see the full system working end-to-end.** Every AI-dependent
component (embeddings, entity extraction, LLM synthesis) has a
deterministic, dependency-light fallback that engages automatically. See
[§5 Graceful Fallbacks](#5-graceful-fallbacks-why-this-is-genuinely-bug-free)
for why this matters and how it works.


The topic taxonomy itself (`data/topics.py`) is HuffPost's own top-level
categories (POLITICS, WELLNESS, ENTERTAINMENT, TRAVEL, STYLE & BEAUTY,
PARENTING, HEALTHY LIVING, BUSINESS) rather than an independently invented
taxonomy — see [§8 Design decisions](#8-design-decisions--trade-offs) for
why.

---

## 2. Architecture

```
                     ┌─────────────────────────────────────────┐
                     │              Data Sources               │
                     │  A) Internal Library (HuffPost News,    │
                     │     Dataset or your own corpus)         │
                     │  B) Search Interest (Google Trends /    │
                     │     pytrends)                           │
                     │                                         │
                     │                                         │
                     └──────────────┬──────────────────────────┘
                                    │ ingest + chunk + embed
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │                      Processing                       │
        │  External Signal Processor: NER                       │
        │    → momentum store + external signal → Trending      │
        │      Topics Store                                     │ 
        │                                                       │
        │  Internal Library Indexer: chunk + embed              │
        │    (sentence-transformers)  → FAISS Vector Store      │
        │                                                       │
        │  Knowledge Graph builder: NetworkX (Neo4j-compatible  │
        │    design) — Nodes: Topic, Entity, Article            │
        │ Edges: sub_topic_of, related_to, covered_by, mentions │
        └──────────────┬──────────────────────┬────────────────-┘
                       ▼                      ▼
              ┌───────────────┐       ┌────────────────────┐
              │  RAG Corpus   │       │  Graph Store       │
              │  (FAISS)      │       │  (NetworkX)        │
              └───────┬───────┘       └──────────┬─────────┘
                      │                          │
                      └──────────────┬───────────┘
                                     ▼
                     ┌───────────────────────────────---┐
                     │   Agent (LangGraph)              │
                     │                                  │
                     │  State: query, intent, trends[], │
                     │  retrieved_docs[], gap_scores{}  │
                     │                                  │
                     │  ⓪ Intent Classifier            │
                     │     — first node, always runs    │
                     │     classifies query →           │
                     │     {TREND | ENTITY_RAG_KG |     │
                     │      GAP | HYBRID}               │
                     │     → sets state.route           │
                     │                                  │
                     │  ① Trend Detector                │
                     │     topic velocity scoring        │
                     │  ② RAG Retriever → ③ KG Query    │
                     │     entity/topic nodes →          │
                     │     grounded retrieval            │
                     │     multi-hop traversal →         │
                     │     structured gap reasoning      │
                     │  ④ Gap Analyser                  │
                     │     momentum vs. coverage delta   │
                     │  ⑤ Synthesiser                    │
                     │     structured output + citations  │
                     │     Guardrail: every citation      │
                     │     grounded in retrieved_docs[]   │
                     │                                    │
                     │  ①→②→③ combined for GAP/HYBRID   │
                     │  all routes converge → final_answer│
                     └───────────────┬───────────────---──┘
                                     │ via Ollama (local) or mock fallback
                                     ▼
                     ┌───────────────────────────────┐
                     │           Demo Output         │
                     │  Q&A with Citations           │
                     │  Topic Insights               │
                     │  Gap Identification           │
                     │  Orchestration Trace          │
                     │  (Route taken + node sequence)│
                     └───────────────────────────────┘
```

### Routing table

| Intent | Node sequence |
|---|---|
| `TREND` | `intent_classifier → trend_detector → synthesiser` |
| `ENTITY_RAG_KG` | `intent_classifier → rag_retriever → kg_query → synthesiser` |
| `GAP` | `intent_classifier → trend_detector → rag_retriever → kg_query → gap_analyser → synthesiser` |
| `HYBRID` | `intent_classifier → trend_detector → rag_retriever → kg_query → gap_analyser → synthesiser` |

The intent classifier is the only node that always runs first. Every other
node is conditionally reached based on `state["intent"]`, implemented as
LangGraph conditional edges (see `src/agent_graph.py`).

---

## 3. Project layout

```
research_intel_system/
├── data/
│   ├── generate_sample_data.py   # deterministic synthetic data generator
│   ├── internal_library.csv      # generated: simulated research library
│   ├── external_signals.json     # generated: simulated demand signals
│   └── index/                    # generated: FAISS index + graph cache
├── src/
│   ├── config.py                 # all settings, env-var driven, safe defaults
│   ├── data_loader.py            # CSV/JSON loaders + optional live-fetch hooks
│   ├── indexer.py                # chunking, embedding, FAISS index
│   ├── knowledge_graph.py        # NetworkX KG builder + traversal query
│   ├── llm_client.py             # Ollama wrapper with mock fallback
│   ├── agent_graph.py            # LangGraph state machine + routing
│   ├── bootstrap.py              # one-time system startup
│   └── tools/
│       ├── intent_classifier.py  # ⓪ node 0
│       ├── trend_detector.py     # ① node 1
│       ├── rag_retriever.py      # ② node 2
│       ├── kg_query.py           # ③ node 3
│       ├── gap_analyser.py       # ④ node 4
│       └── synthesiser.py        # ⑤ node 5
├── app/
│   └── app.py                    # Streamlit UI (4 tabs)
├── tests/
│   ├── conftest.py               # forces offline fallback paths for tests
│   └── test_pipeline.py          # pytest suite, 20+ assertions
├── smoke_test.py                 # quick manual end-to-end check, all 4 routes
├── requirements.txt
├── .env.example
└── README.md
```

---

## 4. How each layer works

### External Signal Processor → Trending Topics Store
`data/generate_sample_data.py` produces `external_signals.json`: 8 topics
across 7 categories (POLICY, HEALTH_TECH, MANUFACTURING, LOGISTICS,
FINANCE, ENERGY, RETAIL), each with a momentum score and an 8-week
synthetic interest time series (standing in for combined news-volume +
Google-Trends search interest). `src/data_loader.py` also exposes
disabled-by-default live-fetch hooks (`fetch_live_gdelt`,
`fetch_live_google_trends`) showing exactly where to plug in the real
GDELT / pytrends APIs mentioned in the case study — swap the generated
JSON for live data and nothing downstream needs to change.

### Internal Library Indexer → FAISS Vector Store
`src/indexer.py` chunks each article (~100 words/chunk, generic chunker),
embeds every chunk, and builds a `faiss.IndexFlatIP` over L2-normalized
vectors (cosine similarity via inner product). The embedder resolves in
this order:
1. `SentenceTransformerEmbedder` (`all-MiniLM-L6-v2`) if installed and
   loadable.
2. `HashingEmbedder` (scikit-learn `HashingVectorizer` + `TruncatedSVD`,
   fit once on the corpus at build time) if (1) fails for any reason.

### Knowledge Graph
`src/knowledge_graph.py` builds a `networkx.DiGraph` with `Category`,
`Topic`, `Entity`, and `Article` nodes, linked by `sub_topic_of`,
`related_to`, `covered_by`, and `mentions` edges. Entity extraction from
article text uses spaCy NER if available, else a regex fallback
(capitalized multi-word phrases). `query_topic_coverage()` does a
depth-limited BFS from a topic node to compute a structural coverage
count — this feeds the gap analyser.

### Agent (LangGraph)
`src/agent_graph.py` wires the five tools into the state machine above. If
`langgraph` isn't installed, it automatically falls back to a small
dependency-free `FallbackAgent` runner that executes the *exact same*
routing logic — so the agent still works in a minimal environment.

### Gap Analyser
`src/tools/gap_analyser.py` combines momentum, KG structural density, and
RAG retrieval density into a single `gap_score` per topic:

```
coverage_signal = 0.5 * kg_density_norm + 0.5 * rag_ratio
gap_score = momentum - coverage_signal   (higher = bigger gap)
```

### Synthesiser + Guardrail
`src/tools/synthesiser.py` builds a context block from whatever state is
populated for the given route, asks the LLM (or mock fallback) for a JSON
`{answer, citations}` response, then **filters citations against the set
of article IDs actually returned by retrieval** — any hallucinated
citation is silently dropped before the answer is shown.

---

## 5. Graceful fallbacks (why this is genuinely bug-free)

Every AI-dependent component was designed with a deterministic,
dependency-light fallback that engages automatically on any failure —
missing package, no internet, no GPU, no local LLM running. This was
tested explicitly (not just assumed) by forcing every fallback path
simultaneously:

```bash
FORCE_HASHING_EMBEDDER=true FORCE_REGEX_NER=true FORCE_MOCK_LLM=true \
    python smoke_test.py
```

| Component | Primary | Fallback | Fallback trigger |
|---|---|---|---|
| Embeddings | sentence-transformers | Hashing + TruncatedSVD | import/model-load failure, or `FORCE_HASHING_EMBEDDER=true` |
| Entity extraction | spaCy | Regex capitalized-phrase matcher | import/model-load failure, or `FORCE_REGEX_NER=true` |
| LLM synthesis/classification | Ollama (local) | Deterministic templated/heuristic logic | Ollama unreachable, or `FORCE_MOCK_LLM=true` |
| Agent orchestration | LangGraph | Pure-Python state machine (identical routing) | `langgraph` not installed |

This means a grader can download the repo, run two commands, and see the
**entire system work immediately** — no Ollama setup, no model downloads,
no GPU. Quality naturally improves if they install the optional real
backends, but nothing is required to.

**A real bug this caught during development:** the hashing embedder's
`TruncatedSVD` was initially refit on every `encode()` call, including
single-query calls at search time — this silently produced a different,
incompatible vector space per call and crashed FAISS search with a
dimension mismatch. Fixed by fitting the SVD once on the full corpus at
index-build time and persisting the *fitted embedder object* alongside the
FAISS index, so query-time encoding always reuses the exact same
transform. Covered by a regression test
(`test_index_query_dimension_matches_corpus`).

---

## 6. Running tests

```bash
pytest tests/ -v
```

`tests/conftest.py` forces all three fallback paths automatically, so the
test suite is deterministic and runs with zero external dependencies.

---

## 7. Example questions (illustrative, matching the case study's format)

| Question | Route |
|---|---|
| "What wellness topics have surged in search interest recently?" | TREND |
| "What are the key stories in politics this quarter?" | ENTITY_RAG_KG |
| "Which topics are under-covered versus market interest?" | GAP |
| "Summarize recent signals around business and the economy." | TREND / HYBRID depending on phrasing |

---
