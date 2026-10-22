# Paper Copilot: A Self-Directed Course in Building Agentic Systems

Version 1.1 (October 2026). Revised to work at the abstraction level of a working AI engineer: industry-standard libraries (LangChain, LangGraph, LangSmith) handle plumbing and orchestration, while you own design decisions, prompts, policies and, above all, evaluation.

## Overview

Over 13 assignments you will build **Paper Copilot**, a research assistant that ingests new papers, extracts structured "paper cards", answers questions with citations, plans multi-step research, remembers what you care about, verifies scientific claims, and asks a human before doing anything risky.

Every component is evaluated on a public benchmark with ground truth. By the end you will have one system, a results table for each module, an ablation study, and a set of written analyses you can talk through in an interview.

Constraints that shape the whole course:

* **Local open-weight models only.** No paid APIs. Everything must run on your machine or on free GPU notebooks (Kaggle, Colab).
* **Industry tools, your decisions.** Use the libraries a real team would use for plumbing and orchestration (model clients, vector stores, LangGraph, LangSmith). You own what they do not decide for you: graph design, prompts, retrieval choices, memory policy, routing logic, guardrails, and the evaluation of all of it. Prebuilt agents and chains are baselines to beat, not the system you evaluate.
* **Measure everything.** A feature without a number does not count as done.

### How each assignment is structured

* **Goals**: what you should understand at the end.
* **Problems**: numbered tasks with a short name in parentheses, e.g. `(llm_access)`. Points are for self-assessment only.
* **Deliverables**: code in `copilot/`, eval code in `evals/`, written answers in `reports/aXX.md`.
* **Done when**: a checklist. Do not move on until every box is ticked.
* **Interview angle**: questions you should be able to answer out loud afterwards.

Rough effort: 1 to 2 weeks per assignment part-time. A0 to A3 alone already make a solid portfolio project, so ship and publish after A3 even if the rest takes longer.

---

## Ground rules

### Repository layout

```
paper-copilot/
  AGENTS.md
  ASSIGNMENT.md
  pyproject.toml
  configs/              # model slots, dataset paths, eval settings
  copilot/
    infra/              # config loading, tracing setup, cache storage, small utilities
    llm/                # LLM access layer, cache-key design, model slots
    ingest/             # sources, parsing, event queue, workers
    cards/              # paper card schema and extraction
    retrieval/          # chunking, indexes, rerankers
    answer/             # grounded generation, citation parsing
    tools/              # registry and tool implementations
    agent/              # ReAct loop
    memory/             # short and long term memory
    hitl/               # confidence, abstention, approval gate, audit log
    router/             # model routing
    verify/             # claim verification and debate
    refine/             # self-refinement
    graph/              # final integration
    obs/                # tracing, metrics, alerts
  evals/
    adapters/           # one loader per benchmark
    metrics/            # your metric implementations
    subsets/            # frozen example IDs (committed)
    results/            # summary JSON per run (committed)
    cache/              # model outputs (gitignored)
  tests/
  reports/              # a00.md ... a12.md
```

### Evaluation protocol (applies to every assignment)

1. **Frozen subsets.** For each benchmark, sample example IDs once with a fixed seed and commit them to `evals/subsets/`. Default sizes: 200 dev, 200 test unless stated otherwise.
2. **Dev vs test.** Tune prompts, thresholds and hyperparameters on dev only. Run test once per final configuration. If you peek, note it in the report.
3. **Baselines.** Every result table includes at least one simple baseline from the same model.
4. **Uncertainty.** Report means with 95% bootstrap confidence intervals. Do not claim an improvement when intervals overlap heavily without saying so.
5. **Provenance.** Every results file records: git commit, model names, quantization, context length, temperature, seed, subset file, date.
6. **Cache.** All model calls go through the cache, so any table can be regenerated without a GPU.
7. **Implement metrics yourself.** Read the official metric definitions, implement them in `evals/metrics/`, and check your implementation against a handful of hand-computed examples (and against the official script where one exists).

### Model slots

Define these in `configs/models.yaml` and never hardcode model names in code:

| Slot | Purpose | Typical size |
|---|---|---|
| `SMALL` | cheap calls, routing target | 1B to 4B |
| `MEDIUM` | main reasoning model | 7B to 14B (quantized) |
| `EMBED` | embeddings | small sentence embedding model |
| `RERANK` | optional cross-encoder | small |
| `NLI` | entailment for citation and faithfulness checks | small NLI model |
| `JUDGE` | LLM judge where needed | usually `MEDIUM` |

Serve chat models behind an OpenAI-compatible endpoint (Ollama or llama.cpp for development, vLLM on a free GPU notebook for large eval runs), and access them through LangChain chat models or the OpenAI SDK pointed at that endpoint. Pick models after checking current open-weight leaderboards and your hardware, and justify the choice in `reports/a00.md`.

### Allowed libraries

Allowed from the start:
* Data and ML: `pydantic`, `numpy`, `pandas`, `scikit-learn`, `torch`, `transformers`, `sentence-transformers`, `datasets`, plotting libraries, `pytest`.
* Model access: LangChain chat models (`langchain-openai`, `langchain-ollama`) and the `openai` SDK. Use their built-in retry and timeout settings; you decide the policy (what to retry, how long to wait) and can explain it.
* Tracing: LangSmith (free developer tier; check its limits against your eval volume) or self-hosted Langfuse or Arize Phoenix.
* Retrieval: vector stores and indexes (`faiss`, Chroma, Qdrant, LanceDB or similar, directly or through LangChain integrations), `rank_bm25`.
* Storage helpers: `diskcache` or SQLite for the cache. The cache-key design is yours.
* Orchestration: LangGraph from A5 onwards, for graphs you design yourself.

Allowed only as baselines, never as the system under test: prebuilt agents (`create_react_agent`, `create_agent` and similar) and prebuilt RAG chains. Your own design must be compared against them.

Not allowed: routing libraries (the router is the assignment), and evaluation frameworks that compute your reported metrics for you. You may run Ragas, DeepEval or similar as a cross-check after implementing a metric yourself.

### Evaluation is still hand-built

The evaluation protocol above, your eval harness, dataset adapters, frozen subsets, metrics and bootstrap intervals are yours in every assignment. This is where AI engineers differ most, and what interviews probe hardest.

---

## Assignment 0: Foundations

**Goals:** a reliable way to call local models, cache outputs, run reproducible evals and see traces. Everything later depends on this.

**Problems**

1. `(setup)` (5 pts) Serve `SMALL` and `MEDIUM` behind an OpenAI-compatible endpoint, plus `EMBED` and `NLI` locally. In the report, record model names, quantization, context length, and measured tokens per second for a 512-token prompt with 256 output tokens on your hardware.
2. `(llm_access)` (15 pts) Build the LLM access layer on LangChain chat models (or the OpenAI SDK) configured from your model slots: chat with a slot, messages, temperature, max tokens, thinking on/off and an optional response format. Return text, reasoning (if any), token usage, `finish_reason`, latency and a cache-hit flag. Requirements: configure the library's retries and timeouts deliberately and justify the policy; handle thinking tokens and truncated outputs explicitly; add a disk cache keyed on every input that affects output (sampled calls also keyed on a sample index). The key design is the graded part; storage can use `diskcache` or SQLite. Write unit tests for cache behaviour.
3. `(eval_harness)` (20 pts) A generic runner with three parts: a dataset adapter (load, split, frozen subsets), a system under test (any callable), and metric functions. Output one JSONL row per example plus a summary JSON with bootstrap CIs and provenance. It should run from a single command with task, system and subset as arguments. Build this yourself.
4. `(tracing)` (10 pts) Set up LangSmith (or self-hosted Langfuse or Phoenix). Every LLM call becomes a span with model, tokens, latency and cache hit. Eval runs group their spans under one run ID, with the example ID as metadata.
5. `(oracle_baseline)` (20 pts) On QASPER, run two baselines with `MEDIUM`: closed-book (question only) and oracle (question plus the gold evidence paragraphs). Implement QASPER answer F1 yourself. The oracle number is your ceiling for A3.

**Written (`reports/a00.md`)** (10 pts)
* Model choices and hardware numbers.
* Closed-book vs oracle gap, broken down by QASPER answer type. What does the gap tell you about where errors will come from later: retrieval or reading?

**Done when**
- [ ] Re-running the oracle eval from cache takes under a minute and reproduces identical numbers.
- [ ] Traces for an eval run are visible in the tracing UI.
- [ ] Your F1 matches hand computation on 5 examples.

**Interview angle:** How do you make LLM evaluations reproducible? Why bootstrap intervals? What belongs in a cache key? Which LLM API errors should be retried, and why does backoff use jitter?

---

## Assignment 1: Ingestion and the event pipeline

**Goals:** turn papers into clean, structured text through an event-driven pipeline that survives duplicates, crashes and bad input. (Project idea 8: event-triggered automation.)

**Problems**

1. `(paper_model)` (5 pts) A `Paper` model: ID, version, title, authors, abstract, sections (heading plus paragraphs), source, fetch time.
2. `(sources)` (15 pts) Two sources. (a) QASPER papers loaded from the dataset. (b) Live arXiv: poll the arXiv API for configured categories (for example `cs.CL`, `cs.LG`), respecting arXiv's API terms (at most one request every 3 seconds).
3. `(parsing)` (10 pts) For arXiv papers, use the HTML version when available and fall back to PDF parsing. Keep section structure.
4. `(queue)` (25 pts) Events (`paper_published`, `paper_updated`) go into a durable queue (an SQLite table is fine) and are processed by workers. Requirements: an idempotency key of (paper ID, version); at-least-once delivery with idempotent handlers; retries with exponential backoff and jitter; a maximum attempt count after which the event goes to a dead-letter table; a command to inspect and replay dead letters.
5. `(replay_eval)` (20 pts) Record a fixture of 300 real arXiv events. Build a replay harness that injects 20% duplicate events, 5% malformed payloads, and a handler that fails transiently 10% of the time. Then kill a worker process mid-batch and restart it.

**Metrics:** duplicate processing rate (target 0), eventual success rate for transient failures (target 100%), dead-letter precision (only genuinely malformed events end up there), throughput, and section-structure quality on 20 hand-checked papers (HTML vs PDF).

**Written** (10 pts)
* Why at-least-once plus idempotency instead of exactly-once?
* What did the crash test reveal?
* HTML vs PDF parsing: which failure modes did you see?

**Done when**
- [ ] The kill-and-restart test produces no lost and no duplicated papers.
- [ ] A poisoned event ends up in the dead-letter table and can be replayed after fixing it.

**Interview angle:** at-least-once vs exactly-once delivery, idempotency keys, backoff with jitter, dead-letter queues.

---

## Assignment 2: Paper cards (structured output)

**Goals:** get reliable structured data out of a model, and understand the trade-offs between validation-and-retry and constrained decoding. (Project idea 1.)

**Problems**

1. `(schema)` (10 pts) A `PaperCard` Pydantic model: tasks, datasets, methods, metrics, and results (each with dataset, metric, value, method), plus an optional code URL. Add validators where they catch real errors (numeric values, non-empty strings, no duplicates).
2. `(strategies)` (25 pts) Three extraction strategies:
   * S1: prompt-only, parse JSON.
   * S2: validate, and on failure retry with the validation error fed back (maximum 3 attempts). Log every failure with its error type.
   * S3: constrained decoding using your serving stack's JSON schema or grammar support (LangChain's `with_structured_output` in JSON-schema mode is fine here). Write S1 and S2 yourself so you can see and log every failure.
3. `(long_docs)` (15 pts) Papers do not fit in context. Implement chunked extraction and a merge step that deduplicates entities (aliases, casing, abbreviations).
4. `(scirex_eval)` (20 pts) Evaluate on SciREX, which has gold salient entities of type Task, Dataset, Method and Metric. Count a predicted entity as correct if it matches a mention of a gold cluster after normalization. Run S1, S2 and S3 with both `SMALL` and `MEDIUM`.

**Metrics:** per-field precision, recall and F1; first-attempt schema-valid rate; final valid rate; mean retries; latency per paper.

**Written** (15 pts)
* Which errors did retries fix, and which did they not?
* Did constrained decoding change content quality, not just validity? Show examples.
* How did your merge strategy affect precision vs recall?

**Done when**
- [ ] A results table with 3 strategies times 2 models, with CIs.
- [ ] Every paper ingested in A1 gets a card, and failures are logged rather than crashing the pipeline.

**Interview angle:** why validate model output, constrained decoding trade-offs, handling documents longer than context.

---

## Assignment 3: Retrieval and grounded answers with citations

**Goals:** build retrieval you can measure, and generation where every claim points to evidence. (Project idea 2.)

**Problems**

1. `(chunking)` (10 pts) Two chunkers: fixed-size token windows with overlap, and structure-aware (section and paragraph). Chunk IDs must be stable across re-ingestion.
2. `(retrievers)` (20 pts) BM25, dense (`EMBED` plus a vector store such as FAISS or Chroma, directly or through LangChain), and hybrid with reciprocal rank fusion. Implement the fusion yourself; prebuilt ensemble retrievers are a baseline only. Optional: cross-encoder reranking with `RERANK`.
3. `(retrieval_eval)` (15 pts) On QASPER dev, measure evidence recall@k (k = 1, 3, 5, 10) and MRR, retrieving within the question's paper. Stretch: retrieve across the whole library, which adds a paper-selection step.
4. `(grounded_generation)` (15 pts) Generate answers with inline citation markers that refer to chunk IDs. Parse them, and flag any sentence without a citation.
5. `(citation_eval)` (20 pts) Implement citation recall and citation precision following the ALCE definitions, using `NLI` as the entailment judge. Validate the judge on 50 sentence-citation pairs you label by hand and report agreement.

**Metrics:** answer F1 by answer type (compare with the A0 oracle and closed-book numbers), evidence F1, citation recall and precision, latency.

**Written** (10 pts)
* Which retriever won, and on which question types?
* How much of the gap to the oracle is retrieval vs reading?
* Shuffle the order of retrieved chunks before generation. Does position matter for your model?

**Done when**
- [ ] One table: retrievers by recall@k; one table: end-to-end answer F1 and citation metrics with CIs.
- [ ] Your NLI judge agreement is reported, and you state whether you trust it.

**Interview angle:** BM25 vs dense vs hybrid, chunking trade-offs, how to measure citation quality, NLI judges and their failure modes.

**Milestone:** after A3, write the README, publish the repo, and record a short demo. Everything after this makes it stronger, but this is already a complete project.

---

## Assignment 4: Tools and function calling

**Goals:** a tool registry the agent can rely on, with argument validation, structured errors, permissions and parallel execution. (Project idea 4.)

**Problems**

1. `(registry)` (15 pts) Define tools with LangChain's tool interface (`@tool` with Pydantic argument schemas) and wrap them in your own registry that adds metadata: side effect (yes/no), timeout, permission scope. Schema design, descriptions and argument validation are yours.
2. `(tools)` (20 pts) Implement: `search_library`, `get_paper_card`, `arxiv_search`, `citation_graph` (Semantic Scholar API; cache aggressively and request a free API key for higher limits), `python_exec` (subprocess sandbox with timeout, no network, memory limit), `send_digest` (writes to an `outbox/` folder; this is your side-effect tool, gated in A7).
3. `(calling_modes)` (15 pts) Two modes: native tool calling (`bind_tools`, where your model and server support it) and prompt-based calling with JSON parsing that you write. Validate arguments; on failure, return a structured error the model can act on.
4. `(parallel)` (10 pts) Execute independent tool calls concurrently and measure the wall-clock speedup.
5. `(bfcl_eval)` (20 pts) Evaluate on the Berkeley Function Calling Leaderboard data, categories simple, multiple, parallel, parallel-multiple and irrelevance. Implement AST-based matching following BFCL's documented rules.

**Metrics:** accuracy per BFCL category, per model and per calling mode; irrelevance accuracy (correctly not calling a tool); argument error rate.

**Written** (15 pts)
* Native vs prompt-based calling: where did each fail?
* Analyse 20 irrelevance failures. Why does the model call tools it should not?
* What does your sandbox not protect against?

**Done when**
- [ ] BFCL results table across categories, models and modes.
- [ ] A malformed tool call never crashes the system; it becomes an error message to the model.

**Interview angle:** tool schema design, structured errors, sandboxing, why irrelevance detection matters in production.

---

## Assignment 5: The ReAct planner

**Goals:** design the agent loop yourself as a LangGraph graph and understand how and why it fails. (Project idea 3.) LangGraph is allowed from here on.

**Problems**

1. `(loop)` (25 pts) Implement a Thought, Action, Observation loop on top of your tool registry as your own LangGraph `StateGraph` (your nodes, state and edges, not the prebuilt agent). Requirements: a `final_answer` action; a maximum step count; loop detection (repeated tool and arguments, or no new information for several steps); observation truncation to protect context; graceful failure that returns partial findings and says what is missing.
2. `(hotpot_offline)` (20 pts) HotpotQA distractor setting, dev subset. Give the agent two tools over the 10 provided paragraphs: `search` (BM25) and `lookup` (by title). This keeps the benchmark fully offline. Baselines: closed-book chain-of-thought, single-shot retrieve-then-read, and LangGraph's prebuilt ReAct agent with the same tools and step limit.
3. `(domain_set)` (15 pts) Write 30 multi-hop questions over your own library (for example cross-paper comparisons), each with a gold answer and gold paper IDs. This is a test set: never tune on it.
4. `(failure_taxonomy)` (15 pts) Label 50 failures into categories such as bad search query, ignored evidence, premature stop, loop, invalid tool call, formatting error. Fix the top category and re-measure.

**Metrics:** EM and F1, average steps, loop rate, failure rate, tokens per question.

**Written** (15 pts)
* When was ReAct worse than retrieve-then-read, and why?
* Where did your graph beat or lose to the prebuilt agent, and which of your design choices explains it?
* How does the `SMALL` vs `MEDIUM` gap compare to A3?
* What did fixing the top failure category change?

**Done when**
- [ ] Table with your ReAct graph vs all three baselines on HotpotQA and your domain set.
- [ ] Failure taxonomy with counts, before and after your fix.

**Interview angle:** how ReAct works, why agents loop, stopping criteria, when not to use an agent.

---

## Assignment 6: Memory

**Goals:** a memory system that remembers across sessions, updates stale facts, and knows when it does not know. (Project idea 5.)

**Problems**

1. `(short_term)` (10 pts) Conversation window with a rolling summary when the token budget is exceeded.
2. `(long_term)` (30 pts) Use LangGraph's checkpointer and store for persistence; the memory policy is yours. Write path: after each session, extract memory items (content, type, timestamp, source session, what it supersedes) into a Pydantic model; deduplicate; resolve conflicts when a fact changes. Read path: retrieve by similarity, recency and type, with support for time-aware questions.
3. `(longmemeval)` (25 pts) LongMemEval (the shorter "S" setting), a subset of about 100 questions stratified by question type. Compare: (a) full history truncated to the model's context, (b) naive retrieval over raw turns, (c) your memory system. Use an LLM judge following the benchmark's protocol, and validate it on 40 answers you label by hand.
4. `(product)` (10 pts) The copilot remembers your research interests and reading history and can answer questions like "what did I read about retrieval last week?"

**Metrics:** accuracy per question type, evidence session recall, tokens per answer, judge agreement.

**Written** (15 pts)
* Which question types benefited most from structured memory?
* How did your system handle knowledge updates and abstention questions?
* What is your memory's worst failure, and how would you detect it in production?

**Done when**
- [ ] Results table for the three systems, by question type.
- [ ] A test showing an updated fact replaces the old one at read time.

**Interview angle:** short vs long-term memory, write-time vs read-time processing, staleness, evaluating memory.

---

## Assignment 7: Abstention and human-in-the-loop

**Goals:** know when the system should not answer, and make risky actions wait for a human. (Project idea 6.)

**Problems**

1. `(confidence)` (20 pts) Implement at least three confidence signals: verbalized confidence, self-consistency agreement over k samples, retrieval score, and NLI support of the answer by its cited evidence. Combine them with a small classifier trained on dev.
2. `(policy)` (10 pts) A decision policy: answer, abstain ("the paper does not say"), or escalate to a human.
3. `(qasper_abstention)` (20 pts) QASPER includes unanswerable questions, which gives you ground truth. Measure abstention precision, recall and F1, plus a risk-coverage curve and its area (AURC) for each signal and the combination.
4. `(approval_gate)` (25 pts) Side-effect tools require approval (LangGraph interrupts and a persistent checkpointer are the natural fit; which actions need approval and what the reviewer sees are yours). The workflow pauses, persists its state, lets a human approve, edit or reject from the CLI, and resumes, including after a process restart. Every decision goes to an append-only audit log (what, when, decision, reason).
5. `(tests)` (10 pts) Property-style tests that every side-effect call is gated across randomly generated plans; a resume-after-kill test.

**Written** (15 pts)
* Calibration of verbalized confidence for `SMALL` vs `MEDIUM` (expected calibration error and a reliability diagram).
* Which signal was most useful, and why do you think so?
* At what coverage would you deploy this, and what does that cost in human reviews?

**Done when**
- [ ] Risk-coverage plot with all signals.
- [ ] Approval gate passes the kill-and-resume test, and the audit log is complete.

**Interview angle:** calibration, selective prediction, designing approval workflows, audit trails.

---

## Assignment 8: The model router

**Goals:** spend compute only where it helps, and prove it with a fair comparison. (Project idea 7.)

**Problems**

1. `(labels)` (10 pts) Run `SMALL` and `MEDIUM` on train splits from A3, A5 and A9. Label each example by whether the small model was good enough.
2. `(routers)` (30 pts) Plug each router into your graph as the decision function of a conditional edge; the routing logic is yours. R0: always small, always medium. R1: rules (length, question type). R2: a classifier on embeddings plus features. R3: a cascade (small first, escalate when A7 confidence is low).
3. `(eval)` (25 pts) Plot quality against compute (latency, GPU-seconds or tokens) for every router. Include a random router at the same fraction of medium calls, and an oracle router as the upper bound.
4. `(routerbench)` (15 pts) Train and test your R2 approach on RouterBench's precomputed outputs so you can compare with published routing results without running large models.

**Written** (20 pts)
* Pre-routing vs cascading: which won on your tasks and why?
* How close did you get to the oracle router?
* What would change if latency mattered more than throughput?

**Done when**
- [ ] Quality vs compute plot with all routers plus random and oracle baselines.

**Interview angle:** model routing, cascades, why a random baseline at matched rate is necessary.

---

## Assignment 9: Claim verification with multi-agent debate

**Goals:** test whether multiple agents actually help, under a fair compute budget. (Project idea 9.)

**Problems**

1. `(pipeline)` (20 pts) SciFact: given a claim, retrieve abstracts (reuse A3), predict SUPPORT, CONTRADICT or NOT ENOUGH INFO, and select rationale sentences.
2. `(systems)` (25 pts) Three systems with the same base model: single agent; self-consistency with k samples and majority vote; debate with a proponent, a skeptic and a judge over 1 to 3 rounds.
3. `(matched_compute)` (20 pts) Report every system against tokens used. Plot F1 vs tokens.
4. `(product)` (10 pts) A "verify" feature: paste a claim, and the copilot checks it against your library with cited evidence.

**Metrics:** SciFact abstract-level label F1 and sentence-level rationale F1 (implement following the official definitions), tokens per claim.

**Written** (25 pts)
* Did debate beat self-consistency at matched tokens?
* Find cases where debate flipped a correct answer to an incorrect one. What happened?
* Would you ship debate in production? Defend the answer with your numbers.

**Done when**
- [ ] F1 vs tokens plot for all three systems.

**Interview angle:** multi-agent systems, conformity effects, why compute-matched comparisons matter.

---

## Assignment 10: Self-refinement

**Goals:** learn when self-critique helps and when it does not. (Project idea 10.)

**Problems**

1. `(baseline)` (10 pts) SciTLDR: generate one-sentence summaries from the abstract (or from abstract, introduction and conclusion).
2. `(refinement)` (30 pts) Generate, critique, revise for 0 to 3 iterations under two conditions: intrinsic (the model critiques itself) and external feedback (NLI faithfulness against the source, a length check, coverage of key entities from the A2 paper card).
3. `(eval)` (25 pts) ROUGE-1/2/L against the multiple gold summaries, NLI faithfulness, and pairwise preference from `JUDGE`, validated against 30 comparisons you judge yourself.

**Written** (25 pts)
* Plot quality by iteration for both conditions.
* Do your results agree with the finding that intrinsic self-correction is weak without external feedback?
* Where did revision make outputs worse?

**Done when**
- [ ] Iteration curves for both conditions with CIs.

**Interview angle:** Self-Refine and Reflexion, external vs intrinsic feedback, LLM-as-judge validation.

---

## Assignment 11: Integration and observability

**Goals:** turn modules into one system you can operate, monitor and debug. (Project idea 11.) Compose your existing LangGraph graphs as subgraphs of one parent graph.

**Problems**

1. `(graph)` (20 pts) Compose the system: intent detection, router, then the right path (question answering, claim verification, digest, memory). Every module can be switched off by config for ablations.
2. `(tracing)` (10 pts) Spans for every node and tool call (LangSmith traces LangGraph runs automatically; add the metadata and tags you need). A dashboard with p50 and p95 latency, tokens, cache hit rate, tool error rate, abstention rate and escalation rate.
3. `(faults)` (20 pts) A fault injection harness: tool timeouts, malformed tool output, model server down, slow embedding service. Verify graceful degradation, and implement simple threshold alerts.
4. `(e2e_eval)` (20 pts) An end-to-end suite: your A5 domain set, a QASPER sample and a SciFact sample. Run an ablation removing each module in turn.
5. `(ci)` (10 pts) CI runs unit tests and replays evals from cached outputs to catch pipeline regressions without a GPU.
6. `(demo)` (5 pts) A simple UI (Gradio, Streamlit or CLI) and a 3-minute demo video.

**Written** (15 pts)
* Ablation table: which module contributes most end to end?
* Which injected fault was hardest to detect, and how did you fix that?

**Done when**
- [ ] Every injected fault either degrades gracefully or raises an alert.
- [ ] Ablation table with CIs.

**Interview angle:** tracing agents, what to alert on, regression testing for LLM systems, ablations.

---

## Assignment 12: Open source and the write-up

**Goals:** contribute to real-world software and present your work. (Project idea 12.)

**Problems**

1. `(contribution)` (40 pts) Make a contribution to an agent or evaluation project you used or studied, for example a benchmark adapter for QASPER or SciFact in an evaluation framework, a local-model human-in-the-loop example for LangGraph docs, or a fix for a bug you hit. Read the contribution guide, open an issue first, keep the PR small.
2. `(readme)` (30 pts) Final README: architecture diagram, one results table per module with CIs, ablations, failure analysis, limitations, hardware and runtime.
3. `(blog)` (20 pts) A blog post on the most surprising result (debate vs self-consistency and intrinsic vs external refinement are good candidates).
4. `(retro)` (10 pts) `reports/a12.md`: what you would design differently if you started again.

**Done when**
- [ ] PR opened (merged is the goal).
- [ ] README, blog post and demo video published.

---

## Appendix A: Datasets

All are public. Check each dataset's license and loading instructions; some Hugging Face datasets have moved from loading scripts to Parquet files, so loading code may need adjusting.

| Assignment | Dataset | What it gives you |
|---|---|---|
| A0, A3, A7 | QASPER (AllenAI) | Questions over NLP papers, gold answers, evidence paragraphs, unanswerable questions |
| A1 | arXiv API | Live paper stream (recorded fixture for evals) |
| A2 | SciREX (AllenAI) | Gold Task, Dataset, Method, Metric entities for full papers |
| A3 (stretch) | ALCE | Citation evaluation definitions and data |
| A4 | Berkeley Function Calling Leaderboard | Tool calling categories with AST checking rules |
| A5 | HotpotQA (distractor) | Multi-hop QA with provided paragraphs |
| A6 | LongMemEval | Multi-session chat histories with memory questions by type |
| A8 | RouterBench | Precomputed outputs and costs from many models |
| A9 | SciFact (AllenAI) | Scientific claims, abstracts, labels and rationales |
| A10 | SciTLDR (AllenAI) | Paper summaries with multiple gold references |

## Appendix B: Reading list

Read the relevant paper before starting each assignment.

* A0: any good guide to bootstrap confidence intervals for model evaluation.
* A2: SciREX (Jain et al., 2020).
* A3: QASPER (Dasigi et al., 2021); ALCE, "Enabling Large Language Models to Generate Text with Citations" (Gao et al., 2023); "Lost in the Middle" (Liu et al., 2023).
* A4: Gorilla and the Berkeley Function Calling Leaderboard.
* A5: ReAct (Yao et al., 2022); HotpotQA (Yang et al., 2018).
* A6: LongMemEval (Wu et al., 2024).
* A7: work on selective prediction and calibration of language models.
* A8: RouterBench (Hu et al., 2024); RouteLLM (Ong et al., 2024).
* A9: "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (Du et al., 2023); SciFact (Wadden et al., 2020).
* A10: Self-Refine (Madaan et al., 2023); Reflexion (Shinn et al., 2023); "Large Language Models Cannot Self-Correct Reasoning Yet" (Huang et al., 2023); SciTLDR (Cachola et al., 2020).

## Appendix C: Final portfolio checklist

- [ ] Every module has a results table with a baseline and CIs.
- [ ] All numbers regenerate from cache with one command.
- [ ] Ablation table for the integrated system.
- [ ] Failure analyses for A3, A5 and A9.
- [ ] Demo video and live or recorded demo.
- [ ] Open-source PR.
- [ ] You can answer every "Interview angle" question without notes.
