# Search latency operating evidence (#95)

The [sanitized measurements](search-latency-95.json) retain sample counts,
failures, per-category distributions, same-request backend phases and paired
rank checks. Private queries, IDs, returned text, credentials and raw traces
remain outside the repository. Use the [repeatable runner](../search-benchmark.md)
to reproduce a run on a chosen private snapshot.

## Optimized backend baseline

The frozen schema18 snapshot contains 9,751 notes, 365 entities, 458 mentions
and zero accepted inter-note edges. The optimized build contains only the #95
instrumentation relative to the retrieval baseline; these measurements precede
#93's entity and graph changes. Hardware is Apple Silicon macOS, 18 logical CPUs
and 48 GiB memory, with parallel epic builds/checks. An isolated Ollama listener
uses the existing bge-m3 model (1,024 dimensions), without unloading production
models. Four cases cover an exact title, entity terms, source/time filters and a
broad query. The broad case is a performance/provenance check with unjudged
answerability; it is not counted as evidence of useful-answer quality.

All **1,352 backend samples** passed search and independent revision-pinned
readback: eight initial/unloaded-model observations, 672 cached traced samples
at one/four lanes, 336 uncached traced samples and 336 cached untraced samples.
Each policy/load has twenty warm repetitions per case (80 pooled samples).
Lanes include guarded readbacks between searches; this is bounded client load,
not a claim of four searches continuously running simultaneously.

| Warm policy | One lane p50 / p95 ms | Four lanes p50 / p95 ms | Target p95 ms |
| --- | ---: | ---: | ---: |
| keyword/off | 3.6 / 286.8 | 4.3 / 333.5 | 250 |
| hybrid/off | 510.1 / 817.1 | 605.8 / 972.1 | 1,000 |
| hybrid/auto | 506.0 / 816.1 | 615.5 / 982.0 | 1,000 |
| hybrid/on | 509.2 / 833.7 | 626.4 / 992.7 | 1,000 |

Pooled p95 must not hide category misses. Narrow keyword title/entity/filter
p95 is 2.9–6.3 ms, but the broad keyword case is 303.4 ms at one lane and
337.8 ms at four. Broad graph p95 at four lanes is 1,004.9 ms (auto) and
1,013.4 ms (on), slightly above target. Keep the stated budgets as operating
goals and document these corpus/load limits rather than relaxing the targets
based on this one workload.

The first hybrid request was made after confirming the isolated provider had
no loaded model. It took 1,393.2 ms; same-RPC query embedding was 808.4 ms,
vector retrieval 577.4 ms and full text 3.5 ms. This captures model-load-inclusive
embedding time, not a separately measured model load duration or cold OS cache.
The known-title hybrid warm request took 488.3 ms.

Warm cached embedding phases were 0.1–0.4 ms. Explicit --no-cache runs observed
embedding p50 17.7–19.6 ms and p95 21.2–21.4 ms. Exact vector retrieval remains
the dominant phase: cached one-lane hybrid vector p50 475.4 ms/p95 535.6 ms,
compared with full-text p50 1.9 ms/p95 290.2 ms (the broad-query tail).
Those phases share the actual RPC hash. Application and phase timers can nest;
do not sum them or infer transport from unrelated requests.

Cached untraced one-lane p95 was 257.0 ms keyword, 716.7 ms hybrid/off,
714.3 ms graph/auto and 729.8 ms graph/on. Background load changed between
runs, so these differences do not isolate tracing overhead or establish that
disabling the embedding cache improves total latency. Retain both runs and
repeat on an otherwise quiet host before making a causal claim.

Paired graph-minus-hybrid p95 deltas were 36.6–65.7 ms across the two graph
policies and loads, within the 250 ms target. Each group has 80 pairs, zero
failed pairs and zero judged-positive reciprocal-rank regressions in its 60
answerable comparisons. This narrow timing workload does not replace #92's
larger relevance suite or establish graph quality improvement.

## Final graph candidate after the typed pilot

The optimized #93 candidate (compiled inputs `3dc616a09ce3`) was measured
separately after reprocessing eight selected notes. The same frozen corpus now
has 482 entities and 582 mentions; note bodies, vectors, source generations and
all 458 historical mentions remain intact. **672 further samples passed with
zero failures**, with twenty warmed observations per category/policy/load.
This run has RPC timing only, without the #95 same-RPC phase instrumentation.
Concurrent refresh/build activity continued on the same host.

| Warm policy | One lane p50 / p95 ms | Four lanes p50 / p95 ms |
| --- | ---: | ---: |
| keyword/off | 3.1 / 255.7 | 4.3 / 352.5 |
| hybrid/off | 462.4 / 717.4 | 582.3 / 911.6 |
| hybrid/auto | 623.3 / 1,067.0 | 860.9 / 1,365.3 |
| hybrid/on | 633.0 / 1,108.3 | 860.5 / 1,363.7 |

Hybrid/off meets the pooled one-second target; graph exceeds it. Paired
RPC graph-minus-hybrid p95 is **356.5–361.7 ms** at one lane and
**473.6–485.9 ms** at four, above the 250 ms target. Each group has 80 pairs,
zero failed pairs and no regression in 60 judged-positive RR comparisons.
The broad graph category reaches 1,198.7 ms p95 at one lane and 1,417.5 ms
at four; narrow graph categories stay below one second. Broad keyword still
misses its 250 ms target. These are observed operating limits, not relaxed
budgets.

The [full relevance comparison](graph-relevance-93.json) shows the graph
candidate removes four old calibration ranking regressions and matches
hybrid/off's judged RR/recall in this limited suite. The typed pilot adds
validated evidence without further ranking gains. Larger fictional dense
fixtures demonstrate bounded candidate selection, but the new query-relevant
seed preparation has a practical cost. Profile and reduce that work before
expanding extraction. Differences from the earlier build/run do not isolate
backend phases or establish a causal speedup/slowdown under changing load.

## Normal clawd OpenClaw interface

A fresh dedicated webchat session made **86 event-verified direct commands**:
two command-contract checks plus first/20 warmed observations for four routes.
Every search returned the known note at rank one. Delivery was disabled and
there was no positive model-token usage in the fresh acceptance history.
This used the separately deployed live corpus/service, not the frozen backend
snapshot. Explicit flags selected each route; the installed default hybrid/on
and all saved settings were left unchanged.

| Warm route (20 observations each) | UI reply p50 / p95 ms | Handler total p50 / p95 ms | Handler search p50 / p95 ms |
| --- | ---: | ---: | ---: |
| keyword/off title | 124.1 / 531.6 | 37.5 / 120 | 17.5 / 66 |
| hybrid/off title | 723.1 / 1,205.8 | 504 / 580 | 476.5 / 548 |
| hybrid/on title | 628.8 / 800.2 | 508.5 / 643 | 484 / 559 |
| hybrid/on body terms | 617.5 / 1,007.0 | 511 / 601 | 476 / 589 |

The handler reports connect and search on each actual command; private event
proof retains those observations. Handler connection p95 was 52–94 ms.
Connection includes MCP/SSH transport and SDK setup, rather than pure wire RTT.
UI timing spans chat.send to the matching final event, adding gateway dispatch.
These nested observations are separate from same-RPC backend tracing above;
no backend decomposition is claimed for the deployed gateway run.

A separate isolated clawd gateway exercised actual OpenAI OAuth native MCP
orchestration with gpt-6.1-sol/openai-chatgpt-responses. All **21 observations**
passed paired search_notes/get_record execution and exact revision/content/
provenance checks: one first-in-series observation and twenty repeated warm
observations, each with a fresh empty session. Warm chat.send-to-terminal-event
p50 was **12,603.4 ms** and p95 **18,554.9 ms**, with zero failures. The first
observation was 11,896.0 ms; the route had already been exercised during capture
recovery, so this is not a cold model measurement.

This used a fictional one-note corpus and deterministic inference doubles,
rather than the live or frozen real corpus. Gateway startup and session creation
are excluded; tool/backend durations are not inferred from model wall time.
These observations establish the separate orchestration cost of this tested
route, not an absolute latency comparison across the different corpora. Model
planning, tool selection and answer generation must not be labelled backend
search latency. Normal daily configuration was preserved.

## Prioritized next work

1. Profile the final graph seed preparation and reuse bounded retrieval evidence
   where possible. Preserve high-degree useful-seed coverage, scoped current
   aliases, visibility, accepted-only traversal and #92 judgments. Verify
   actual same-request phases before attributing its RPC overhead.
2. Improve exact vector retrieval using measured storage/scan/allocation costs,
   preserving exact cosine results, filtering, ordering and revision/provenance.
   Use paired #92 judgments and #95 traces; do not introduce ANN or increase
   concurrency without evidence.
3. Investigate broad full-text scoring/candidate hydration, which accounts for
   the keyword tail and adds roughly 300 ms to broad hybrid searches. Preserve
   the candidate bounds and stable ranking contract.
4. Profile normal gateway dispatch and connection reuse with event/handler
   observations; UI p95 can exceed one second even when search is under it.
5. Keep a small, explicit model warmup option and honest first-use guidance;
   do not force it at service startup or change model defaults based on this run.

Hardware-dependent thresholds remain opt-in. Normal CI validates protocol,
private reporting, correlation, bounded lanes and metric/error accounting.
