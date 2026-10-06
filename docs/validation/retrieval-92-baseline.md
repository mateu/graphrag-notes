# Private-corpus retrieval baseline (#92)

This baseline freezes 50 source-grounded cases before changing ranking: 34
calibration and 16 held-out cases, split by topic. It includes 34 full questions,
14 ordinary short queries, two exact titles and four no-answer cases (the four
negatives are included among the full questions). The questions and judgments
remain private. Curation and independent review were delegated to Codex under
owner authorization; this is not an observed human relevance study.

The [sanitized report](retrieval-92-baseline.json) records aggregate results,
counts and source/snapshot hashes. All **200 policy comparisons** passed search
and independent revision-pinned readback. Four policies used identical limits
(top five), filters, judgments and a restored snapshot of **9,751 notes**.
The installed release binary was compiled from `31b56957`; repository revision
`e71748b` differs only in documentation. The embedding provider was Ollama,
`bge-m3:latest`, 1,024 dimensions; exact private configuration is retained.

## Useful-answer evidence

Means below count the 32 calibration and 14 held-out answerable cases. They are
**reciprocal rank of judged positives**, a lower bound while other hits remain
unjudged; they are not exact MRR of every useful note in the corpus.

| Policy | Calibration judged-positive RR | Holdout judged-positive RR | Calibration known-positive recall | Holdout known-positive recall |
| --- | ---: | ---: | ---: | ---: |
| keyword/off | 0.245 | 0.429 | 0.238 | 0.429 |
| hybrid/off | 0.714 | 0.857 | 0.678 | 0.857 |
| hybrid/auto | 0.685 | 0.857 | 0.610 | 0.857 |
| hybrid/on | 0.685 | 0.857 | 0.610 | 0.857 |

Query wording matters. Keyword found the two exact titles at rank one and had
judged-positive RR 0.783 across ten calibration short queries and 1.000 across
four held-out short queries. Full natural-language questions had keyword RR
0.000 in this sample. Hybrid/off handled questions better (0.674 calibration,
0.750 holdout). This supports short terms for provider-free keyword use and
hybrid for paraphrased questions; it does not establish population-wide quality.

Graph auto/on had identical behavior in this snapshot. Relative to hybrid/off,
each changed 32 of 50 rankings, improved no judged-positive reciprocal ranks,
and worsened four. The snapshot has only 365 entities, 458 mentions and **zero
accepted inter-note traversal edges**. This evaluates sparse entity seeds,
not a mature relationship graph. Keep graph opt-in; #93 should improve bounded
query-relevant seeds and measure a small explicit extraction pilot before
broader rollout.

## Misses, uncertainty and next work

Keyword returned no hits for all four deliberately absent-answer questions;
hybrid and graph returned nearest records for all four: each produced 20
top-five false positives across those four cases. A nonempty retrieval
result is not proof that a question is answerable: inspect the cited text.
No implicit mode fallback occurred. Repeated full text under distinct canonical
IDs is counted separately in the report; canonical duplicate IDs are rejected.

Judgments enumerate known positives rather than exhaustively labeling every
record. Hybrid top-five pools contain many unjudged hits, so positive-case
precision and nDCG are withheld. Empty answerable keyword results are misses.
No-answer cases have undefined recall/nDCG and explicit returned false-positive
counts. Compare scored denominators in the JSON, not just means across policies.
Three independent source-grounding corrections produced this final baseline;
provisional runs remain private and were not replaced without explanation.

The next quality work is bounded evidence-weighted graph seeds, validated entity
types/aliases, and a revised small pilot (#93). Latency evidence belongs in #95:
these one-off search timings do not establish warm/cold/concurrent percentiles
or conversational model latency. Repeat this unchanged suite against a candidate
using the [evaluation procedure](../retrieval-evaluation.md); retain both raw
reports and publish only reviewed aggregates. Keep held-out cases out of tuning.
