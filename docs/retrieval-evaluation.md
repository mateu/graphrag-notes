# Evaluating useful search results

`scripts/evaluate-retrieval.py` compares keyword/off and hybrid/off, auto, and
on against the same private graded cases. It calls the existing remote CLI for
search and revision-pinned inspection; it never captures, uploads, edits, or
opens an embedded database. Hybrid still uses the configured query embedding
provider. An unavailable policy is recorded as failed without changing mode.

Use an isolated service restored from a verified portable backup. Record the
archive/payload hashes, source revision and binary hash, config hash, embedding
identity, source counts, and entity/accepted-edge coverage in suite metadata.
Keep that snapshot unchanged throughout comparisons. A supplied metadata claim
does not freeze an arbitrary live endpoint; preparing and preserving the
snapshot is the operator's responsibility.

Keep cases, judgments, raw rankings, and readbacks in a private directory with
mode `0700`. Reports are created exclusively with mode `0600`; existing reports
are refused. Credentials come only from the named environment variable and are
never passed as argument values. For example:

```sh
mkdir -p /tmp/graphrag-private-eval
chmod 700 /tmp/graphrag-private-eval
python3 scripts/evaluate-retrieval.py \
  --suite /tmp/graphrag-private-eval/cases.json \
  --binary /path/to/graphrag \
  --endpoint http://127.0.0.1:3000/mcp \
  --credential-env GRAPHRAG_TOKEN \
  --output /tmp/graphrag-private-eval/report-001.json \
  --summary /tmp/graphrag-private-eval/aggregate-001.json
```

Supply the credential through your existing private environment setup. Use a
read-only principal. The CLI validates HTTPS or an encrypted-tunnel loopback
endpoint. Graph policies and result limits are explicit rather than inherited
from the daily client's defaults.

## Cases and judgments

Create 30–60 actual questions before changing ranking. Include exact titles,
direct questions, paraphrases, ambiguous entities, relationships, source/time
filters, and absent answers. Freeze the calibration/holdout split by topic before
tuning; holdout questions should not merely paraphrase calibration questions.
Review expected answers against actual source text and record reviewer/method
and limits in metadata. Delegated agent review must be labeled as such, never
presented as an observed human judgment or user study.

The suite wraps the existing version-two eval query/relevance fields:

```json
{
  "schema_version": 1,
  "metadata": {
    "snapshot_manifest_sha256": "record-the-real-archive-hash",
    "build_revision": "record-the-exact-source-revision",
    "judgment_review": {"method": "owner", "reviewed_at": "record-the-review-time"}
  },
  "cases": [{
    "name": "fictional-atlas",
    "split": "holdout",
    "category": "paraphrase",
    "answerability": "answerable",
    "judgment": "This fictional note directly states the launch time.",
    "eval": {
      "schema_version": 2,
      "query": "When does Atlas launch?",
      "scope": "notes",
      "limit": 5,
      "k": 5,
      "relevance": [{"id": "note:fictional", "grade": 3}]
    }
  }]
}
```

Categories are `title`, `direct`, `paraphrase`, `entity`, `relationship`,
`filters`, and `negative`. Splits are `calibration`/`holdout`. Answerability is
`answerable`, explicitly reviewed `unanswerable`, or `unjudged`. A no-answer case
must have no positive judgments; an unreviewed question is not automatically a
no-answer case. `since_days` and `source_uri` use the normal search contracts.
An optional `formulation` labels `question`, `terms`, or `title` (default
`question`). Include ordinary short queries as well as full questions, and
report their counts and separate scored denominators so the workload does not
accidentally favor one channel.

Use grade 0 for reviewed irrelevant records, 1 for useful context, 2 for a
partial answer, and 3 for a direct answer. As in eval v2, `expected_ids` is
ungraded positive relevance. Unlisted records remain unjudged. Record whether
judgments cover a candidate pool or the whole snapshot; several chunks repeating
a fact can legitimately be relevant, so known-positive recall is not necessarily
recall of all relevant documents.

Judgment IDs and scored result IDs are trimmed and Unicode-lowercased like the
existing eval-v2 implementation; inspection still requires the exact server ID
and revision. Explicit conflicting grades for one normalized ID are rejected.

## Reading reports

The private report retains each requested policy/case, ranks, guarded readback
hashes, and failures. Search snippets need not equal full inspected content;
canonical ID, revision, title, and provenance must match. Duplicate IDs, missing
provenance/revisions, and stale/foreign readbacks fail that comparison.
Repeated full inspected content under different IDs is counted separately;
such repeated chunks are not confused with duplicate canonical IDs.

- Useful-result lower/upper bounds identify uncertainty from unjudged hits.
- Recall counts judged positive IDs only. Reciprocal rank of judged positives
  is a lower bound while unjudged hits remain; it is not exact MRR of all useful
  notes. Precision and nDCG are withheld when top-k contains unjudged hits.
- Reviewed no-answer cases report empty-result success and false-positive
  counts across all returned records (including results beyond k); their
  recall/nDCG are undefined rather than artificially perfect. Precision,
  usefulness and ranking metrics still use top-k.
- Aggregate metrics include scored-case counts and failures, with separate
  calibration and holdout groups. Compare those denominators, not just means.
- Single search timings are diagnostic. They exclude guarded inspection and
  do not establish latency percentiles, direct gateway timing, or model time.

The separately requested summary exports a fixed allowlist: policy/split counts,
metric means/denominators, uncertainty, failures, and the suite hash. It omits
queries, case labels, IDs, bodies, URIs, credentials, and arbitrary metadata.
The [initial private-corpus baseline](validation/retrieval-92-baseline.md)
illustrates the strengths and limits of these measures.
Review this sanitized artifact before adding it to public evidence. Never publish
the private report or case file.

For deterministic offline tests, `--recorded` accepts a private JSON object with
`rankings[case_name][policy]` and `inspections[canonical_id]`. It excludes live
binary/endpoint arguments. Missing ranking/readback entries become failed rows;
one unavailable policy does not erase the other comparisons. Exit 0 means all
retrieval/readback calls passed, not that relevance was perfect. Exit 1 records
failed comparisons; exit 2 indicates input/connection/output setup failure.

## Graph context evaluation

`scripts/evaluate-graph-context.py` uses the repo's standard-library MCP client
and calls only `build_context`. Use a read-only principal and an HTTP loopback
endpoint (or an encrypted SSH tunnel ending on loopback) with path `/mcp`.
Credentials are read only from `--credential-env` (default `GRAPHRAG_TOKEN`);
never supply their value in arguments. It works with Python 3.9+.

Keep the suite and source map private. A suite has this shape:

```json
{
  "questions": [{
    "id": "Q1",
    "question": "What prerequisite explains the fictional rollout order?",
    "seed": ["memory/fictional-seed.md"],
    "support": ["memory/fictional-support.md"],
    "support_chunks": {"memory/fictional-support.md": ["note:fictional-answer-chunk"]},
    "allowed_relations": ["depends_on"],
    "allowed_directions": ["outbound", "inbound"],
    "answer_differs_materially": "unknown"
  }]
}
```

`support_chunks` names, per support path, the opaque record ID of each
answer-bearing chunk (the `id` that `build_context` returns for it). A daily
file can hold many unrelated chunks, so a file-level match is not support.
Every listed chunk must be delivered. Keys must be support paths and lists
must be non-empty and unique. A question without chunk IDs for every support
path still runs but cannot pass (`support_chunk_unspecified`).
`allowed_directions` optionally narrows the traversal labels a path step may
use (`outbound`, `inbound`; default both).

`build_context` renders one clipped span per chunk, not the chunk. With the
default estimated token counter (one token per UTF-8 byte) a 400-token chunk
budget renders at most about 400 bytes, so a returned chunk ID does not show
that its answer passage was rendered. Optional `support_evidence` names, per
support chunk ID, what the rendered span must contain:

```json
"support_evidence": {
  "note:fictional-answer-chunk": {
    "required_terms": ["fictional-term-a", "fictional-term-b"],
    "answer_span": [120, 310]
  }
}
```

| Field | Check | Trade-off |
| --- | --- | --- |
| `required_terms` | Every term is a substring of that chunk's rendered `snippet`, after NFC normalization and case folding. | Survives re-chunking and small edits. The terms are private text and stay in the suite. |
| `answer_span` | The byte range `[start, end)` of the chunk content lies inside `selected_span_start..selected_span_end`. An untruncated chunk covers any range. | Holds no text. Goes stale when the chunk content changes, so pin it to a chunk revision. |

Give either field or both; with both, both must hold. List only the terms
without which the answer is not there. Keys must be chunk IDs listed in
`support_chunks`; terms must be non-empty and unique; a span is two integers
with `0 <= start < end`. The check is exact matching only: it can show that an
answer is absent, never that a context answers the question.

The source map accepts `{ "upload-hash": "workspace-relative/path.md" }` or
the importer's registry `files[path].parts[*].id = "source:<hash>"` (parts can
be an object or list). Hashes map the tail of `mcp://upload/<hash>` to paths.
Unmapped results are labeled `unmapped`; inspect coverage before interpreting
misses. Neither the registry nor its importer credential should be published.

The fixed protocol is `off,on,on,off`, scope `notes`, two chunks, 1000 total
tokens, 400 tokens per chunk. `--policies`, `--scope`, `--max-chunks`,
`--max-total-tokens`, and `--max-chunk-tokens` override it and are recorded.
Policies must include at least two rounds of each off/on. Optional
`--max-chunks-per-source` is forwarded unchanged; leave it unset on older
servers that do not support it. It does not emulate diversity client-side.

A pass requires all four suite conditions:

1. Graph-off lacks every required support, by file and by expected chunk ID,
   in every off round at the specified budget.
2. **Every** graph-on round independently returns every expected support chunk
   (by ID, from its support file) through a contiguous accepted path. The path
   has `hops ≥ 1` steps (`hops` equals the step count). It starts at the
   graph's `seed_note_id`, each step starts at the previous step's `to_id`, no
   node repeats, and the last `to_id` is that chunk's ID.
3. Every step has an edge ID, a type in `allowed_relations` whose edge ID
   belongs to that relation table (`<type>:<key>`), and a permitted direction.
4. A human judges that the answer differs materially (`answer_differs_materially: "yes"`).

`related_to` never satisfies a typed requirement such as dependency, causality,
supersession, or remediation. Allow it only for a question that explicitly
requires relatedness. Zero-hop graph/entity matches do not prove traversal.
The runner checks the service's path evidence: its shape, continuity, terminal
and the suite's allowed types and directions. It does not check that the graph
seed is the suite's seed file. Humans must review semantic correctness and
source-revision freshness.
The final condition defaults to `unknown`: otherwise successful retrieval is
`needs_judgement`, not `pass`. A missing retrieval/path condition, call failure,
missing field, or a human `"no"` is `fail`. Each summary lists sorted
`fail_reasons`, which the `--summary` line also prints:

- consistency: `inconsistent_support` (some successful on rounds delivered the
  support and others did not; `on_support_by_round` shows which);
- semantic/path, per on round: `support_missing` (neither the chunk nor its
  file returned), `unrelated_chunk` (only other chunks of the support file),
  `invalid_path` (the chunk without a valid path as above),
  `support_source_mismatch` (the chunk ID mapped to another file);
- rendered span, per on round: `support_span_missing` (the chunk was delivered
  as above, but its rendered span lacks a required term or does not cover the
  answer span);
- baseline: `off_has_support`;
- data/protocol: `call_error`, `insufficient_rounds` (fewer than two off or two
  on rounds), `support_chunk_unspecified`, `missing_fields` (no chunk IDs or
  misaligned per-chunk fields);
- judgement: `human_rejected`.

Seed/support found flags still summarize the union of repeated rounds at file
level; only the pass conditions are per round and per chunk.

The span check runs only for a chunk that is already `delivered`, so the
path reasons keep priority. One failing on round fails the item. A covered span
changes nothing else: the item stays `needs_judgement` until a human decides,
and the check never produces `pass`. `span_check` in each summary is
`unchecked` (the suite has no evidence for these chunks; verdicts are exactly
as before), `not_delivered` (evidence, but no delivered chunk to check),
`covered` or `missing`. `on_span_by_round` gives checked and covered chunk
counts and found and required term counts per on round, so a partial delivery
is visible. `span_unchecked_chunks` counts support chunks without evidence.
Delivered evidence chunks whose snippet or span offsets are absent fail as
`missing_fields`.

`off_answer_terms_found` is a flag, not a fail reason. It is `true` when every
required term already appears somewhere in the rendered graph-off context in
every off round, which suggests graph-off already answers from another chunk.
Terms spread over unrelated chunks also set it, and an answer worded
differently does not, so it prompts a look at the private context and decides
nothing. It is `null` without `required_terms`.

Evidence terms, snippets and questions never reach the report, `--json`
output, summary lines or error messages. Those carry statuses, counts and the
numeric span offsets the service returned. No hashes of terms are written,
because a short term can be guessed from its hash.

```sh
python3 -I scripts/evaluate-graph-context.py \
  --suite /private/suite.json --source-map /private/state.json \
  --endpoint http://127.0.0.1:33157/mcp --credential-env GRAPHRAG_TOKEN \
  --output /private/graph-run-002.json --private-dir /private/graph-snippets-002 \
  --compare /private/graph-run-001.json --summary
```

Reports are exclusively created `0600` and refuse overwrite. Raw MCP responses
(including snippets/rendered context) are retained **only** when `--private-dir`
is supplied; that directory must be owned by the current user, mode `0700`, and
not a symlink. Otherwise content is discarded. The public report and `--json`
stdout omit questions, note content, query entities and arbitrary text diagnostic
fields. They contain source paths, opaque chunk record IDs and graph IDs: review
these identifiers before sharing. Errors never print server exception strings.
Use new output/private paths per run. Failed calls exit 1, setup failures exit
2; exit 0 says calls succeeded, not that questions passed.

Reports (`schema_version` 3) record per-round paths, chunk record IDs
(`returned_ids`, aligned with `returned_paths`, `chunk_graph` and
`returned_spans`), scores, graph hops/steps/types/IDs, diagnostics, found
flags, elapsed milliseconds and sanitized-response SHA-256 stability.
`returned_spans` holds each chunk's `start`, `end` and `truncated`;
`span_evidence` holds the per-chunk check status and term counts. Each question
row also records its `support_chunks`, effective `allowed_directions` and, per
evidence chunk, the number of required terms and the answer span.
Hashes exclude timing, policy, and private fields (not byte hashes of the full
response). Metadata includes endpoint host only, MCP protocol, timestamps,
source-map size, suite and runner hashes. Optional `--binary-sha256`,
`--source-revision`, `--embedding-identity`, `--source-count` and
`--accepted-edge-count` record operator-supplied snapshot identity; these claims
do not freeze the live corpus. `--compare` prints per-question gained/lost paths
for off/on, boolean flag flips, mean per-call elapsed delta and pass transitions,
without note text, and the `span_check` transition. It accepts schema 1 and 2
baselines and reads their span check as `unchecked`. Schema 1 rounds have no chunk
IDs, so they cannot be re-graded as passes; the committed 2026-11-02 and
2026-11-03 fixtures stay unchanged and remain 0/8 under the current rules.
The sanitized 2026-11-02 fixture predates tri-state scoring;
its absent edge paths are classified as fail. Its old support paths also
predate the dated-support suite revision, so record that coverage difference
when comparing. Preserve the same corpus, suite and protocol to isolate ranking
changes; timing drift alone is not evidence of graph usefulness.


### Recording a judgement

`needs_judgement` is not a result. To close the fourth pass condition for an
item:

1. Build a private sheet from the run's `--private-dir` responses. For each
   item show the question, the expected answer and the graph-off and graph-on
   contexts as rendered. Judge the rendered text, not the chunk ID: a delivered
   chunk is a truncated span and may omit its answer passage. `support_evidence`
   removes the items whose span misses the answer before anyone reads them.
2. Note what graph-on replaced. At a fixed chunk budget it always displaces a
   graph-off chunk, and the answer can depend on the chunk that left.
3. The repository owner decides `yes`, `no` or `unsure` per item. A delegated
   recommendation may be attached but is not the judgement. `no` covers both a
   delivered chunk that does not answer and one that only repeats what
   graph-off already returned.
4. Append the decision to the private decision log, and publish counts and
   classifications only. Grade guard items mechanically in the same record.

[The `98fb72f` judgement](validation/graph-answer-judgement-251.md) is the
worked example. Setting `answer_differs_materially` in a suite changes its
hash, so record decisions against a retained run instead of editing an
approved suite, unless a rerun is intended.

### Suite revisions

Revise a question or its expected seed only after reading the admitted source and
recording the exact supporting passage and reason (including why a retired seed
was not answer evidence). Keep the original suite rows and dated fixtures for
lineage; write a new dated suite and baseline rather than overwriting them. Diff
old versus new paths and explain each changed question, coverage target, and any
concurrent corpus/accepted-edge change separately from retrieval improvements.
Record whether the new seed is already found graph-off (only support/edge lift
remains to test) or is a budgeted retrieval miss; do not infer a component-level
miss class without a trace. A narrowed question is not evidence for its retired,
broader claim.
