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
    "allowed_relations": ["depends_on"],
    "answer_differs_materially": "unknown"
  }]
}
```

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

1. Graph-off lacks every required support at the specified budget.
2. Graph-on returns every support with a nonempty accepted graph path, hops ≥ 1.
3. Every edge in that support's path has an ID and a type in `allowed_relations`.
4. A human judges that the answer differs materially (`answer_differs_materially: "yes"`).

`related_to` never satisfies a typed requirement such as dependency, causality,
supersession, or remediation. Allow it only for a question that explicitly
requires relatedness. Zero-hop graph/entity matches do not prove traversal.
The runner checks the service's accepted path evidence and the suite's allowed
types; humans must review semantic correctness and source-revision freshness.
The final condition defaults to `unknown`: otherwise successful retrieval is
`needs_judgement`, not `pass`. A missing retrieval/path condition, call failure,
or a human `"no"` is `fail`. Seed/support found flags summarize the union of
repeated rounds; consult per-round paths and stability before claiming a result.

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
fields. They contain source paths and graph IDs: review these identifiers before
sharing. Errors never print server exception strings. Use new output/private
paths per run. Failed calls exit 1, setup failures exit 2; exit 0 says calls
succeeded, not that questions passed.

Reports record per-round paths, scores, graph hops/steps/types/IDs, diagnostics,
found flags, elapsed milliseconds and sanitized-response SHA-256 stability.
Hashes exclude timing, policy, and private fields (not byte hashes of the full
response). Metadata includes endpoint host only, MCP protocol, timestamps,
source-map size, suite and runner hashes. Optional `--binary-sha256`,
`--source-revision`, `--embedding-identity`, `--source-count` and
`--accepted-edge-count` record operator-supplied snapshot identity; these claims
do not freeze the live corpus. `--compare` prints per-question gained/lost paths
for off/on, boolean flag flips, mean per-call elapsed delta and pass transitions,
without note text. The sanitized 2026-11-02 fixture predates tri-state scoring;
its absent edge paths are classified as fail. Its old support paths also
predate the dated-support suite revision, so record that coverage difference
when comparing. Preserve the same corpus, suite and protocol to isolate ranking
changes; timing drift alone is not evidence of graph usefulness.
