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
