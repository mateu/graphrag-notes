# Review proposed connections

`garden review` puts both notes, their excerpts and provenance, and the
proposal's confidence, reason, generator, and lifecycle state in one place.
Once proposals exist, review and decisions work without inference providers.
These commands are in current source and the published Apple Silicon
`0.1.0-rc.2` prerelease; the published
`v0.1.0-rc.1` predates the inbox. The [daily workflow](daily-workflow.md) connects
review to capture and retrieval. Native Linux and Intel macOS release acceptance
remains in [#66](https://github.com/mateu/graphrag-notes/issues/66).

```sh
graphrag garden scan
graphrag garden review
graphrag garden review --interactive
```

The inbox defaults to pending proposals, with up to 20 cards. Increase the
bounded scope with `--limit 100` (maximum 200). Skipping leaves a proposal pending
for a future session. No review position is persisted as a record identity.
For one proposal, copy its printed **Review interactively** command; it retains
the selected configuration and database even when run from another directory.
If a selected path contains non-UTF-8 bytes, replay hints are omitted with a
warning. Cards and interactive decisions remain available; add `--interactive`
to the same invocation to review without a copyable hint. No lossy replacement
path is printed. This does not add raw-byte database-path support to the current
SurrealDB backend.

```sh
graphrag --config /path/to/config.toml --db-path /path/to/database \
  garden review --id proposed_edge:ID --interactive
graphrag garden review --status accepted
graphrag garden review --all-statuses --format json
graphrag garden review --format jsonl
```

An interactive card offers these actions:

| Action | Outcome |
| --- | --- |
| `a` / `accept` | Type `accept` to confirm, then enter an optional audit reason. Creates the existing governed edge and records a manual review. |
| `r` / `reject` | Type `reject` to confirm, then enter an optional audit reason. Records the existing rejection and creates no edge. |
| `s` / `skip` | Advances to the next card without changing the proposal, timestamps, or audit. |
| `v` / `view` | Reads both complete stored notes in the same session. |
| `u` / `undo` | Available on an accepted proposal with a recorded edge ID. Type `undo` to confirm, then enter an optional reason. Removes the edge and supersedes the proposal while retaining its original review audit. |
| `q` / `quit` | Ends the session. EOF also exits; no unconfirmed decision is written. |

Enter at a confirmation prompt cancels that decision. Enter at the reason
prompt records a default such as `explicit interactive review accept`.
The reviewer is `cli interactive review`, and acceptance remains manual.
Each completed decision advances to the next selected proposal. An empty inbox
is a successful read-only result. Prompts and refusals are on stderr.

Both endpoint notes must be visible for acceptance. A missing note or an
unavailable source generation is shown explicitly; rejected and superseded
proposals cannot be accepted from the inbox. After confirmation and reason
entry, the CLI reloads the proposal and both note revisions. Any change cancels
the decision and displays a refreshed card for a new review. The repository
also applies its existing locked endpoint-visibility and lifecycle checks.
The CLI comparison is a pre-action guard; it does not add an atomic revision
transaction for arbitrary future in-process writers.

An `accepting` proposal is a recoverable acceptance claim. Review it with
`--status accepting`, then use the existing explicit command to finish the
claim:

```sh
graphrag garden proposals accept proposed_edge:ID --yes
```

An accepted card includes a copyable undo command with the exact recorded edge
ID and selected configuration/database. Human cards show update and review times,
the reviewer and reason, and whether acceptance was manual. You can also undo using
`garden review --status accepted --interactive`. Undo supersedes the proposal;
neither a repeated scan nor another inbox acceptance resurrects it. The
existing explicit command supports repeatable cleanup:

```sh
graphrag edges undo related_to:EDGE_ID --yes
```

Source IDs, source types/generations, headings/lines, and available chat
identities are shown beside excerpts. Restored backups may redact local source
URIs; those records retain source identity and are labeled URI unavailable.
They are not presented as manual notes. **Inspect full note** commands use
revision guards and work offline. Reviewing does not open an editor or a file.

For automation, JSON returns a versioned `garden.review` envelope whose
`data.proposals` array contains cards. Each card has stable proposal `id`,
status/confidence/reason/generator/audit fields, `from` and `to` endpoints,
`accept_allowed`, `accept_blocked_reason`, replay commands, and card `warnings`.
`review_command`, endpoint `inspect_command`, and `undo_command` can be null
when the selected replay paths are not valid UTF-8; card warnings explain why.
Interactive undo remains available for an accepted proposal with its recorded
edge ID even when no copyable command can be printed. Endpoints expose
`available`, title, a maximum 500-character excerpt (including any truncation
ellipsis), provenance, revision, and warnings. JSONL uses one envelope per card
with the card directly in `data`; an empty inbox produces zero lines. Machine
output remains read-only and rejects `--interactive` with exit 2. Missing
focused proposals exit 3; invalid IDs, limits, or incompatible options exit 2.
Interactive action refusals keep the current card available for another
decision or a read-only skip/quit.

Batch acceptance remains an independent, deliberate policy command:

```sh
graphrag garden proposals accept --all --min-confidence 0.9 --yes \
  --reason "Reviewed the similarity policy for this collection"
```

Both the explicit finite threshold in `[0, 1]` and `--yes` are required. The
existing batch path applies only to Gardener similarity `related_to` proposals;
it cannot bulk-approve logical support or contradiction claims.

## High-precision candidate scan

`garden scan` without options keeps the legacy similarity scan. Add
`--max-candidates N` (1–50) to run the bounded candidate generator instead. The
same operation runs inside the owning service, so it also works while
`graphrag serve` holds the database:

```sh
graphrag garden scan --max-candidates 5 --dry-run
graphrag garden scan --max-candidates 5 --source source:ID --counterpart-source source:OTHER
graphrag --server http://127.0.0.1:3300/mcp garden scan --max-candidates 5 --dry-run --format json
```

A candidate is a cross-source `related_to` pair that passes every gate:

- Notes from the same source, or sharing a chat conversation, are never paired.
  They are excluded before the per-endpoint neighbour limit, so adjacent chunks
  cannot crowd out other sources. Identical content is a duplicate, not a relation.
- Cosine similarity is at least `similarity_floor` (default 0.80).
- A second signal is required: a shared *specific* entity, or an explicit
  cross-reference (one note names the other's record ID, source URI, wikilink
  stem or file name). Dates and number-only names never count; a name mentioned
  by more than `max_entity_note_frequency` notes (default 10) is generic.
- No accepted relation of any type, pending proposal or prior decision already
  joins the pair.

Anchors come from `--note`, `--source`, or orphans by default (notes without an
accepted relation of any type, `derived_from` included), in record-ID order.
Candidates are ranked deterministically: cross-reference, then the number of
shared specific entities, then similarity, then IDs. A note appears in at most
`max_per_endpoint` candidates (default 2). `--max-candidates`, `--max-endpoints`
(default 25) and `--deadline-ms` (default 10000, at most 60000) bound the run.
`--min-similarity` can raise the floor and `--max-per-endpoint` can lower the
cap; requests never loosen the configured policy:

```toml
[gardener.candidates]
similarity_floor = 0.80
neighbors_per_endpoint = 8
max_per_endpoint = 2
max_entity_note_frequency = 10
```

The receipt echoes every resolved limit and policy value, counts why nominated
pairs were rejected, and reports `stop_reason` (`completed`, `endpoint_budget`,
`candidate_budget` or `deadline`) plus a `--after` cursor while anchors remain.
The cursor never advances past candidates cut by `--max-candidates` or the
deadline: repeat the same scan to reach them (pairs already recorded are then
skipped). Pairs held back by the per-endpoint cap are reached by a later scan of
the same scope. The deadline also bounds waiting for the proposal lifecycle
lock; a write that has started completes. Zero candidates is a valid
abstention. The CLI refuses a receipt that does not echo its request.

Without `--dry-run`, each candidate becomes a **pending** proposal through the
normal lifecycle, with generator `gardener-candidate` and its features
(similarity, signals, shared entities with note frequencies, cross-reference,
policy and requester) stored as versioned JSON in `reason`. Existing proposals
are never refreshed and decided ones are never resurrected. The scan never
accepts anything. `garden apply` and `garden proposals accept --all` select only
legacy `gardener-similarity` proposals, so candidates are accepted only through
individual review. Each examined endpoint runs one exact similarity scan over
the notes, like the legacy scan, so keep `--max-endpoints` modest on large corpora.
