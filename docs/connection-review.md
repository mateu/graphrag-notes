# Review proposed connections

`garden review` puts both notes, their excerpts and provenance, and the
proposal's evidence, reason, generator, and lifecycle state in one place. A
typed proposal shows its evidence mode, verbatim spans, lineage, link,
candidate score, alternative explanation and scope; a legacy proposal reads as
`unassessed` and its stored weight is labelled, never shown as a percentage.
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
status/confidence/reason/generator/audit fields, a report-only `evidence`
summary plus the full `evidence_assessment` when typed, `from` and `to` endpoints,
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

- Notes from the same source, or from one chat conversation (directly or
  through any of its messages), are never paired.
  They are excluded before the per-endpoint neighbour limit, so adjacent chunks
  and conversation siblings cannot crowd out other origins. Identical content is
  a duplicate, not a relation.
- Cosine similarity is at least `similarity_floor` (default 0.80).
- A second signal is required: a shared *specific* entity, or an explicit
  cross-reference (one note names the other's record ID, source URI, wikilink
  stem or file name). A reference written with a path (a qualified wikilink
  such as `[[project-a/design]]`, a file URI, or a relative link such as
  `[design](project-a/design.md)`) must match the source's trailing
  directories; only an unqualified name falls back to the file name or stem.
  `.` segments are ignored, and a `..` path cannot be verified, so it never
  counts. A source URI counts only when the complete URI is named.
  Dates and number-only names never count; a name mentioned
  by more than `max_entity_note_frequency` notes (default 10) is generic.
- No accepted relation of any type, pending proposal or prior decision already
  joins the pair. Such notes never take a place in the neighbour budget (they
  are read separately and only reported), so recorded pairs never hide the
  next-best neighbours from later scans.

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
Without `--dry-run`, the cursor never advances past candidates cut by
`--max-candidates` or the deadline: repeat the same scan to reach them (pairs
already recorded are then skipped). A dry run records nothing, so a repeat
would return the same candidates forever: when it cuts candidates at
`--max-candidates` it reports how many (`dry_run_cut`) and returns **no**
cursor. Narrow the scope, raise `--max-candidates` (at most 50), or run without
`--dry-run` so the recorded pending proposals let later scans progress. A dry
run that cuts nothing advances the cursor normally. Pairs held back by the per-endpoint cap are reached by a later scan of
the same scope. Each pair is re-checked under the proposal lifecycle lock, in
the same critical section as its insert, so a relation or proposal recorded
concurrently is skipped and counted, never refreshed. The deadline also bounds
waiting for that lock; a write that has started completes. Zero candidates is a valid
abstention. The CLI refuses a receipt that does not echo its request.

Without `--dry-run`, each candidate becomes a **pending** proposal through the
normal lifecycle, with generator `gardener-candidate` and its features
(similarity, signals, shared entities with note frequencies, cross-reference,
policy and requester) stored as versioned JSON in `reason`. Candidate proposals
carry no evidence assessment: review surfaces report them as unassessed and read
the stored weight as candidate similarity (a generator score, not a
probability). Existing proposals
are never refreshed and decided ones are never resurrected. The scan never
accepts anything. `garden apply` and `garden proposals accept --all` select only
legacy `gardener-similarity` proposals, so candidates are accepted only through
individual review. Each examined endpoint runs one exact similarity scan over
the notes, like the legacy scan, so keep `--max-endpoints` modest on large corpora
unless similarity batching (below) is on.

### Opt-in quality gates (#230)

The first reviewed live batch kept 3 of 30 candidates. The rejects were
overlapping chunks or copies, serial logs from one recurring lineage, and
machine output. Every kept pair rested on an explicit textual pointer. These
options target those failure modes. **All are off by default**: with no new
keys, scans, receipts and persisted features are exactly as above.

```toml
[gardener.candidates]
near_duplicate_ceiling = 0.95   # cosine at or above: near-duplicate, never a candidate
max_shared_span = 0.5           # longest shared run (>= 12 words) covering this share of the shorter note
max_machine_line_share = 0.6    # notes this share JSON/YAML/table/stack-frame lines neither anchor nor pair
max_lineage_documents = 2       # lineage-aware specificity (see below)
min_specific_entities = 2       # without a cross-reference or cue, need this many specific entities
textual_cues = true             # explicit textual cues are a second signal
similarity_batch_anchors = 64   # anchors per exact similarity pass (1 = one query per anchor)
```

- **Near-duplicates.** Pairs above the ceiling, or sharing a long verbatim
  span, are counted as `near_duplicate` and go to duplicate handling, not
  review. Similarity above about 0.94 predicted duplicates and templates, not
  relationships, so the floor is not raised.
- **Lineage-aware specificity.** A note's *origin document* is the URI its
  uploader declared for the source (one file split into several uploaded parts
  shares it), else the source URI. Two notes of one origin document are never a
  pair (`same_document`). A document's *family* is its origin URI with digit
  runs collapsed, so dated or numbered runs of one log, heartbeat or job
  (`memory/2026-07-01.md`, `memory/2026-07-02.md`) are one family. A shared
  name that is specific corpus-wide but mentioned by more than
  `max_lineage_documents` documents of the pair's families is templating and
  is counted as `lineage_template`. With `min_specific_entities`, a pair
  without an explicit pointer needs that many lineage-specific entities
  (`insufficient_specific_entities` otherwise).
- **Machine dumps.** A note whose non-blank lines are at least
  `max_machine_line_share` JSON/YAML structure, table rows, stack frames or
  addresses, or mostly symbols and digits, is skipped as an anchor
  (`machine_dump_anchors`, still counted as examined) and rejected as a
  counterpart (`machine_dump`).
- **Textual cues** (signal `textual_cue`, ranked with cross-references). Both
  documents must carry an ISO date in the file name:
  - `relative_date`: "yesterday" in a note whose document is dated the day
    after the other's.
  - `dated_reference`: a back-reference phrase ("compared with", "as noted in",
    "follow-up to", …) naming the other, earlier document's date.
  - `answered_question`: a question in the earlier note whose content words
    (at least three, and at least three quarters of them) appear in one
    statement of a same-day or later note.

  The cue's `direction` names the note that carries it.
- **Throughput.** Each anchor's neighbour read was a full exact cosine scan
  (about 0.47 s per anchor on a 9.8k-note corpus). That cost does not depend
  on how many anchors share the scan. With `similarity_batch_anchors > 1`, one
  pass reads the visible corpus once in record-ID pages of 2048 and ranks the
  neighbours of up to that many anchors together. It applies the same
  exclusions, floor, limits and tie order, and computes the cosine exactly as
  `vector::similarity::cosine` does, so results are identical. A group is
  examined as a unit, so with a short deadline use a smaller group.

#### Cue precision (#230, opt-in)

In the second reviewed batch, 6 of 11 rejects were recurring heartbeat or
status entries nominated by `relative_date` alone: "since 22:37 yesterday" or
"yesterday's notes" named a monitoring window or an aggregate log, not the
paired note. Three stricter options address this. Each is off by default and
only ever removes pairs:

```toml
[gardener.candidates]
resolve_relative_dates = true    # requires textual_cues = true
periodic_status_entries = 3      # 2..=1000
dedupe_counterparts = true
```

- **`resolve_relative_dates`.** A "yesterday" cue counts only if it resolves
  to the counterpart: one sentence that says "yesterday" names no clock time
  other than those in the counterpart's entry heading, *and* shares a two-word
  topic span with the counterpart (adjacent content words; stop words, temporal
  words and numbers break a span). A pair that also shares a specific entity
  keeps the cue either way. An unresolved cue falls through to the weaker cue
  kinds; if nothing else signals, the pair is counted as
  `unresolved_relative_date`.
- **`periodic_status_entries`.** This is a structural test for recurring
  heartbeat, watchdog or status logs, with no names hard-coded. A note's
  *entry template* is its last heading, lowercased with digit runs collapsed,
  and applies only when that heading carries a clock time
  (`19:07 MDT — Heartbeat recheck` becomes `#:# mdt — heartbeat recheck`). An
  entry is a run of consecutive chunks of the source, in chunk order, that
  share one heading path. A long section split into several chunks is one
  entry, and a repeated heading under another section is another. If at least
  this many entries of the note's own source share its template, the note is a
  periodic status entry. Such a note pairs only through
  an explicit cross-reference (record ID, source URI, wikilink or file name).
  Otherwise the pair is counted as `periodic_status`. The check costs one
  indexed heading read per source involved.
- **`dedupe_counterparts`.** After ranking, an endpoint already selected with
  a counterpart of identical (trimmed) content is not paired with another
  copy. The better-ranked copy is kept, and the rest are counted as
  `duplicate_counterpart`.

Every receipt echoes the new policy values (`null`/`false`/`1` when off) and
counts every new rejection reason. When any gate is on, persisted candidate
features also record `policy.quality` and `textual_cue`. The three
cue-precision options appear in `policy.quality` only while they are on. Measurements and the
recommended live settings are in
[gardener-quality-230](validation/gardener-quality-230.md).
