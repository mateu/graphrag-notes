# Review proposed connections

`garden review` puts both notes, their excerpts and provenance, and the
proposal's confidence, reason, generator, and lifecycle state in one place.
Once proposals exist, review and decisions work without inference providers.
These commands are in the current source build; the published `v0.1.0-rc.1`
binary predates the inbox.

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
`accept_allowed`, `accept_blocked_reason`, and replay commands. Endpoints expose
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
