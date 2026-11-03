# Alchemist: read-only, citation-verified synthesis

The Alchemist explains what your notes collectively say about a question or
around one note. It reads current chunks and **accepted** (reviewed) edges,
asks the configured model for a structured draft, and keeps only what Rust can
pin to a verbatim span of a cited chunk at its pinned revision. Design and
acceptance criteria: [#212](https://github.com/mateu/graphrag-notes/issues/212).

| Role | Writes | Output |
|---|---|---|
| Librarian | notes, entities | ingested knowledge |
| Gardener | pending proposals only | candidates for review |
| Review / typing (#205, #206) | pending typed proposals; people accept | accepted relations |
| **Alchemist** | **nothing** | a regenerable explanation that is never stored, is not evidence and never feeds retrieval |

## Enable it (opt-in)

Synthesis is disabled by default. Enable it on the host that owns the
database:

```toml
[alchemist]
enabled = true
# Defaults shown. Each value is both the default and the ceiling for a
# request; requests may only tighten them.
max_chunks = 8            # 1–32 evidence chunks
max_context_tokens = 4000 # 1–32768 evidence tokens shown to the model
max_output_tokens = 1024  # 1–8192 generated tokens
deadline_ms = 60000       # 1–300000; covers retrieval, generation and verification
max_hops = 2              # 1–2 accepted-edge hops around a seed
citation_policy = "drop"  # or "strict"
```

The provider and model are the existing `[inference]` extraction settings
(`extraction_provider`, `extraction_url`, `extraction_model`, timeouts and
`ollama_options`). There is no separate model configuration. When
`enabled = false`, the CLI and the service refuse before contacting any
provider.

## Use it

```bash
graphrag synthesize "what changed in the irrigation schedule?"   # query mode
graphrag synthesize --seed note:abc123 --hops 1                 # seed mode
graphrag synthesize --seed note:abc123 --policy strict --format json
graphrag --server https://notes.example/mcp synthesize --seed note:abc123
```

- **Query mode** reuses `build_context` retrieval (hybrid search, the
  accepted-edge graph channel and the existing packing), including its graph
  paths.
- **Seed mode** walks accepted `supports`, `contradicts`, `derived_from` and
  `related_to` edges in both directions, at most two hops. Each note and table
  has a fanout limit, and a full fanout page is reported as a truncated
  neighbourhood. Only visible notes are included, and proposal tables are
  never read.
- In both modes, the accepted edges *among* the evidence come from an indexed
  query. Endpoints are restricted to the evidence set before any limit, so
  edges from a high-degree note to outside notes cannot hide a lineage or
  contradiction edge between evidence notes.

MCP clients call `synthesize_notes` with the same fields: `query` or `seed`,
`scope`, `graph`, `max_chunks`, `max_context_tokens`, `max_output_tokens`,
`deadline_ms`, `max_hops` and `citation_policy`.

## Capability

`synthesize_notes` needs the new `alchemist` capability **and** `read`, because
its output quotes note text. `alchemist` grants no other tool, and no existing
grant implies it. The provisioning script does not grant it; add it explicitly
to a credential entry:

```json
{"instance_id": "reader-with-synthesis", "token_sha256": "…", "capabilities": ["read", "alchemist"]}
```

## Output

| Part | Meaning |
|---|---|
| `claims` | Each claim has at least one citation: chunk ID, pinned revision, verbatim `span`, byte offsets and origin. |
| `strength` | Wording from the number of lineage-deduplicated origins among a claim's citations. It is never a probability. |
| `relation_chains` | Accepted paths expressed between origin groups. A `derived_from` step inside one lineage is collapsed, so a summary and its origin count once. |
| `conflicts` | Accepted `contradicts` edges among the evidence, plus claims the model flags as conflicting. Each conflict needs at least two verified claims. Conflicts are surfaced, never resolved, and have no verdict field. |
| `timeline` | Claims ordered by the creation date of their earliest dated cited note. These are note dates, not event dates. |
| `open_questions` | Model-generated and unverified. |
| `gaps` | Deterministic: uncited evidence, dropped claims, single-origin claims, a truncated neighbourhood, and unassessed relations. |
| `lineage`, `relations`, `evidence` | The origin groups, accepted edges and pinned chunks that the synthesis was allowed to use. |
| `receipt` | Resolved budgets and ceilings, policy, usage and verification counts. |

### Origin groups

Chunks share an origin group when any of these holds:
- an accepted `derived_from` edge links them;
- they come from the same source document;
- they come from the same conversation;
- their content is identical after whitespace and case normalization.

Two chunks in one group never count as corroborating each other.

## The verifier fails closed

A citation is kept only if all of these hold:
- its chunk is in the pinned evidence set (an unknown label or ID is rejected);
- any revision the draft supplies equals the pinned revision;
- its span has at least 12 characters and appears **verbatim** in that chunk's pinned content. A quote that exists only in a different chunk is rejected; it is never re-attributed.

After generation, every cited chunk is re-read. A chunk that was edited,
superseded or removed since pinning cannot be cited.

Claims with no verified citation are dropped and counted (`drop`). Under
`strict`, any rejection fails the request. Model output that cannot be parsed
always fails it. Unknown fields in the model output are ignored, such as an
attempted `proposals` or `edges` list. The Alchemist has no write path to give
them.

## Guarantees and limits

- **No writes.** Storage is reached only through a read-only interface: record
  inspection, accepted-edge reads, accepted edges among the evidence, and
  visible-note reads. Tests snapshot
  every table before and after synthesis in both modes.
  - Caveat: query mode reuses `build_context`. Its shared embedding adapter
    can record a query-embedding cache entry, exactly as `build_context`
    already does. That entry is an inference cache, not a knowledge record.
- **No caching or persistence** of syntheses in v1.
- **Evidence assessments are absent.** Typed relation evidence (#205/#211) has
  not landed yet, so every accepted edge is reported as `unassessed`.
  Rendering `evidence_mode` and declared lineage is tracked in
  [#213](https://github.com/mateu/graphrag-notes/issues/213).
- **Lineage grouping is conservative.** Two different documents that copy one
  upstream source without an accepted `derived_from` edge still count as
  separate origins.
- A later version **may** submit hypothesis-mode **pending** proposals
  through #211's evidence assessment for human review. Those proposals are
  never auto-accepted.
