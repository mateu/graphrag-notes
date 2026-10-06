# Entity quality and graph seeds

Extraction recognizes Person (`per`), Organization (`organisation`, `org`),
Concept, Project, Technology (`tech`, `tool`), Location (`loc`, `gpe`), Date
(`time`), and Other, ignoring case and surrounding whitespace. An omitted type
keeps the older Concept behavior; an explicit unsupported type becomes Other.
The provider prompt and optional JSON schema request these supported names.

Aliases are optional. Older string/object entity payloads and cached payloads
without aliases remain readable. A scalar alias field, non-string elements,
blank/control-character labels, and labels longer than 80 characters are
ignored. Valid aliases are normalized by case and whitespace, deduplicated,
sorted deterministically, and capped at eight per prepared entity. An alias
equal to the canonical label is redundant. Original mention and valid alias
spellings remain in extraction metadata on the owning mention; the linked note and its source retain
the original text and provenance. Alias hints never assert entity identity or
create accepted note edges. Strict malformed JSON still fails; the existing
tolerant JSON recovery remains available when configured.

Schema 19 adds an explicit entity identity key, optional mention evidence, a
durable optional note extraction scope, and an index on entity/note mention pairs. It preserves every existing entity ID, type, mention, and
canonical label. Older/manual entity upserts and older portable backups use
the legacy canonical-name key. New extracted keys include type and evidence
scope; matching still uses the readable canonical label and aliases.

People and projects use note-local evidence scopes. Generated source chunks
retain an opaque extraction scope across safely reconciled one-to-one successors,
even when a generation has no entities. A new unmatched chunk starts with its
final note ID, so positional shifts cannot reuse another live chunk's scope.
Older scoped Person/Project evidence can supply a validated scope before its
first durable adoption; conflicting scopes fail before replacing mentions. Other types can share a canonical label only within
the same generated source; manual/detached notes use their final note ID even
when retaining a source link as provenance. Capture and edit preparation uses
the final note ID, including authenticated capture request IDs, so later forced
extraction reuses these identities. Same spelling across sources,
different types, and aliases do not merge rows. Two same-named people/projects
in different notes remain ambiguous independent entities. Within one note the
provider must distinguish them with different labels; extraction has no
evidence to infer a global identity. Safe source successors preserve scoped
keys across earlier insertions, removals and forced reprocessing; an uncertain
reconciliation creates independent identities. This deliberately avoids silently relabeling the
older globally deduplicated entities.

Extracted entity display metadata reflects the latest prepared result instead
of accumulating aliases across edits. Graph alias matching reads current,
source/time-eligible mention evidence: replacing one note retracts its old
aliases while preserving another chunk's valid aliases on a shared source
entity. Exact-content source successors copy this evidence. Deleted notes and
hidden generations contribute no aliases. Each mention and returned match
keeps the eight-alias cap; legacy/manual aliases retain their older behavior.
Seed ranking and strength use only the candidate note's own eligible mention
aliases, so a shared entity cannot lend another chunk's query evidence. A bounded
query-matching mention page precedes the direct-rank and ID fallback pages.

Graph matching keeps whole-query/contained-phrase/prefix tiers and stable ties.
For retrieval it requires a visible, source/time-eligible mention and prefers
entities attached to the bounded direct search candidates. Orphaned legacy
entities cannot crowd out useful scoped matches. An indexed provider-free full-text supplement recovers lexical candidates
just below the requested hybrid result count. Seed selection intersects up to
200 candidates, ranked by supported query terms and exact titles, with indexed
mentions before the bounded ID-ordered fallback page. Each entity keeps its configured seed page, and the
unique seed cap still reserves coverage across matched entities.

Before selecting seed notes, graph strength uses the fraction of meaningful
query terms supported by the candidate note/title and matched entity labels or
aliases. A verified mention retains a small recovery floor; it does not claim
unsupported query details. Strength is calibrated by the existing fusion
strategy, then accepted-edge confidence and hop decay apply. Exact direct
titles, source/time/current-generation filters, visited-node exclusion,
accepted-only traversal, and all configured bounds remain enforced. Graph
explanations retain only the actual entity associations and accepted paths.

## Correct a small pilot explicitly

First verify a private backup and run the frozen direct/graph evaluation on an
isolated restored corpus. Select a small, named set of note IDs and keep that
selection private. Use the candidate binary and the intended extraction model:

```sh
graphrag extract-entities --note-id note:PILOT_A --note-id note:PILOT_B --force
graphrag jobs list
graphrag jobs show JOB_ID
graphrag jobs resume JOB_ID
graphrag show-entities note:PILOT_A
graphrag search --graph=off -- 'Fictional Atlas retry policy'
graphrag search --graph=auto -- 'Fictional Atlas retry policy'
```

`--force` performs inference before replacing the selected note's complete
mention set and evidence in one transaction. Failure preserves its previous mentions and last good source
generation; durable jobs pin their selected items and resume after their
checkpoint. Repeating a completed forced selection reuses its scoped entity
IDs without duplicate mentions. Unselected entities/notes are not rewritten.
The extraction cache version changes with the type/alias prompt, so an explicit
reprocessing pass does not silently reuse the older schema's extraction.

Review type/alias examples and relevant search evidence, then compare the same
frozen calibration and holdout reports plus paired latency observations.
Document improvements and regressions before approving a wider pilot. This
change does not schedule broad corpus extraction or guarantee that adding
entities improves every query; extracted relationships remain provider hints,
and accepted note edges require their existing governed review workflow.

See the [fictional validation observations](validation/graph-relevance-93.md)
for paired scaling results and regression coverage.
