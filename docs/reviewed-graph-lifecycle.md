# Reviewed graph lifecycle (development slice)

This is a service-owner library conversion plus an opt-in pending-only MCP proposal tool. It does not install a worker, grant credentials, run Gardener, or accept proposals automatically.

## Vector-only source conversion

`GraphRagApplication::enrich_uploaded_source` takes an explicit `SourceEnrichmentPlan`: canonical source ID, owner instance, request identity, opening source revision, content SHA-256, generation, original processing policy and complete target processing policy, plus confirmation. Target policy must exactly match the configured entity-enabled runtime with unchanged chunking/embedding policy. The original vector-only upload must be completed and its live source generation ready. Direct/manual entity overlays are refused.

Private `source.metadata.entity_enrichment_v1` stores the exact plan, original upload origin, ordered note IDs/revisions and bounded sequential entity batches (16 MiB total, max 200 chunks, max 128 entities/chunk). Provider failure never publishes mentions, changes the visible extraction policy or marks graph freshness. Identical retries resume completed batches; a dedicated owning-datastore gate prevents concurrent identical application calls repeating providers. This is not an unattended distributed job queue.

Promotion atomically fences exact source metadata/content/generation and complete source-owned note inventory, publishes all entity/mention replacements, changes the applied policy and increments a separate graph policy epoch. Source ID, URI, content generation, upload request identity, note IDs, embeddings and content remain unchanged. Source readback reports processing/extraction policy currency independently of content freshness. Note inspection revisions include graph policy epoch.

`rollback_source_entity_enrichment(plan, true)` is explicit forward compensation. It restores the original upload origin, removes only the complete generated mention inventory and increments policy epoch again (no ABA revision rewind). Refuse source replacement, note drift, foreign mentions or any relationship/proposal dependency. Shared/orphan entities are retained, not globally garbage-collected.

## Future source generations

Managed enriched sources retain `graph_enrichment_managed=true`. Silent vector-only downgrade is refused. A replacement must opt into entities and supply an exact source revision. The existing durable policy-migration preparation/checkpoint/promotion machinery is reused for graph rebuilding even when processing policy itself is unchanged: last-good chunks/mentions/edges stay visible until every new chunk extraction succeeds, then source visibility and all rebuilt links promote atomically. Failed/cancelled work keeps its private stage for owner-token-fenced resume. Reviewed exact-endpoint edges are not copied to successor chunks; their evidence is superseded during successful old-generation retirement. Later generations retain the same staged-rebuild requirement.

## Exact endpoint proposal

`propose_endpoint_relationship` requires the opt-in `propose` capability. Request fields are `request_id`, exactly one `from` and one `to` (each canonical note ID, SHA-256 revision and exact nonempty content quote), `relationship`, `rationale`, `confirmed=true`. Unknown fields/cardinality expansions are rejected. Allowed relation types only: `supports`, `contradicts`, `derived_from`, `related_to`. Similarity must not be represented as causal dependency. Quotes are bounded to 2048 bytes each, rationale to 2048 bytes. Types/quotes establish a reviewed assertion, not an automatic semantic truth validator.

One transaction fences both full note/source snapshots and creates exactly one pending proposal plus one mutation receipt. No model inference, entity mutations, graph edge or automatic acceptance occurs. Request-key replay returns the same persisted outcome; changed payload under the same shared instance/request identity conflicts, including cross-operation reuse. Symmetric endpoint order is canonical. Reviewed proposal dedupe includes both immutable endpoint revisions/quotes; materialized-edge dedupe remains endpoint/type-specific.

Acceptance is separate. Remote acceptance uses the existing confirm/revision/decision capability and atomic journal. Generic service-owner acceptance for this family requires an explicit manual reviewer and uses one complete note/source/proposal-snapshot-fenced transaction (no intermediate `accepting` state). Supported note edit/policy promotion/replacement invalidates only this reviewed family and its derived edges; unrelated manually created edges are not deleted. Raw out-of-contract datastore mutation is not an authorized editing API.

## Compatibility and limitations

Schema 22 adds optional nonnegative `source.metadata.graph_policy_revision`; staging remains inside the existing flexible source metadata. Existing schema-21 stores migrate forward without a corpus scan. Older binaries must refuse the newer store. The lexical migration replay fixture now removes all subsequent migration history when manufacturing a schema20 corpus.

Default/automatic memory clients receive no new capability. Existing readers/status tool catalogs remain unchanged; `propose` exposes only creation, not `decide`/`accept`. Conversion is intentionally not an MCP tool yet: an authenticated review transport/queue and complete backup/restore/restart qualification remain rollout work. The importer requires separate explicitly reviewed receipt adoption; local unchanged-document skip behavior is not automatically rewritten.

The synthetic off/on/on/off fixture proves one-hop accepted-edge evidence reaches a preselected fact under fixed 2/1000/400 context budgets and fixed `per_hop_decay=1.0`; this is not a default-ranking or live-corpus benchmark. It asserts edge/path/endpoint identity, positive traversal diagnostics and disappearance after explicit undo.
