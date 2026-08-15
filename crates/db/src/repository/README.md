# Repository module ownership

`repository.rs` is the stable `Repository` façade. It owns construction and
the public re-exports; callers must not depend on a domain module path.

| Module | Owns | Transaction / ordering invariant |
| --- | --- | --- |
| `ids` | Canonical `table:key` parsing and raw note-key normalization | Bind a typed `RecordId` once; never add a second table prefix. |
| `models` | Shared public query/result rows | Data-only types are re-exported through the façade; SQL remains in the owning domain. |
| `metadata` | Embedding/extraction compatibility reads and initialization | Delegates to compatibility helpers; never rewrites a non-empty corpus identity. |
| `jobs` | Durable inference cache, processing jobs, and staged reindex ownership | Cache writes are a single semantic-key UPSERT; lease, staging, and commit statements retain their original owner/fingerprint transaction checks. |
| `notes` | Note CRUD plus note/message/conversation search retrieval | Search SQL, fusion inputs, filters, and explicit ordering are moved verbatim. |
| `graph` | Entities, accepted edges, graph traversal, and proposal lifecycle | Proposal acceptance/undo retain the shared lifecycle lock; traversal keeps deterministic fanout/order. |
| `sources` | Source staging, promotion, deletion, and dependent copy | Visibility replacement and proposal supersession remain in their original transaction/lock order. |
| `chats` | Conversation/message persistence and note provenance links | Conversation/message link checks remain writable-note guarded. |
| `portable` | Logical archive table allow-list, paged export, and restore conversion | Export remains `ORDER BY id ASC`; every restore record's timestamp/reference casts are bound in the same create statement. |
| `stats` | Aggregate `DbStats` query | Counts are returned by one query snapshot, not independent table reads. |
| `repository.rs` | Stable façade construction, shared constants/helpers, and public re-exports | Callers continue using `Repository`; no domain module path is part of the public API. |

Migrations remain in `crate::migrations`; repository modules never alter a
published migration. Cross-domain lifecycle promotion, proposal acceptance,
and dependent-copy operations retain their shared connection-scoped lifecycle
lock when their owning methods move.
