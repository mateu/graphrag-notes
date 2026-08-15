# Repository module ownership

`repository.rs` is the stable `Repository` façade. It owns construction and
the public re-exports; callers must not depend on a domain module path.

| Module | Owns | Transaction / ordering invariant |
| --- | --- | --- |
| `ids` | Canonical `table:key` parsing and raw note-key normalization | Bind a typed `RecordId` once; never add a second table prefix. |
| `portable` | Logical archive table allow-list, paged export, and restore conversion | Export remains `ORDER BY id ASC`; every restore record's timestamp/reference casts are bound in the same create statement. |
| `stats` | Aggregate `DbStats` query | Counts are returned by one query snapshot, not independent table reads. |
| `repository.rs` (transitional) | Notes, search, chats, sources, entities, graph, jobs, metadata, and proposals | These move mechanically into like-named modules in follow-up slices; existing SQL and lifecycle locks must move verbatim. |

Migrations remain in `crate::migrations`; repository modules never alter a
published migration. Cross-domain lifecycle promotion, proposal acceptance,
and dependent-copy operations retain their shared connection-scoped lifecycle
lock when their owning methods move.
