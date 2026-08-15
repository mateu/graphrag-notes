use super::Migration;

/// A reindex worker owns a short renewable lease. Staged source snapshots and
/// the same owner token make a stale worker unable to publish its generation.
pub(super) const MIGRATION: Migration = Migration {
    version: 13,
    name: "reindex_ownership",
    sql: r#"
DEFINE FIELD IF NOT EXISTS reindex_lease_owner ON processing_job TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_lease_expires_at ON processing_job TYPE option<datetime>;

DEFINE FIELD IF NOT EXISTS reindex_source_text ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_staging_owner ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_source_text ON message TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_staging_owner ON message TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_source_text ON conversation TYPE option<string>;
DEFINE FIELD IF NOT EXISTS reindex_staging_owner ON conversation TYPE option<string>;
"#,
};
