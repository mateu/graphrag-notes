use super::Migration;

/// Reindex staging must retain the source fields that formed the canonical
/// ingestion input, not merely a displayed body string. Existing staging is
/// intentionally discarded so an interrupted pre-v014 worker cannot publish
/// a body-only vector after the new model identity is cut over.
pub(super) const MIGRATION: Migration = Migration {
    version: 14,
    name: "reindex_input_snapshots",
    sql: r#"
DEFINE FIELD IF NOT EXISTS reindex_source_snapshot ON note TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS reindex_source_snapshot ON message TYPE option<object> FLEXIBLE;
DEFINE FIELD IF NOT EXISTS reindex_source_snapshot ON conversation TYPE option<object> FLEXIBLE;
-- Claude content blocks are provider-shaped JSON. Preserve their text
-- fallback verbatim rather than rejecting a nested block during reindex.
DEFINE FIELD IF NOT EXISTS content_blocks.* ON message TYPE object FLEXIBLE;
DEFINE FIELD IF NOT EXISTS content_blocks.*.* ON message TYPE any;

UPDATE note SET reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE;
UPDATE message SET reindex_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE;
UPDATE conversation SET reindex_summary_embedding = NONE, reindex_source_text = NONE, reindex_staging_owner = NONE;
UPDATE processing_job SET completed_count = 0, checkpoint = NONE
    WHERE job_type = 'reindex' AND status IN ['queued', 'running', 'failed', 'cancelled'];
"#,
};
