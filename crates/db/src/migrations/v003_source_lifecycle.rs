use super::Migration;

/// Source lifecycle metadata. This migration is intentionally self-contained:
/// it neither rewrites legacy source rows nor changes chat identifiers.
pub(super) const MIGRATION: Migration = Migration {
    version: 3,
    name: "source_lifecycle",
    sql: r#"
-- File source identity is a normalized URI. Manual sources leave this NONE,
-- so intentionally duplicated manual thoughts remain distinct.
DEFINE FIELD IF NOT EXISTS normalized_uri ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content_hash ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS generation ON source TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS successful_generation ON source TYPE int DEFAULT 0;
DEFINE FIELD IF NOT EXISTS status ON source TYPE string DEFAULT 'ready';
DEFINE FIELD IF NOT EXISTS last_error ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS updated_at ON source TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS last_ingested_at ON source TYPE option<datetime>;
DEFINE INDEX IF NOT EXISTS idx_source_normalized_uri ON source FIELDS normalized_uri UNIQUE;

-- A source generation marks records the import owns. Legacy/manual notes keep
-- NONE and are therefore never selected by lifecycle replacement/deletion.
DEFINE FIELD IF NOT EXISTS source_generation ON note TYPE option<int>;
DEFINE INDEX IF NOT EXISTS idx_note_source_generation ON note FIELDS source_id, source_generation;
"#,
};
