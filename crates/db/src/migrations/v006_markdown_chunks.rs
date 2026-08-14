use super::Migration;

/// Structure-aware Markdown chunk provenance and reconciliation keys.
pub(super) const MIGRATION: Migration = Migration {
    version: 6,
    name: "markdown_chunks",
    sql: r#"
DEFINE FIELD IF NOT EXISTS chunk_key ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS chunk_location_key ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS chunk_ordinal ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS chunk_heading_path ON note TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS source_start_line ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS source_end_line ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS source_start_byte ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS source_end_byte ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS chunk_overlap_from ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS chunk_overlap_chars ON note TYPE option<int>;
DEFINE FIELD IF NOT EXISTS content_hash ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS search_content ON note TYPE option<string>;

-- A source may contain the same displayed text twice, so identity includes the
-- normalized structural location. It is unique only within a source.
DEFINE INDEX IF NOT EXISTS idx_note_source_chunk_key ON note FIELDS source_id, chunk_key UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_note_source_chunk_location ON note FIELDS source_id, chunk_location_key;
DEFINE INDEX IF NOT EXISTS idx_note_search_content ON note FIELDS search_content
    FULLTEXT ANALYZER ascii BM25;
"#,
};
