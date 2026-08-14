use super::Migration;

/// Structure-aware Markdown chunk provenance and reconciliation keys.
pub(super) const MIGRATION: Migration = Migration {
    version: 8,
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
DEFINE FIELD IF NOT EXISTS split_fenced_code ON note TYPE option<bool>;
DEFINE FIELD IF NOT EXISTS content_hash ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS search_content ON note TYPE option<string>;

-- Updating legacy rows validates every schemafull field. Materialize the
-- collection default before the search-content backfill touches those rows.
UPDATE note SET chunk_heading_path = [] WHERE chunk_heading_path = NONE;

-- Upgrade rows must receive the same FTS contribution as new rows. Without
-- this, old notes only match the lower-weight `content` predicate.
UPDATE note SET search_content = content WHERE search_content = NONE;

-- A source may contain the same displayed text twice, so identity includes the
-- normalized structural location. It is unique within each staged source
-- generation, allowing crash-safe copy-on-write reconciliation.
-- Reconciliation stages a complete pending generation before promotion, so a
-- structurally stable chunk key is unique per source generation, not across
-- all historical generations of a source.
DEFINE INDEX IF NOT EXISTS idx_note_source_chunk_key ON note FIELDS source_id, source_generation, chunk_key UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_note_source_chunk_location ON note FIELDS source_id, chunk_location_key;
DEFINE INDEX IF NOT EXISTS idx_note_search_content ON note FIELDS search_content
    FULLTEXT ANALYZER ascii BM25;
"#,
};
