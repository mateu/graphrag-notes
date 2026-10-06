use super::Migration;

/// Keep every note in the lexical population, including unembedded and hidden
/// generations: BM25 statistics must match the original primary indexes.
/// Mirroring the native key preserves numeric/string/UUID/object tie ordering.
/// Events update the derived row inside the primary write transaction. Portable
/// archives contain primary records only; restored writes rebuild this table.
pub(super) const MIGRATION: Migration = Migration {
    version: 21,
    name: "note_search",
    sql: r#"
DEFINE TABLE note_search SCHEMAFULL;
DEFINE FIELD record_id ON note_search TYPE record<note>;
DEFINE FIELD content ON note_search TYPE option<string>;
DEFINE FIELD search_content ON note_search TYPE option<string>;
DEFINE FIELD title ON note_search TYPE option<string>;
DEFINE FIELD created_at ON note_search TYPE option<datetime>;
DEFINE FIELD source_id ON note_search TYPE option<record<source>>;
DEFINE FIELD source_generation ON note_search TYPE option<int>;
DEFINE INDEX idx_note_search_content ON note_search FIELDS content FULLTEXT ANALYZER ascii BM25;
DEFINE INDEX idx_note_search_context ON note_search FIELDS search_content FULLTEXT ANALYZER ascii BM25;
DEFINE INDEX idx_note_search_title ON note_search FIELDS title FULLTEXT ANALYZER ascii BM25;

DEFINE FUNCTION fn::materialize_note_search($row: object) {
    UPSERT type::record('note_search', record::id($row.id)) CONTENT {
        record_id: $row.id, content: $row.content,
        search_content: $row.search_content, title: $row.title,
        created_at: $row.created_at, source_id: $row.source_id,
        source_generation: $row.source_generation
    };
    RETURN NONE;
};
DEFINE EVENT materialize_note_search ON note WHEN $event != 'DELETE'
    THEN fn::materialize_note_search($after);
DEFINE EVENT delete_note_search ON note WHEN $event = 'DELETE'
    THEN (DELETE type::record('note_search', record::id($before.id)));

-- The pinned engine can return NONE for an empty VALUE selection.
FOR $id IN ((SELECT VALUE id FROM note ORDER BY id ASC) ?? []) {
    fn::materialize_note_search($id.*);
};
"#,
};
