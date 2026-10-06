use super::Migration;
use crate::{DbConnection, Result};
use std::ops::Bound;
use surrealdb::types::{RecordId, RecordIdKey, RecordIdKeyRange};

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
DEFINE EVENT materialize_note_search ON note WHEN $event = 'CREATE' OR (
    $event = 'UPDATE' AND (
        $before.content != $after.content OR
        $before.search_content != $after.search_content OR
        $before.title != $after.title OR
        $before.created_at != $after.created_at OR
        $before.source_id != $after.source_id OR
        $before.source_generation != $after.source_generation
    )
)
    THEN fn::materialize_note_search($after);
DEFINE EVENT delete_note_search ON note WHEN $event = 'DELETE'
    THEN (DELETE type::record('note_search', record::id($before.id)));

-- The runner backfills by exclusive native-key ranges, at most 128 IDs per
-- query. It records this migration only after every page has committed.
"#,
};

pub(super) async fn backfill(db: &DbConnection) -> Result<()> {
    let mut after: Option<RecordId> = None;
    loop {
        let range = RecordId::new(
            "note",
            RecordIdKey::Range(Box::new(RecordIdKeyRange {
                start: after
                    .as_ref()
                    .map_or(Bound::Unbounded, |id| Bound::Excluded(id.key.clone())),
                end: Bound::Unbounded,
            })),
        );
        let ids: Vec<RecordId> = db
            .query("RETURN ((SELECT VALUE id FROM $range ORDER BY id ASC LIMIT 128) ?? []);")
            .bind(("range", range))
            .await?
            .take(0)?;
        let Some(last) = ids.last().cloned() else {
            return Ok(());
        };
        db.query("FOR $id IN $ids { fn::materialize_note_search($id.*); }; RETURN NONE;")
            .bind(("ids", ids))
            .await?
            .check()?;
        after = Some(last);
    }
}
