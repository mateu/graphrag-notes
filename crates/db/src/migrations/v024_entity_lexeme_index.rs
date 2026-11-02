use super::Migration;
use crate::{DbConnection, Result};
use std::ops::Bound;
use surrealdb::types::{RecordId, RecordIdKey, RecordIdKeyRange};

/// Materialize every entity name and alias spelling into a compact lexical
/// candidate index. Graph query SQL remains authoritative for eligibility,
/// scope, ranking, and caps; this projection only narrows candidates.
pub(super) const MIGRATION: Migration = Migration {
    version: 24,
    name: "entity_lexeme_index",
    sql: r#"
DEFINE TABLE IF NOT EXISTS entity_lexeme SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS entity ON entity_lexeme TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS lexeme ON entity_lexeme TYPE string;
DEFINE INDEX IF NOT EXISTS uq_entity_lexeme ON entity_lexeme FIELDS entity, lexeme UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_entity_lexeme_lookup ON entity_lexeme FIELDS lexeme;

DEFINE FUNCTION fn::materialize_entity_lexemes($entity: record<entity>) {
    DELETE entity_lexeme WHERE entity = $entity;
    LET $row = (SELECT * FROM $entity)[0];
    IF $row != NONE {
        LET $values = array::flatten(array::concat([$row.name, $row.canonical_name], IF type::is_array($row.metadata.aliases) THEN $row.metadata.aliases ELSE [] END, (SELECT VALUE metadata.aliases ?? [] FROM mentions WHERE out = $entity)));
        FOR $value IN $values {
            IF type::is_string($value) {
                LET $lexeme = string::trim(string::replace(string::lowercase($value), <regex>"[^\\p{L}\\p{N}]+", ' '));
                IF $lexeme != '' { INSERT INTO entity_lexeme (entity, lexeme) VALUES ($entity, $lexeme) ON DUPLICATE KEY UPDATE entity = $entity; };
            };
        };
    };
    RETURN NONE;
};
DEFINE EVENT materialize_entity_lexemes ON entity WHEN $event = 'CREATE' OR ($event = 'UPDATE' AND ($before.name != $after.name OR $before.canonical_name != $after.canonical_name OR $before.metadata.aliases != $after.metadata.aliases)) THEN fn::materialize_entity_lexemes($after.id);
DEFINE EVENT delete_entity_lexemes ON entity WHEN $event = 'DELETE' THEN (DELETE entity_lexeme WHERE entity = $before.id);
-- Mention removals may leave extra candidates: the authoritative query rejects
-- them. Incremental additions avoid quadratic rebuilds for high-degree entities.
DEFINE FUNCTION fn::add_mention_entity_lexemes($entity: record<entity>, $aliases: any) {
    IF type::is_array($aliases) {
        FOR $value IN array::distinct($aliases) {
            IF type::is_string($value) {
                LET $lexeme = string::trim(string::replace(string::lowercase($value), <regex>"[^\\p{L}\\p{N}]+", ' '));
                IF $lexeme != '' { INSERT INTO entity_lexeme (entity, lexeme) VALUES ($entity, $lexeme) ON DUPLICATE KEY UPDATE entity = $entity; };
            };
        };
    };
    RETURN NONE;
};
DEFINE EVENT materialize_mention_entity_lexemes ON mentions WHEN $event != 'DELETE' THEN fn::add_mention_entity_lexemes($after.out, $after.metadata.aliases);
"#,
};

pub(super) async fn backfill(db: &DbConnection) -> Result<()> {
    let mut after: Option<RecordId> = None;
    loop {
        let range = RecordId::new(
            "entity",
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
        db.query("FOR $id IN $ids { fn::materialize_entity_lexemes($id); }; RETURN NONE;")
            .bind(("ids", ids))
            .await?
            .check()?;
        after = Some(last);
    }
}
