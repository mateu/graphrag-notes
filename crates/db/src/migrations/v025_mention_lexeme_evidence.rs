use super::Migration;
use crate::{DbConnection, Result};

pub(super) const MIGRATION: Migration = Migration {
    version: 25,
    name: "mention_lexeme_evidence",
    sql: r#"
DEFINE TABLE IF NOT EXISTS mention_lexeme SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS mention ON mention_lexeme TYPE record<mentions>;
DEFINE FIELD IF NOT EXISTS entity ON mention_lexeme TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS note ON mention_lexeme TYPE record;
DEFINE FIELD IF NOT EXISTS lexeme ON mention_lexeme TYPE string;
DEFINE FIELD IF NOT EXISTS mention_identity ON mention_lexeme TYPE string;
DEFINE INDEX IF NOT EXISTS uq_mention_lexeme ON mention_lexeme FIELDS mention_identity, lexeme UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_mention_lexeme_lookup ON mention_lexeme FIELDS lexeme;
DEFINE INDEX IF NOT EXISTS idx_mention_lexeme_entity ON mention_lexeme FIELDS entity;
DEFINE FUNCTION fn::materialize_mention_lexemes($mention: record<mentions>) {
    DELETE mention_lexeme WHERE mention_identity = type::string($mention);
    LET $row = (SELECT * FROM $mention)[0];
    IF $row != NONE {
        LET $entity = (SELECT canonical_name FROM $row.out)[0];
        LET $values = array::concat([$entity.canonical_name], IF type::is_array($row.metadata.aliases) THEN array::slice($row.metadata.aliases, 0, 8) ELSE [] END);
        FOR $value IN $values {
            IF type::is_string($value) {
                LET $lexeme = string::trim(string::replace(string::lowercase($value), <regex>"[^\\p{L}\\p{N}]+", ' '));
                IF $lexeme != '' { INSERT INTO mention_lexeme (mention, mention_identity, entity, note, lexeme) VALUES ($mention, type::string($mention), $row.out, $row.in, $lexeme) ON DUPLICATE KEY UPDATE entity = $row.out, note = $row.in; };
            };
        };
    };
    RETURN NONE;
};
DEFINE EVENT materialize_mention_lexemes ON mentions WHEN $event != 'DELETE' THEN fn::materialize_mention_lexemes($after.id);
DEFINE EVENT delete_mention_lexemes ON mentions WHEN $event = 'DELETE' THEN (DELETE mention_lexeme WHERE mention_identity = type::string($before.id));
DEFINE EVENT rename_mention_lexemes ON entity WHEN $event != 'DELETE' AND $before.canonical_name != $after.canonical_name THEN {
    FOR $mention IN (SELECT VALUE id FROM mentions WHERE out = $after.id) { fn::materialize_mention_lexemes($mention); };
};
DEFINE EVENT delete_entity_mention_lexemes ON entity WHEN $event = 'DELETE' THEN (DELETE mention_lexeme WHERE entity = $before.id);
"#,
};

pub(super) async fn backfill(db: &DbConnection) -> Result<()> {
    let started = std::time::Instant::now();
    // Native primary-key ranges preserve numeric, UUID, and compound keys.
    use std::ops::Bound;
    use surrealdb::types::{RecordId, RecordIdKey, RecordIdKeyRange};
    let mut after: Option<RecordId> = None;
    let mut count = 0usize;
    loop {
        let range = RecordId::new(
            "mentions",
            RecordIdKey::Range(Box::new(RecordIdKeyRange {
                start: after
                    .as_ref()
                    .map_or(Bound::Unbounded, |id| Bound::Excluded(id.key.clone())),
                end: Bound::Unbounded,
            })),
        );
        let ids: Vec<RecordId> = db
            .query("RETURN ((SELECT VALUE id FROM $range LIMIT 128) ?? []);")
            .bind(("range", range))
            .await?
            .take(0)?;
        let Some(last) = ids.last().cloned() else {
            break;
        };
        count += ids.len();
        db.query("FOR $id IN $ids { fn::materialize_mention_lexemes($id); }; RETURN NONE;")
            .bind(("ids", ids))
            .await?
            .check()?;
        after = Some(last);
    }
    tracing::info!(
        mentions = count,
        elapsed_ms = started.elapsed().as_millis(),
        "mention evidence backfill complete"
    );
    Ok(())
}
