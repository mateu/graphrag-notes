//! Aggregate statistics ownership.
//!
//! This read-only query deliberately remains one database statement so the
//! returned counts share one snapshot. Do not split it into per-table calls.
//!
//! Note-to-note edges are counted from [`ACCEPTED_NOTE_EDGE_TABLES`], the same
//! list orphan detection reads, so a new relation table cannot be traversed
//! without also being counted. `references` is an `EdgeType` variant but has
//! no table: persisted note-edge writes reject it, so there is nothing to
//! count for it.

use super::*;
use std::collections::BTreeMap;

/// Aggregate row returned by the single-snapshot statistics query.
#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
pub struct DbStats {
    #[serde(default)]
    pub note_count: i64,
    #[serde(default)]
    pub entity_count: i64,
    #[serde(default)]
    pub source_count: i64,
    #[serde(default)]
    pub conversation_count: i64,
    #[serde(default)]
    pub message_count: i64,
    #[serde(default)]
    pub mention_count: i64,
    #[serde(default)]
    pub note_conversation_link_count: i64,
    #[serde(default)]
    pub note_message_link_count: i64,
    /// Total accepted note-to-note edges across every table in
    /// [`ACCEPTED_NOTE_EDGE_TABLES`]; always the sum of `edge_counts`.
    #[serde(default)]
    pub edge_count: i64,
    /// Accepted note-to-note edges per relation table. Every table in
    /// [`ACCEPTED_NOTE_EDGE_TABLES`] is present, including zero counts.
    #[serde(default)]
    pub edge_counts: BTreeMap<String, i64>,
}

/// One-statement snapshot query. Per-type edge counts are generated from the
/// shared accepted note-edge table list rather than spelled out here.
fn stats_sql() -> String {
    let edge_counts = ACCEPTED_NOTE_EDGE_TABLES
        .iter()
        .map(|table| format!("{table}: (SELECT count() FROM {table} GROUP ALL)[0].count ?? 0"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
                RETURN {{
                    note_count: (SELECT count() FROM note GROUP ALL)[0].count,
                    entity_count: (SELECT count() FROM entity GROUP ALL)[0].count,
                    source_count: (SELECT count() FROM source GROUP ALL)[0].count,
                    conversation_count: (SELECT count() FROM conversation GROUP ALL)[0].count,
                    message_count: (SELECT count() FROM message GROUP ALL)[0].count,
                    mention_count: (SELECT count() FROM mentions GROUP ALL)[0].count,
                    note_conversation_link_count: (SELECT count() FROM note_from_conversation GROUP ALL)[0].count,
                    note_message_link_count: (SELECT count() FROM note_from_message GROUP ALL)[0].count,
                    edge_count: 0,
                    edge_counts: {{ {edge_counts} }}
                }}
            "#
    )
}

impl Repository {
    /// Get database statistics from one consistent query snapshot.
    #[instrument(skip(self))]
    pub async fn get_stats(&self) -> Result<DbStats> {
        let stats: Vec<DbStats> = self.db.query(stats_sql()).await?.take(0)?;

        let mut stats = stats
            .into_iter()
            .next()
            .ok_or_else(|| DbError::QueryFailed("stats".into()))?;
        // The total is derived from the breakdown so the two cannot disagree.
        stats.edge_count = stats.edge_counts.values().sum();
        Ok(stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;

    const ALL_EDGE_TYPES: [EdgeType; 7] = [
        EdgeType::Supports,
        EdgeType::Contradicts,
        EdgeType::DerivedFrom,
        EdgeType::References,
        EdgeType::RelatedTo,
        EdgeType::Mentions,
        EdgeType::TaggedWith,
    ];

    fn sorted_tables() -> Vec<&'static str> {
        let mut tables = ACCEPTED_NOTE_EDGE_TABLES.to_vec();
        tables.sort_unstable();
        tables
    }

    async fn note(repo: &Repository, content: &str) -> RecordId {
        repo.create_note(Note::new(content))
            .await
            .unwrap()
            .id
            .unwrap()
    }

    #[tokio::test]
    async fn stats_count_one_edge_of_every_persisted_note_edge_type() {
        let repo = Repository::new(init_memory().await.unwrap());
        let empty = repo.get_stats().await.unwrap();
        assert_eq!(empty.edge_count, 0);
        assert_eq!(
            empty
                .edge_counts
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            sorted_tables()
        );
        assert!(empty.edge_counts.values().all(|count| *count == 0));

        for edge_type in ALL_EDGE_TYPES
            .into_iter()
            .filter(|edge_type| graph::note_edge_table(edge_type).is_ok())
        {
            let from = note(&repo, &format!("Fictional {edge_type} origin")).await;
            let to = note(&repo, &format!("Fictional {edge_type} target")).await;
            repo.create_edge(&from, &to, edge_type, None).await.unwrap();
        }
        // A second derived_from edge keeps per-type counts distinguishable.
        let from = note(&repo, "Fictional second summary").await;
        let to = note(&repo, "Fictional second log").await;
        repo.create_edge(&from, &to, EdgeType::DerivedFrom, None)
            .await
            .unwrap();

        let stats = repo.get_stats().await.unwrap();
        let expected = BTreeMap::from([
            ("contradicts".to_owned(), 1),
            ("derived_from".to_owned(), 2),
            ("related_to".to_owned(), 1),
            ("supports".to_owned(), 1),
        ]);
        assert_eq!(stats.edge_counts, expected);
        assert_eq!(stats.edge_count, 5);
        assert_eq!(stats.note_count, 10);
    }

    /// Counting, orphan detection and Gardener evidence reads share
    /// `ACCEPTED_NOTE_EDGE_TABLES`. Pin that list to the edge-type mapping used
    /// for writes and to every schema table joining two notes, so a new
    /// note-to-note table cannot be written or traversed without being counted.
    #[tokio::test]
    async fn accepted_note_edge_tables_match_writable_types_and_schema() {
        let mut writable = ALL_EDGE_TYPES
            .iter()
            .filter_map(|edge_type| graph::note_edge_table(edge_type).ok())
            .collect::<Vec<_>>();
        writable.sort_unstable();
        assert_eq!(writable, sorted_tables());
        for table in ACCEPTED_NOTE_EDGE_TABLES {
            let edge_type = graph::persisted_note_edge_type(table).unwrap();
            assert_eq!(graph::note_edge_table(&edge_type).unwrap(), table);
        }
        // `references` is a note relation kind but is never persisted.
        assert!(EdgeType::References.is_note_edge());
        assert!(graph::note_edge_table(&EdgeType::References).is_err());

        let db = init_memory().await.unwrap();
        let schema = db
            .query("INFO FOR DB")
            .await
            .unwrap()
            .take::<Option<serde_json::Value>>(0)
            .unwrap()
            .unwrap();
        let mut note_to_note = Vec::new();
        for table in schema["tables"].as_object().unwrap().keys() {
            let info = db
                .query(format!("INFO FOR TABLE {table}"))
                .await
                .unwrap()
                .take::<Option<serde_json::Value>>(0)
                .unwrap()
                .unwrap();
            let endpoint_is_note = |field: &str| {
                info["fields"][field]
                    .as_str()
                    .is_some_and(|definition| definition.contains("TYPE record<note>"))
            };
            // Proposals are pending review, not accepted edges.
            if table != "proposed_edge" && endpoint_is_note("in") && endpoint_is_note("out") {
                note_to_note.push(table.clone());
            }
        }
        note_to_note.sort_unstable();
        assert_eq!(note_to_note, sorted_tables());
    }
}
