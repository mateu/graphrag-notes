//! Bounded reads for the read-only Alchemist synthesis.

use super::*;

/// Accepted relation tables the Alchemist may read. Proposal tables never are.
const RELATION_TABLES: [&str; 4] = ["supports", "contradicts", "derived_from", "related_to"];
/// Most notes [`Repository::accepted_edges_among`] accepts; matches the
/// Alchemist's evidence ceiling.
pub const MAX_EDGES_AMONG_NOTES: usize = 32;
/// Rows read per source note and table. One note can have at most one edge
/// per other member and table, so this only trips on duplicate edges.
const EDGES_AMONG_PER_NOTE_LIMIT: usize = 2 * MAX_EDGES_AMONG_NOTES;

/// One indexed statement per relation table and source note: the `in`
/// endpoint is matched by equality and `out` is restricted to the member set
/// *before* the limit applies, so edges to notes outside the set can never
/// crowd out edges inside it.
pub(super) fn edges_among_sql(note_count: usize) -> String {
    let mut sql = String::new();
    for table in RELATION_TABLES {
        for index in 0..note_count {
            sql.push_str(&format!(
                "SELECT id, '{table}' AS edge_type, in AS in_id, out AS out_id, proposal_id, confidence, reason, provenance, is_manual, created_at \
                 FROM {table} WHERE in = $note_{index} AND out IN $members \
                 ORDER BY id ASC LIMIT $limit;"
            ));
        }
    }
    sql
}

impl Repository {
    /// Resolve a public `note:key` ID to its stored record ID. Application
    /// notes use string keys, including UUID-shaped ones that
    /// [`parse_record_id`] reads as native UUIDs; portable imports may carry
    /// native numeric or UUID keys. As in `inspect_record`, an ambiguous pair
    /// is rejected and a hidden literal record never redirects to another key.
    /// Missing and hidden notes resolve to `None`.
    #[instrument(skip(self))]
    pub async fn resolve_visible_note_id(&self, id: &str) -> Result<Option<RecordId>> {
        #[derive(Deserialize, SurrealValue)]
        struct Candidate {
            id: RecordId,
            visible: bool,
        }
        let typed = parse_record_id(id, Some("note"))?;
        let literal = normalize_note_id(id.trim());
        let mut candidates = vec![literal.clone()];
        if typed != literal {
            candidates.push(typed);
        }
        let rows: Vec<Candidate> = self
            .db
            .query(format!(
                "SELECT id, {VISIBLE_NOTE_CONDITION} AS visible FROM $candidates"
            ))
            .bind(("candidates", candidates))
            .await?
            .take(0)?;
        if rows.len() > 1 {
            return Err(DbError::QueryFailed(format!(
                "{id} exists with both string and native keys; reimport these records with distinct IDs"
            )));
        }
        Ok(rows
            .into_iter()
            .next()
            .filter(|row| row.visible)
            .map(|row| row.id))
    }

    /// Accepted note edges whose two endpoints are both in `note_ids`, in
    /// every relation table, sorted by table and ID. The flag reports whether
    /// a per-note limit was reached (only possible with duplicate edges).
    #[instrument(skip(self, note_ids))]
    pub async fn accepted_edges_among(
        &self,
        note_ids: &[RecordId],
    ) -> Result<(Vec<NoteEdgeRow>, bool)> {
        let mut seen = HashSet::new();
        let members: Vec<RecordId> = note_ids
            .iter()
            .filter(|id| id.table.as_str() == "note" && !id.key.is_range())
            .filter(|id| seen.insert(record_id_to_string(id)))
            .cloned()
            .collect();
        if members.len() > MAX_EDGES_AMONG_NOTES {
            return Err(DbError::QueryFailed(format!(
                "accepted_edges_among supports at most {MAX_EDGES_AMONG_NOTES} notes"
            )));
        }
        if members.len() < 2 {
            return Ok((Vec::new(), false));
        }
        let limit = i64::try_from(EDGES_AMONG_PER_NOTE_LIMIT + 1)
            .map_err(|_| DbError::QueryFailed("edge limit exceeds integer range".into()))?;
        let mut query = self
            .db
            .query(edges_among_sql(members.len()))
            .bind(("members", members.clone()))
            .bind(("limit", limit));
        for (index, note) in members.iter().enumerate() {
            query = query.bind((format!("note_{index}"), note.clone()));
        }
        let mut response = query.await?.check()?;
        let mut truncated = false;
        let mut rows = Vec::new();
        for index in 0..RELATION_TABLES.len() * members.len() {
            let mut edges: Vec<NoteEdgeRow> = response.take(index)?;
            if edges.len() > EDGES_AMONG_PER_NOTE_LIMIT {
                truncated = true;
                edges.truncate(EDGES_AMONG_PER_NOTE_LIMIT);
            }
            rows.extend(edges);
        }
        rows.sort_by(|a, b| {
            a.edge_type
                .cmp(&b.edge_type)
                .then_with(|| record_id_to_string(&a.id).cmp(&record_id_to_string(&b.id)))
        });
        Ok((rows, truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::super::graph_retrieval_batch_tests::mention_plan_operators;
    use super::*;

    #[tokio::test]
    async fn edges_among_filter_members_before_the_limit_and_use_the_in_index() {
        let db = crate::init_memory().await.unwrap();
        // Two members, each with 80 higher-confidence edges to outsiders in
        // the same table, and one low-confidence edge between them.
        db.query(
            r#"
            CREATE note:a SET content='Fictional member A';
            CREATE note:b SET content='Fictional member B';
            FOR $i IN 0..80 {
                LET $outside = type::record('note', 'outside' + <string>$i);
                CREATE $outside SET content='Fictional outsider';
                CREATE type::record('derived_from', 'a' + <string>$i) SET in=note:a, out=$outside, confidence=1.0f, is_manual=true, created_at=time::now();
                CREATE type::record('derived_from', 'b' + <string>$i) SET in=$outside, out=note:b, confidence=1.0f, is_manual=true, created_at=time::now();
            };
            CREATE derived_from:inside SET in=note:a, out=note:b, confidence=0.1f, is_manual=true, created_at=time::now();
            CREATE contradicts:reverse SET in=note:b, out=note:a, is_manual=true, created_at=time::now();
            "#,
        )
        .await
        .unwrap()
        .check()
        .unwrap();
        let repo = Repository::new(db.clone());
        let (a, b) = (RecordId::new("note", "a"), RecordId::new("note", "b"));
        let (rows, truncated) = repo
            .accepted_edges_among(&[a.clone(), b.clone(), a.clone()])
            .await
            .unwrap();
        assert!(!truncated);
        let ids: Vec<String> = rows
            .iter()
            .map(|row| record_id_to_string(&row.id))
            .collect();
        assert_eq!(ids, ["contradicts:reverse", "derived_from:inside"]);
        assert!(repo
            .accepted_edges_among(std::slice::from_ref(&a))
            .await
            .unwrap()
            .0
            .is_empty());

        // UUID-shaped string keys (manual capture) keep their string type.
        let uuid = "6f1c2a5e-8d1b-4c4f-9a43-1d2e3f4a5b6c";
        db.query(format!(
            "CREATE type::record('note', '{uuid}') SET content='Fictional capture'"
        ))
        .await
        .unwrap()
        .check()
        .unwrap();
        let resolved = repo
            .resolve_visible_note_id(&format!("note:{uuid}"))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(resolved, RecordId::new("note", uuid.to_string()));
        assert_ne!(
            resolved,
            parse_record_id(&format!("note:{uuid}"), None).unwrap()
        );
        assert_eq!(
            repo.resolve_visible_note_id("note:a").await.unwrap(),
            Some(a.clone())
        );
        assert_eq!(
            repo.resolve_visible_note_id("note:missing").await.unwrap(),
            None
        );

        let sql = edges_among_sql(1);
        for (index, statement) in sql.split_terminator(';').enumerate() {
            let plans: Vec<serde_json::Value> = db
                .query(format!("{statement} EXPLAIN FULL"))
                .bind(("note_0", a.clone()))
                .bind(("members", vec![a.clone(), b.clone()]))
                .bind(("limit", 5i64))
                .await
                .unwrap()
                .take(0)
                .unwrap();
            let operators = mention_plan_operators(&plans[0]);
            let expected = format!("idx_{}_in", RELATION_TABLES[index]);
            assert!(
                !operators.iter().any(|op| op["operator"] == "TableScan"),
                "{plans:?}"
            );
            assert!(
                operators.iter().any(
                    |op| op["operator"] == "IndexScan" && op["attributes"]["index"] == expected
                ),
                "{expected}: {plans:?}"
            );
        }
    }
}
