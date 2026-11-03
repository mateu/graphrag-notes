//! Bounded, read-only evidence reads for Gardener relationship candidates.
//!
//! Accepted note relations are ordinary records written with explicit
//! `in`/`out` fields (`INSERT`/`CREATE`, never `RELATE`). SurrealDB graph
//! arrows such as `->supports->note` therefore cannot see them, so
//! connectivity is always read from the endpoint fields of every accepted
//! note-edge table. Every read here has an explicit order and caller cap.

use super::graph::graph_endpoint_visible_sql;
use super::*;

/// Every persisted, accepted note-to-note relation table.
pub const ACCEPTED_NOTE_EDGE_TABLES: [&str; 4] =
    ["supports", "contradicts", "derived_from", "related_to"];

/// Distinct visible notes joined to another visible note by an accepted
/// relation of any type, in either direction.
pub(super) fn connected_note_ids_sql() -> String {
    let in_visible = graph_endpoint_visible_sql("in");
    let out_visible = graph_endpoint_visible_sql("out");
    let tables = ACCEPTED_NOTE_EDGE_TABLES
        .iter()
        .map(|table| {
            format!("(SELECT VALUE [in, out] FROM {table} WHERE {in_visible} AND {out_visible})")
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("array::distinct(array::flatten(array::flatten([{tables}])))")
}

/// Normalize a name exactly like the `mention_lexeme` projection (v025):
/// lowercase, collapse runs outside Unicode letters/numbers to one space, trim.
pub(crate) fn mention_lexeme(name: &str) -> String {
    static BOUNDARY: std::sync::OnceLock<surrealdb_types::Regex> = std::sync::OnceLock::new();
    let boundary =
        BOUNDARY.get_or_init(|| r"[^\p{L}\p{N}]+".parse().expect("constant lexical regex"));
    boundary
        .replace_all(&name.to_lowercase(), " ")
        .trim()
        .to_owned()
}

/// Which visible notes a candidate scan may examine as anchors.
#[derive(Debug, Clone, Copy)]
pub enum CandidateAnchorScope<'a> {
    /// Exactly these notes (missing, hidden or embedding-free IDs are skipped).
    Notes(&'a [RecordId]),
    /// Visible notes owned by these sources.
    Sources(&'a [RecordId]),
    /// Visible notes without an accepted relation to another visible note.
    Orphans,
}

/// A visible note with the provenance needed to judge a candidate pair.
#[derive(Debug, Clone, Deserialize, SurrealValue)]
pub struct CandidateNote {
    pub id: RecordId,
    #[serde(default)]
    pub source_id: Option<RecordId>,
    #[serde(default)]
    pub source_uri: Option<String>,
    pub content: String,
    #[serde(default)]
    pub content_hash: Option<String>,
    /// Present for anchors; neighbours omit it.
    #[serde(default)]
    #[surreal(default)]
    pub embedding: Vec<f32>,
    /// Cosine similarity to the anchor; anchors leave it unset.
    #[serde(default)]
    pub similarity: Option<f32>,
}

/// One entity mention of a candidate note.
#[derive(Debug, Clone, Deserialize, SurrealValue)]
pub struct CandidateMention {
    pub note: RecordId,
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub kind: Option<String>,
}

/// Bounded document frequency for one entity name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CandidateNameFrequency {
    /// Distinct notes observed with a mention lexeme equal to the name.
    pub notes: usize,
    /// The probe hit its row cap, so `notes` is only a lower bound.
    pub truncated: bool,
}

/// A proposal already joining two candidate notes.
#[derive(Debug, Clone, Deserialize, SurrealValue)]
pub struct CandidateExistingProposal {
    pub from_id: RecordId,
    pub to_id: RecordId,
    pub edge_type: String,
    pub status: String,
    pub generator: String,
}

/// What recording one scan candidate did.
#[derive(Debug)]
pub enum CandidateProposalOutcome {
    /// Recorded as a new pending proposal.
    Created(Box<ProposedEdge>),
    /// Something already joins the pair, as read under the proposal
    /// lifecycle lock; nothing was written or refreshed.
    Existing(CandidateExistingLinks),
}

/// Accepted relations and proposals among a set of candidate notes.
#[derive(Debug, Clone, Default)]
pub struct CandidateExistingLinks {
    /// Accepted relations of any type, as `(in, out)`.
    pub accepted: Vec<(RecordId, RecordId)>,
    pub proposals: Vec<CandidateExistingProposal>,
}

const CANDIDATE_NOTE_FIELDS: &str = "id, source_id, (source_id.normalized_uri ?? source_id.uri) AS source_uri, content, content_hash";

fn note_ids_only(ids: &[RecordId]) -> Vec<RecordId> {
    let mut ids = ids
        .iter()
        .filter(|id| id.table.as_str() == "note" && !id.key.is_range())
        .cloned()
        .collect::<Vec<_>>();
    ids.sort_by_key(record_id_to_string);
    ids.dedup();
    ids
}

impl Repository {
    /// Visible anchor notes that have an embedding, ordered by record ID,
    /// strictly after `after`, at most `limit` rows.
    #[instrument(skip(self, scope))]
    pub async fn candidate_anchor_notes(
        &self,
        scope: CandidateAnchorScope<'_>,
        after: Option<&RecordId>,
        limit: usize,
    ) -> Result<Vec<CandidateNote>> {
        let filter = format!(
            "{VISIBLE_NOTE_CONDITION} AND embedding IS NOT NONE AND ($after = NONE OR id > $after)"
        );
        let order = "ORDER BY id ASC LIMIT $limit";
        let mut query = match scope {
            CandidateAnchorScope::Notes(ids) => {
                let ids = note_ids_only(ids);
                if ids.is_empty() {
                    return Ok(Vec::new());
                }
                // Selecting bound records reads only the requested notes.
                self.db
                    .query(format!(
                        "SELECT {CANDIDATE_NOTE_FIELDS}, embedding FROM $ids WHERE {filter} {order}"
                    ))
                    .bind(("ids", ids))
            }
            CandidateAnchorScope::Sources(sources) => self
                .db
                .query(format!(
                    "SELECT {CANDIDATE_NOTE_FIELDS}, embedding FROM note WHERE source_id IN $sources AND {filter} {order}"
                ))
                .bind(("sources", sources.to_vec())),
            CandidateAnchorScope::Orphans => self.db.query(format!(
                "LET $connected = {}; SELECT {CANDIDATE_NOTE_FIELDS}, embedding FROM note WHERE {filter} AND id NOT IN $connected {order};",
                connected_note_ids_sql()
            )),
        };
        query = query.bind(("after", after.cloned())).bind(("limit", limit));
        let index = usize::from(matches!(scope, CandidateAnchorScope::Orphans));
        Ok(query.await?.check()?.take(index)?)
    }

    /// Nearest visible notes from a *different* origin than the anchor, at or
    /// above `floor`, ordered by similarity then record ID. Same-source notes
    /// are excluded before the limit, so adjacent chunks cannot crowd out
    /// cross-source neighbours.
    #[instrument(skip(self, anchor, counterpart_sources))]
    pub async fn candidate_similar_notes(
        &self,
        anchor: &CandidateNote,
        floor: f32,
        limit: usize,
        counterpart_sources: Option<&[RecordId]>,
    ) -> Result<Vec<CandidateNote>> {
        if anchor.embedding.is_empty() || limit == 0 {
            return Ok(Vec::new());
        }
        let rows: Vec<CandidateNote> = self
            .db
            .query(format!(
                "SELECT {CANDIDATE_NOTE_FIELDS}, vector::similarity::cosine(embedding, $embedding) AS similarity \
                 FROM note \
                 WHERE id != $anchor AND embedding IS NOT NONE AND array::len(embedding) = $dimension \
                 AND ($anchor_source = NONE OR source_id IS NONE OR source_id != $anchor_source) \
                 AND ($counterparts = NONE OR source_id IN $counterparts) \
                 AND {VISIBLE_NOTE_CONDITION} \
                 AND vector::similarity::cosine(embedding, $embedding) >= $floor \
                 ORDER BY similarity DESC, id ASC LIMIT $limit"
            ))
            .bind(("anchor", anchor.id.clone()))
            .bind(("embedding", anchor.embedding.clone()))
            .bind(("dimension", anchor.embedding.len()))
            .bind(("anchor_source", anchor.source_id.clone()))
            .bind(("counterparts", counterpart_sources.map(<[RecordId]>::to_vec)))
            .bind(("floor", floor))
            .bind(("limit", limit))
            .await?
            .take(0)?;
        Ok(rows)
    }

    /// Entity names and types mentioned by the supplied notes.
    #[instrument(skip(self, notes))]
    pub async fn candidate_note_mentions(
        &self,
        notes: &[RecordId],
    ) -> Result<Vec<CandidateMention>> {
        let notes = note_ids_only(notes);
        if notes.is_empty() {
            return Ok(Vec::new());
        }
        Ok(self
            .db
            .query(
                "SELECT in AS note, out.canonical_name AS name, out.entity_type AS kind \
                 FROM mentions WHERE in IN $notes",
            )
            .bind(("notes", notes))
            .await?
            .take(0)?)
    }

    /// For each name, count distinct notes whose mention evidence carries the
    /// same lexeme, reading at most `probe` indexed rows per name.
    #[instrument(skip(self, names))]
    pub async fn candidate_name_frequencies(
        &self,
        names: &[String],
        probe: usize,
    ) -> Result<Vec<CandidateNameFrequency>> {
        if names.is_empty() {
            return Ok(Vec::new());
        }
        let probe = probe.max(1);
        let sql = (0..names.len())
            .map(|index| {
                format!("SELECT VALUE note FROM mention_lexeme WHERE lexeme = $lexeme_{index} LIMIT $probe;")
            })
            .collect::<String>();
        let mut query = self.db.query(sql).bind(("probe", probe));
        for (index, name) in names.iter().enumerate() {
            query = query.bind((format!("lexeme_{index}"), mention_lexeme(name)));
        }
        let mut response = query.await?.check()?;
        let mut frequencies = Vec::with_capacity(names.len());
        for index in 0..names.len() {
            let rows: Vec<RecordId> = response.take(index)?;
            let truncated = rows.len() >= probe;
            let notes = rows
                .iter()
                .map(record_id_to_string)
                .collect::<HashSet<_>>()
                .len();
            frequencies.push(CandidateNameFrequency { notes, truncated });
        }
        Ok(frequencies)
    }

    /// Accepted relations (any type) and proposals (any generator/status)
    /// whose two endpoints are both among `notes`.
    #[instrument(skip(self, notes))]
    pub async fn candidate_existing_links(
        &self,
        notes: &[RecordId],
    ) -> Result<CandidateExistingLinks> {
        let notes = note_ids_only(notes);
        if notes.len() < 2 {
            return Ok(CandidateExistingLinks::default());
        }
        #[derive(Deserialize, SurrealValue)]
        struct EdgeRow {
            from_id: RecordId,
            to_id: RecordId,
        }
        let sql = ACCEPTED_NOTE_EDGE_TABLES
            .iter()
            .map(|table| {
                format!("SELECT in AS from_id, out AS to_id FROM {table} WHERE in IN $notes AND out IN $notes;")
            })
            .collect::<String>()
            + "SELECT in AS from_id, out AS to_id, edge_type, status, generator FROM proposed_edge WHERE in IN $notes AND out IN $notes;";
        let mut response = self.db.query(sql).bind(("notes", notes)).await?.check()?;
        let mut links = CandidateExistingLinks::default();
        for index in 0..ACCEPTED_NOTE_EDGE_TABLES.len() {
            let rows: Vec<EdgeRow> = response.take(index)?;
            links
                .accepted
                .extend(rows.into_iter().map(|row| (row.from_id, row.to_id)));
        }
        links.proposals = response.take(ACCEPTED_NOTE_EDGE_TABLES.len())?;
        Ok(links)
    }

    /// Record a scan candidate as a new pending proposal unless an accepted
    /// relation of any type or a proposal of any kind already joins the pair.
    /// The pair-wide check and the insert run in one hold of the proposal
    /// lifecycle lock, so no lifecycle writer can slip in between them, and
    /// an existing proposal is never refreshed: losing the unique dedupe-key
    /// race to any other writer is reported as `Existing` too. `Ok(None)`
    /// means the lock was not acquired by `deadline`; nothing was read or
    /// written. Once the lock is held, the check and write run to completion.
    pub async fn record_candidate_proposal_before(
        &self,
        mut draft: EdgeProposalDraft,
        deadline: tokio::time::Instant,
    ) -> Result<Option<CandidateProposalOutcome>> {
        let Ok(_lifecycle_guard) =
            tokio::time::timeout_at(deadline, self.proposal_acceptance_lock.lock()).await
        else {
            return Ok(None);
        };
        let dedupe_key = super::graph::prepare_edge_proposal(&mut draft)?;
        let pair = [draft.from_id.clone(), draft.to_id.clone()];
        let existing = self.candidate_existing_links(&pair).await?;
        if !existing.accepted.is_empty() || !existing.proposals.is_empty() {
            return Ok(Some(CandidateProposalOutcome::Existing(existing)));
        }
        Ok(Some(
            match self
                .insert_pending_edge_proposal(draft, &dedupe_key)
                .await?
            {
                Some(created) => CandidateProposalOutcome::Created(Box::new(created)),
                None => {
                    CandidateProposalOutcome::Existing(self.candidate_existing_links(&pair).await?)
                }
            },
        ))
    }
}
