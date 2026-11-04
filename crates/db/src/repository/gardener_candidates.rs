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
    /// The document this note was imported from: the origin URI an uploader
    /// declared for its source (one file split into several uploaded parts
    /// shares it), else the source's own URI. Used only for lineage checks.
    #[serde(default)]
    pub origin_uri: Option<String>,
}

/// Origin document of a note's source, as [`CandidateNote::origin_uri`].
const ORIGIN_URI_SQL: &str = "(IF type::is_string(source_id.metadata.remote_upload.source.uri) THEN source_id.metadata.remote_upload.source.uri ELSE (source_id.normalized_uri ?? source_id.uri) END)";

/// Corpus rows read per page of a batched similarity pass.
const CANDIDATE_BATCH_PAGE: usize = 2048;

/// Bounded note evidence for one entity name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateNameNotes {
    /// Distinct notes observed with a mention lexeme equal to the name.
    pub notes: Vec<RecordId>,
    /// The probe hit its row cap, so `notes` is only a lower bound.
    pub truncated: bool,
}

/// Exactly what `vector::similarity::cosine(stored, bound)` computes in
/// SurrealDB 3.3: f64 products summed left to right, divided by the product
/// of the two magnitudes. Mirroring it keeps a batched pass bit-identical to
/// the per-anchor query.
fn surreal_cosine(stored: &[f64], anchor: &[f64]) -> f64 {
    let dot = stored
        .iter()
        .zip(anchor)
        .fold(0.0_f64, |sum, (left, right)| sum + left * right);
    let magnitude = |vector: &[f64]| vector.iter().map(|value| value.powi(2)).sum::<f64>().sqrt();
    dot / (magnitude(stored) * magnitude(anchor))
}

/// Similarity-descending, then record-ID-ascending, like the per-anchor
/// `ORDER BY similarity DESC, id ASC`. Native record IDs keep distinct keys
/// with one display form (`note:42` and `note:⟨42⟩`) distinct.
fn neighbour_order(left: &(f64, RecordId), right: &(f64, RecordId)) -> std::cmp::Ordering {
    right
        .0
        .total_cmp(&left.0)
        .then_with(|| left.1.cmp(&right.1))
}

/// Keep the best `limit` entries of `ranked` (unsorted), in neighbour order.
fn keep_best(ranked: &mut Vec<(f64, RecordId)>, limit: usize) {
    ranked.sort_by(neighbour_order);
    ranked.truncate(limit);
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
    /// An endpoint is no longer visible (deleted, or its source generation
    /// retired) under the lifecycle lock; nothing was written.
    Vanished,
}

/// Accepted relations and proposals among a set of candidate notes.
#[derive(Debug, Clone, Default)]
pub struct CandidateExistingLinks {
    /// Accepted relations of any type, as `(in, out)`.
    pub accepted: Vec<(RecordId, RecordId)>,
    pub proposals: Vec<CandidateExistingProposal>,
}

fn candidate_note_fields() -> String {
    format!("id, source_id, (source_id.normalized_uri ?? source_id.uri) AS source_uri, content, content_hash, {ORIGIN_URI_SQL} AS origin_uri")
}

/// Statements (indices 0–2) binding `$excluded` (notes sharing a chat
/// conversation with `$anchor`) and `$joined` (notes already joined to it by
/// an accepted relation or any proposal).
fn anchor_exclusions_sql() -> String {
    let joined = ACCEPTED_NOTE_EDGE_TABLES
        .iter()
        .chain(std::iter::once(&"proposed_edge"))
        .map(|table| {
            format!(
                "(SELECT VALUE out FROM {table} WHERE in = $anchor), \
                 (SELECT VALUE in FROM {table} WHERE out = $anchor)"
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "LET $origins = array::union(\
             (SELECT VALUE out FROM note_from_conversation WHERE in = $anchor), \
             (SELECT VALUE (out.conversation_id ?? out) FROM note_from_message WHERE in = $anchor)); \
         LET $excluded = IF array::len($origins) = 0 THEN [] ELSE array::union(\
             (SELECT VALUE in FROM note_from_conversation WHERE out IN $origins), \
             (SELECT VALUE in FROM note_from_message WHERE out IN array::union($origins, \
                 (SELECT VALUE id FROM message WHERE conversation_id IN $origins)))) END; \
         LET $joined = array::distinct(array::flatten([{joined}]));"
    )
}

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
        let fields = candidate_note_fields();
        let mut query = match scope {
            CandidateAnchorScope::Notes(ids) => {
                let ids = note_ids_only(ids);
                if ids.is_empty() {
                    return Ok(Vec::new());
                }
                // Selecting bound records reads only the requested notes.
                self.db
                    .query(format!(
                        "SELECT {fields}, embedding FROM $ids WHERE {filter} {order}"
                    ))
                    .bind(("ids", ids))
            }
            CandidateAnchorScope::Sources(sources) => self
                .db
                .query(format!(
                    "SELECT {fields}, embedding FROM note WHERE source_id IN $sources AND {filter} {order}"
                ))
                .bind(("sources", sources.to_vec())),
            CandidateAnchorScope::Orphans => self.db.query(format!(
                "LET $connected = {}; SELECT {fields}, embedding FROM note WHERE {filter} AND id NOT IN $connected {order};",
                connected_note_ids_sql()
            )),
        };
        query = query.bind(("after", after.cloned())).bind(("limit", limit));
        let index = usize::from(matches!(scope, CandidateAnchorScope::Orphans));
        Ok(query.await?.check()?.take(index)?)
    }

    /// Nearest visible notes from a *different* origin than the anchor
    /// (another source and chat conversation), at or above `floor`, ordered
    /// by similarity then record ID: up to `limit` not yet joined to the
    /// anchor, then up to `limit` already joined to it by an accepted relation
    /// or any proposal, which the gate only reports. Same-origin and joined
    /// notes are excluded before the limit, so adjacent chunks, conversation
    /// siblings and recorded pairs cannot crowd out eligible neighbours.
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
        // Notes sharing a chat conversation with the anchor, directly or
        // through any of its messages (the origins `candidate_note_origins`
        // compares), are excluded like same-source notes. Notes already
        // joined to the anchor by an accepted relation or any proposal never
        // take a place in the neighbour budget either: they are read
        // separately (at most `limit` of them) only so the gate can report
        // them, and recording proposals can never hide the next neighbours.
        let exclusions = anchor_exclusions_sql();
        let filters = format!(
            "id != $anchor AND id NOT IN $excluded \
             AND embedding IS NOT NONE AND array::len(embedding) = $dimension \
             AND ($anchor_source = NONE OR source_id IS NONE OR source_id != $anchor_source) \
             AND ($counterparts = NONE OR source_id IN $counterparts) \
             AND {VISIBLE_NOTE_CONDITION} \
             AND vector::similarity::cosine(embedding, $embedding) >= $floor"
        );
        let select = format!(
            "SELECT {}, vector::similarity::cosine(embedding, $embedding) AS similarity",
            candidate_note_fields()
        );
        let mut response = self
            .db
            .query(format!(
                "{exclusions} \
                 {select} FROM note WHERE {filters} AND id NOT IN $joined \
                 ORDER BY similarity DESC, id ASC LIMIT $limit; \
                 {select} FROM $joined WHERE {filters} \
                 ORDER BY similarity DESC, id ASC LIMIT $limit;"
            ))
            .bind(("anchor", anchor.id.clone()))
            .bind(("embedding", anchor.embedding.clone()))
            .bind(("dimension", anchor.embedding.len()))
            .bind(("anchor_source", anchor.source_id.clone()))
            .bind((
                "counterparts",
                counterpart_sources.map(<[RecordId]>::to_vec),
            ))
            .bind(("floor", floor))
            .bind(("limit", limit))
            .await?
            .check()?;
        let mut rows: Vec<CandidateNote> = response.take(3)?;
        let joined: Vec<CandidateNote> = response.take(4)?;
        rows.extend(joined);
        Ok(rows)
    }

    /// [`Repository::candidate_similar_notes`] for many anchors in one exact
    /// pass: the visible corpus is read once, in record-ID pages of bounded
    /// size, instead of one full similarity scan per anchor. Each anchor's
    /// result equals what the per-anchor read returns (same exclusions,
    /// floor, limits, order and similarity values): the cosine is computed
    /// exactly as SurrealDB's `vector::similarity::cosine`.
    #[instrument(skip(self, anchors, counterpart_sources))]
    pub async fn candidate_similar_notes_batch(
        &self,
        anchors: &[CandidateNote],
        floor: f32,
        limit: usize,
        counterpart_sources: Option<&[RecordId]>,
    ) -> Result<Vec<Vec<CandidateNote>>> {
        use std::collections::{BTreeMap, BTreeSet};
        use std::ops::Bound;
        use surrealdb::types::{RecordIdKey, RecordIdKeyRange};

        struct Ranking {
            excluded: BTreeSet<RecordId>,
            joined: BTreeSet<RecordId>,
            anchor: Vec<f64>,
            fresh: Vec<(f64, RecordId)>,
            recorded: Vec<(f64, RecordId)>,
        }
        #[derive(Deserialize, SurrealValue)]
        struct Exclusions {
            #[serde(default)]
            excluded: Vec<RecordId>,
            #[serde(default)]
            joined: Vec<RecordId>,
        }
        #[derive(Deserialize, SurrealValue)]
        struct Row {
            id: RecordId,
            #[serde(default)]
            source_id: Option<RecordId>,
            embedding: Vec<f64>,
        }

        let mut rankings = Vec::with_capacity(anchors.len());
        for anchor in anchors {
            if anchor.embedding.is_empty() || limit == 0 {
                rankings.push(None);
                continue;
            }
            let mut response = self
                .db
                .query(format!(
                    "{} RETURN {{ excluded: $excluded, joined: $joined }};",
                    anchor_exclusions_sql()
                ))
                .bind(("anchor", anchor.id.clone()))
                .await?
                .check()?;
            let exclusions: Option<Exclusions> = response.take(3)?;
            let exclusions = exclusions.unwrap_or(Exclusions {
                excluded: Vec::new(),
                joined: Vec::new(),
            });
            rankings.push(Some(Ranking {
                excluded: exclusions.excluded.into_iter().collect(),
                joined: exclusions.joined.into_iter().collect(),
                anchor: anchor
                    .embedding
                    .iter()
                    .map(|value| f64::from(*value))
                    .collect(),
                fresh: Vec::new(),
                recorded: Vec::new(),
            }));
        }
        if rankings.iter().all(Option::is_none) {
            return Ok(vec![Vec::new(); anchors.len()]);
        }

        let floor = f64::from(floor);
        let counterparts = counterpart_sources.map(<[RecordId]>::to_vec);
        let mut after: Option<RecordId> = None;
        loop {
            // A record range continues strictly after the last row read, so
            // every page is an ordered key-range read, never a full re-scan.
            let range = RecordId::new(
                "note",
                RecordIdKey::Range(Box::new(RecordIdKeyRange {
                    start: after
                        .as_ref()
                        .map_or(Bound::Unbounded, |id| Bound::Excluded(id.key.clone())),
                    end: Bound::Unbounded,
                })),
            );
            let page: Vec<Row> = self
                .db
                .query(format!(
                    "SELECT id, source_id, embedding FROM $range \
                     WHERE embedding IS NOT NONE \
                       AND ($counterparts = NONE OR source_id IN $counterparts) \
                       AND {VISIBLE_NOTE_CONDITION} \
                     ORDER BY id ASC LIMIT {CANDIDATE_BATCH_PAGE}"
                ))
                .bind(("range", range))
                .bind(("counterparts", counterparts.clone()))
                .await?
                .check()?
                .take(0)?;
            let Some(last) = page.last() else {
                break;
            };
            after = Some(last.id.clone());
            let full_page = page.len() >= CANDIDATE_BATCH_PAGE;
            for (index, row) in page.into_iter().enumerate() {
                // Cosine work is synchronous: yield regularly so the scan's
                // deadline and other tasks are never starved.
                if index % 256 == 255 {
                    tokio::task::yield_now().await;
                }
                for (anchor, ranking) in anchors.iter().zip(rankings.iter_mut()) {
                    let Some(ranking) = ranking else { continue };
                    if row.id == anchor.id
                        || row.embedding.len() != ranking.anchor.len()
                        || ranking.excluded.contains(&row.id)
                        || anchor.source_id.as_ref().is_some_and(|source| {
                            row.source_id.as_ref().is_some_and(|row| row == source)
                        })
                    {
                        continue;
                    }
                    let similarity = surreal_cosine(&row.embedding, &ranking.anchor);
                    if similarity.is_nan() || similarity < floor {
                        continue;
                    }
                    let bucket = if ranking.joined.contains(&row.id) {
                        &mut ranking.recorded
                    } else {
                        &mut ranking.fresh
                    };
                    bucket.push((similarity, row.id.clone()));
                    if bucket.len() >= limit.saturating_mul(4).max(64) {
                        keep_best(bucket, limit);
                    }
                }
            }
            if !full_page {
                break;
            }
        }

        let mut selected = Vec::new();
        for ranking in rankings.iter_mut().flatten() {
            keep_best(&mut ranking.fresh, limit);
            keep_best(&mut ranking.recorded, limit);
            for (_, id) in ranking.fresh.iter().chain(&ranking.recorded) {
                selected.push(id.clone());
            }
        }
        selected.sort();
        selected.dedup();
        let hydrated: Vec<CandidateNote> = if selected.is_empty() {
            Vec::new()
        } else {
            self.db
                .query(format!("SELECT {} FROM $ids", candidate_note_fields()))
                .bind(("ids", selected))
                .await?
                .check()?
                .take(0)?
        };
        // Native `RecordId` keys: distinct records may share a display form.
        #[allow(clippy::mutable_key_type)]
        let hydrated = hydrated
            .into_iter()
            .map(|note| (note.id.clone(), note))
            .collect::<BTreeMap<_, _>>();
        Ok(rankings
            .into_iter()
            .map(|ranking| {
                let Some(ranking) = ranking else {
                    return Vec::new();
                };
                ranking
                    .fresh
                    .into_iter()
                    .chain(ranking.recorded)
                    .filter_map(|(similarity, id)| {
                        hydrated.get(&id).map(|note| CandidateNote {
                            similarity: Some(similarity as f32),
                            ..note.clone()
                        })
                    })
                    .collect()
            })
            .collect())
    }

    /// For each name, the distinct notes whose mention evidence carries the
    /// same lexeme, reading at most `probe` indexed rows per name.
    #[instrument(skip(self, names))]
    pub async fn candidate_name_notes(
        &self,
        names: &[String],
        probe: usize,
    ) -> Result<Vec<CandidateNameNotes>> {
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
        let mut evidence = Vec::with_capacity(names.len());
        for index in 0..names.len() {
            let rows: Vec<RecordId> = response.take(index)?;
            let truncated = rows.len() >= probe;
            let mut notes = rows;
            notes.sort_by_key(record_id_to_string);
            notes.dedup();
            evidence.push(CandidateNameNotes { notes, truncated });
        }
        Ok(evidence)
    }

    /// Origin document ([`CandidateNote::origin_uri`]) of each supplied note
    /// that has one, keyed by native note ID (distinct records may share a
    /// display form).
    #[allow(clippy::mutable_key_type)]
    #[instrument(skip(self, notes))]
    pub async fn candidate_note_origin_uris(
        &self,
        notes: &[RecordId],
    ) -> Result<std::collections::BTreeMap<RecordId, String>> {
        let notes = note_ids_only(notes);
        if notes.is_empty() {
            return Ok(Default::default());
        }
        #[derive(Deserialize, SurrealValue)]
        struct OriginRow {
            id: RecordId,
            #[serde(default)]
            origin_uri: Option<String>,
        }
        let rows: Vec<OriginRow> = self
            .db
            .query(format!(
                "SELECT id, {ORIGIN_URI_SQL} AS origin_uri FROM $ids"
            ))
            .bind(("ids", notes))
            .await?
            .check()?
            .take(0)?;
        Ok(rows
            .into_iter()
            .filter_map(|row| Some((row.id, row.origin_uri?)))
            .collect())
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

    /// Chat origins of candidate notes: each note's conversations, linked
    /// directly or through the conversation of each message it came from (a
    /// message whose record is gone stands for itself). Two notes share an
    /// origin when their sets intersect.
    pub async fn candidate_note_origins(
        &self,
        notes: &[RecordId],
    ) -> Result<std::collections::HashMap<String, std::collections::BTreeSet<String>>> {
        let notes = note_ids_only(notes);
        let mut origins =
            std::collections::HashMap::<String, std::collections::BTreeSet<String>>::new();
        if notes.is_empty() {
            return Ok(origins);
        }
        #[derive(Deserialize, SurrealValue)]
        struct OriginRow {
            note: RecordId,
            origin: RecordId,
        }
        let mut response = self
            .db
            .query(
                "SELECT in AS note, out AS origin FROM note_from_conversation WHERE in IN $notes; \
                 SELECT in AS note, (out.conversation_id ?? out) AS origin FROM note_from_message WHERE in IN $notes;",
            )
            .bind(("notes", notes))
            .await?
            .check()?;
        for index in 0..2 {
            let rows: Vec<OriginRow> = response.take(index)?;
            for row in rows {
                origins
                    .entry(record_id_to_string(&row.note))
                    .or_default()
                    .insert(record_id_to_string(&row.origin));
            }
        }
        Ok(origins)
    }

    /// Record a scan candidate as a new pending proposal unless an endpoint
    /// is no longer visible or an accepted relation of any type or a proposal
    /// of any kind already joins the pair. The visibility and pair-wide
    /// checks and the insert run in one hold of the proposal lifecycle lock
    /// (which note deletion and source retirement also hold), so no
    /// lifecycle writer can slip in between them, and
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
        if !self.note_is_visible(&draft.from_id).await?
            || !self.note_is_visible(&draft.to_id).await?
        {
            return Ok(Some(CandidateProposalOutcome::Vanished));
        }
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
