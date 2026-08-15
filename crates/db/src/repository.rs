//! Repository pattern for database operations

// The public façade remains this module. Implementations are split by table
// ownership below; transaction-sensitive cross-domain paths stay documented
// at their owning domain rather than introducing a second data-access layer.
mod chats;
mod graph;
mod ids;
mod jobs;
mod metadata;
mod models;
mod notes;
mod portable;
mod sources;
mod stats;

pub use chats::{EdgeProposalDraft, GraphEntityMatch, GraphEntityNoteSeed, NoteEdgeRow};
pub use graph::{SourceDeleteSummary, SourceImportAction, SourceImportPlan};
use ids::normalize_note_id;
pub use ids::parse_record_id;
pub use jobs::{
    InferenceCacheEntry, ProcessingJob, ProcessingJobStatus, ProcessingJobType,
    ProcessingJobUpdate, ReindexItem,
};
pub use models::{
    ConversationSearchResult, MessageSearchResult, RelatedNotes, SearchResult, SimilarNote,
};
pub use portable::PORTABLE_TABLES;
pub use stats::DbStats;

use crate::{
    compatibility::{
        check_embedding_compatibility, record_embedding_metadata, CompatibilityState,
        EmbeddingIdentity, ExtractionIdentity,
    },
    fusion::{self, FusionConfig, FusionEvidence, FusionRecord},
    migrations, DbConnection, DbError, Result,
};
use chrono::{DateTime, Utc};
use graphrag_core::{
    record_id_to_string, ChatConversation, ChatMessage, EdgeType, Entity, Note, ProposedEdge,
    ProposedEdgeStatus, Source, SourceIngestionStatus, SourceType,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};
use surrealdb::types::RecordId;
use surrealdb_types::SurrealValue;
use tokio::sync::Mutex;
use tracing::instrument;
use uuid::Uuid;

/// Repository for all database operations
#[derive(Clone)]
pub struct Repository {
    db: DbConnection,
    proposal_acceptance_lock: Arc<Mutex<()>>,
}

// A source generation becomes visible only after promotion. Legacy/manual
// notes have no generation and remain visible, while staged and superseded
// file-import notes are excluded from every user-facing scan.
const VISIBLE_NOTE_CONDITION: &str = "(source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)";

// Edge rows contain record references in `in` and `out`. Resolve both note
// endpoints through the note table before showing graph topology so an
// interrupted import cannot expose relationships owned by a staged or
// superseded generation.
const VISIBLE_NOTE_EDGE_ENDPOINTS_CONDITION: &str = "in IN (SELECT VALUE id FROM note WHERE (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation)) AND out IN (SELECT VALUE id FROM note WHERE (source_id IS NONE OR source_generation IS NONE OR source_generation = source_id.successful_generation))";

fn graph_query_normalize(query: &str) -> String {
    // Entity canonicalization retains arbitrary Unicode text. Split only at
    // lexical boundaries so adjacent punctuation (for example `Atlas?`) does
    // not become part of a term, while CJK and other non-Latin letters remain
    // intact rather than being treated as ASCII-only words.
    Entity::canonicalize(query)
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn graph_prefix_terms(normalized_query: &str) -> Vec<String> {
    normalized_query
        .split_whitespace()
        .filter(|term| term.chars().count() >= 4 && !is_graph_prefix_stop_word(term))
        .map(str::to_string)
        .collect()
}

fn is_graph_prefix_stop_word(term: &str) -> bool {
    matches!(
        term,
        "a" | "an"
            | "and"
            | "are"
            | "as"
            | "at"
            | "be"
            | "by"
            | "can"
            | "changed"
            | "change"
            | "could"
            | "did"
            | "does"
            | "for"
            | "from"
            | "had"
            | "has"
            | "have"
            | "how"
            | "in"
            | "is"
            | "it"
            | "of"
            | "on"
            | "or"
            | "recent"
            | "show"
            | "tell"
            | "that"
            | "the"
            | "this"
            | "to"
            | "was"
            | "what"
            | "when"
            | "where"
            | "which"
            | "who"
            | "why"
            | "will"
            | "with"
            | "would"
            | "you"
    )
}

/// Match tiers for local graph-entity seeding. Keeping exact equality ahead
/// of contained phrases means a specific entity query cannot be crowded out
/// by a shorter entity name when the caller supplies a small seed cap.
#[derive(Clone, Copy)]
enum GraphEntityMatchTier {
    Exact,
    ContainedPhrase,
    Prefix,
}

fn count_to_i64(count: u64) -> Result<i64> {
    i64::try_from(count)
        .map_err(|_| DbError::QueryFailed("processing count exceeds database integer range".into()))
}

/// Derive the default full-text value from a note's displayed content and its
/// Markdown heading metadata.
fn derived_search_content(note: &Note) -> String {
    let headings = note
        .chunk_heading_path
        .iter()
        .filter(|heading| !heading.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" > ");
    if headings.is_empty() {
        note.content.clone()
    } else {
        format!("{headings}\n\n{}", note.content)
    }
}

/// Resolve search text for a note update without treating every existing
/// value as derived. A caller may intentionally supply aliases or other
/// custom searchable text; keep those for metadata-only updates and explicit
/// replacements. Rebuild only when the persisted value was the old derived
/// Markdown/body value carried through a content or heading-context edit.
fn search_content_for_note_update(existing: &Note, replacement: &Note) -> String {
    let existing_derived = derived_search_content(existing);
    let existing_search = existing
        .search_content
        .as_deref()
        .unwrap_or(existing_derived.as_str());
    if let Some(replacement_search) = replacement.search_content.as_deref() {
        if replacement_search != existing_search {
            return replacement_search.to_string();
        }
    }

    let source_changed = existing.content != replacement.content
        || existing.chunk_heading_path != replacement.chunk_heading_path;
    if source_changed && existing_search == existing_derived {
        derived_search_content(replacement)
    } else {
        existing_search.to_string()
    }
}

fn source_content_value(source: &Source) -> Result<serde_json::Value> {
    let mut value = serde_json::to_value(source)
        .map_err(|error| DbError::QueryFailed(format!("source serialization failed: {error}")))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| DbError::QueryFailed("source did not serialize as an object".into()))?;
    object.remove("id");
    object.remove("created_at");
    object.remove("updated_at");
    object.remove("last_ingested_at");
    for key in [
        "uri",
        "content",
        "normalized_uri",
        "content_hash",
        "last_error",
    ] {
        if object.get(key).is_some_and(serde_json::Value::is_null) {
            object.remove(key);
        }
    }
    if object
        .get("metadata")
        .is_some_and(serde_json::Value::is_null)
    {
        object.remove("metadata");
    }
    Ok(value)
}

impl Repository {
    /// Create a new repository
    pub fn new(db: DbConnection) -> Self {
        let proposal_acceptance_lock = db.proposal_lifecycle_lock();
        Self {
            db,
            proposal_acceptance_lock,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;
    use graphrag_core::{EntityType, SourceType};

    async fn begin_markdown(repo: &Repository, content: &str, force: bool) -> SourceImportPlan {
        repo.begin_file_import(
            SourceType::Markdown,
            "alpha.md".into(),
            "file:///notes/alpha.md".into(),
            content.into(),
            format!("sha256:{content}"),
            force,
        )
        .await
        .unwrap()
    }

    async fn create_provenance_records(repo: &Repository) -> (RecordId, RecordId) {
        #[derive(Deserialize, SurrealValue)]
        struct IdRow {
            id: RecordId,
        }

        let conversation: Option<IdRow> = repo
            .db
            .query(
                "CREATE conversation SET uuid = $uuid, title = 'test conversation', summary = NONE, source_uri = NONE, account_uuid = NONE, metadata = {}, summary_embedding = NONE, created_at = time::now(), updated_at = time::now() RETURN AFTER",
            )
            .bind(("uuid", "test-conversation"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let conversation_id = conversation.unwrap().id;
        let message: Option<IdRow> = repo
            .db
            .query(
                "CREATE message SET message_key = $message_key, message_uuid = NONE, conversation_id = $conversation_id, conversation_uuid = $conversation_uuid, message_index = 0, role = 'human', content = 'test message', embedding = NONE, content_blocks = [], attachments = [], files = [], has_files = false, created_at = NONE, updated_at = NONE RETURN AFTER",
            )
            .bind(("message_key", "test-message"))
            .bind(("conversation_id", conversation_id.clone()))
            .bind(("conversation_uuid", "test-conversation"))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        (conversation_id, message.unwrap().id)
    }

    #[tokio::test]
    async fn note_list_source_uri_filter_projects_linked_source_uri() {
        let repo = Repository::new(init_memory().await.unwrap());
        let first_source = repo
            .create_source(Source::from_file("first.md", SourceType::Markdown).unwrap())
            .await
            .unwrap();
        let second_source = repo
            .create_source(Source::from_file("second.md", SourceType::Markdown).unwrap())
            .await
            .unwrap();
        let first = repo
            .create_note(Note::new("first").with_source(first_source.id.clone().unwrap()))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_source(second_source.id.clone().unwrap()))
            .await
            .unwrap();

        let first_uri = first_source.uri.as_deref().unwrap();
        let notes = repo
            .list_notes_filtered(10, &[], Some(first_uri))
            .await
            .unwrap();
        assert_eq!(notes.len(), 1);
        assert_eq!(notes[0].id, first.id.unwrap());
        assert_eq!(notes[0].source_uri.as_deref(), Some(first_uri));
    }

    #[test]
    fn graph_entity_query_normalization_keeps_unicode_terms_and_guards_prefixes() {
        assert_eq!(graph_query_normalize("Where is Atlas?"), "where is atlas");
        assert_eq!(graph_query_normalize("東京？"), "東京");
        assert_eq!(
            graph_prefix_terms("what changed in atla"),
            vec!["atla".to_string()]
        );
        assert!(graph_prefix_terms("ai").is_empty());
    }

    #[tokio::test]
    async fn graph_entity_lookup_normalizes_internal_punctuation_for_names_and_aliases() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut canonical = Entity::new("GPT-4", EntityType::Technology);
        canonical.metadata = serde_json::json!({});
        let canonical = repo.upsert_entity(canonical).await.unwrap();
        let mut aliased = Entity::new("Model reference", EntityType::Technology);
        aliased.metadata = serde_json::json!({"aliases": ["GPT-4"]});
        let aliased = repo.upsert_entity(aliased).await.unwrap();

        let matches = repo
            .find_graph_entities("Where is GPT-4?", 10)
            .await
            .unwrap();
        assert!(matches
            .iter()
            .any(|entity| entity.id == *canonical.id.as_ref().unwrap()));
        assert!(matches
            .iter()
            .any(|entity| entity.id == *aliased.id.as_ref().unwrap()));
    }

    #[tokio::test]
    async fn duplicate_entity_upsert_merges_aliases_without_losing_metadata() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut original = Entity::new("Atlas service", EntityType::Project);
        original.metadata = serde_json::json!({"aliases": ["Atlas"]});
        let original = repo.upsert_entity(original).await.unwrap();

        let mut update = Entity::new("Atlas Service", EntityType::Project);
        update.metadata = serde_json::json!({"aliases": ["Atlas", "Atlas v2"]});
        let updated = repo.upsert_entity(update).await.unwrap();

        assert_eq!(updated.id, original.id);
        assert_eq!(
            updated.metadata["aliases"],
            serde_json::json!(["Atlas", "Atlas v2"])
        );

        let original_matches = repo.find_graph_entities("Atlas", 1).await.unwrap();
        assert_eq!(original_matches.len(), 1);
        assert_eq!(original_matches[0].id, *original.id.as_ref().unwrap());
        let matches = repo.find_graph_entities("Atlas v2", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *original.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn graph_prefix_lookup_prefers_the_likely_entity_fragment_over_context_words() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut zeta = Entity::new("Zeta archive", EntityType::Project);
        zeta.metadata = serde_json::json!({});
        let zeta = repo.upsert_entity(zeta).await.unwrap();
        let mut deployed = Entity::new("Deployed controller", EntityType::Project);
        deployed.metadata = serde_json::json!({});
        repo.upsert_entity(deployed).await.unwrap();

        // Neither multi-word canonical name is contained in the sentence, so
        // this uses prefix recovery. The four-character entity fragment must
        // win before the eight-character context word at a one-entity cap.
        let matches = repo.find_graph_entities("zeta deployed", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *zeta.id.as_ref().unwrap());

        let mut atla = Entity::new("Atlas archive", EntityType::Project);
        atla.metadata = serde_json::json!({});
        repo.upsert_entity(atla).await.unwrap();
        let atla_matches = repo
            .find_graph_entities("atla deployment", 1)
            .await
            .unwrap();
        assert_eq!(atla_matches.len(), 1);
        assert_eq!(atla_matches[0].name, "Atlas archive");
    }

    #[tokio::test]
    async fn graph_entity_lookup_prioritizes_whole_query_names_and_aliases_over_phrases() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut short_name = Entity::new("New", EntityType::Project);
        short_name.metadata = serde_json::json!({});
        repo.upsert_entity(short_name).await.unwrap();
        let mut exact_name = Entity::new("New York", EntityType::Project);
        exact_name.metadata = serde_json::json!({});
        let exact_name = repo.upsert_entity(exact_name).await.unwrap();

        let matches = repo.find_graph_entities("New York", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_name.id.as_ref().unwrap());

        let matches = repo
            .find_graph_entities("status New York today", 1)
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_name.id.as_ref().unwrap());

        let mut short_alias = Entity::new("Big", EntityType::Project);
        short_alias.metadata = serde_json::json!({});
        repo.upsert_entity(short_alias).await.unwrap();
        let mut exact_alias = Entity::new("New York City", EntityType::Project);
        exact_alias.metadata = serde_json::json!({"aliases": ["Big Apple"]});
        let exact_alias = repo.upsert_entity(exact_alias).await.unwrap();

        let matches = repo.find_graph_entities("Big Apple", 1).await.unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_alias.id.as_ref().unwrap());

        let matches = repo
            .find_graph_entities("status Big Apple today", 1)
            .await
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].id, *exact_alias.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn graph_note_seed_query_preserves_ranked_entity_coverage_under_cap() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut atlas = Entity::new("Atlas", EntityType::Project);
        atlas.metadata = serde_json::json!({});
        let atlas = repo.upsert_entity(atlas).await.unwrap();
        let mut beacon = Entity::new("Beacon", EntityType::Project);
        beacon.metadata = serde_json::json!({});
        let beacon = repo.upsert_entity(beacon).await.unwrap();

        for id in ["atlas_a", "atlas_b", "beacon_a"] {
            repo.db
                .query(format!(
                    "CREATE note:{id} SET note_type = 'raw', content = $content, embedding = NONE, tags = [], created_at = time::now(), updated_at = time::now()"
                ))
                .bind(("content", id.to_string()))
                .await
                .unwrap()
                .check()
                .unwrap();
        }
        for id in ["atlas_a", "atlas_b"] {
            repo.link_note_to_entity(&RecordId::new("note", id), atlas.id.as_ref().unwrap())
                .await
                .unwrap();
        }
        repo.link_note_to_entity(
            &RecordId::new("note", "beacon_a"),
            beacon.id.as_ref().unwrap(),
        )
        .await
        .unwrap();

        let seeds = repo
            .graph_notes_for_entities(
                &[
                    atlas.id.as_ref().unwrap().clone(),
                    beacon.id.as_ref().unwrap().clone(),
                ],
                1,
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0].entity_id, *atlas.id.as_ref().unwrap());
        assert_eq!(seeds[1].entity_id, *beacon.id.as_ref().unwrap());
    }

    #[tokio::test]
    async fn source_lifecycle_is_idempotent_and_preserves_manual_notes() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        assert_eq!(first.action, SourceImportAction::Created);
        let source_id = first.source.id.as_ref().unwrap().clone();
        let derived = repo
            .create_note(
                Note::new("derived first")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual association").with_source(source_id.clone()))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let unchanged = begin_markdown(&repo, "first", false).await;
        assert_eq!(unchanged.action, SourceImportAction::Unchanged);
        assert_eq!(unchanged.source.generation, 1);

        let mut changed = begin_markdown(&repo, "second", false).await;
        assert_eq!(changed.action, SourceImportAction::Updated);
        assert_eq!(changed.source.generation, 2);
        let current = repo
            .create_note(
                Note::new("derived second")
                    .with_source(source_id.clone())
                    .with_source_generation(changed.source.generation),
            )
            .await
            .unwrap();
        let cleanup = repo
            .complete_file_import(&mut changed.source)
            .await
            .unwrap();
        assert_eq!(cleanup.notes, 1);
        assert!(repo
            .get_note(&record_id_to_string(derived.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(current.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
        assert!(repo
            .get_note(&record_id_to_string(manual.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());

        let mut failed = begin_markdown(&repo, "third", true).await;
        let partial = repo
            .create_note(
                Note::new("partial")
                    .with_source(source_id.clone())
                    .with_source_generation(failed.source.generation),
            )
            .await
            .unwrap();
        repo.fail_file_import(&mut failed.source, "embedding unavailable")
            .await
            .unwrap();
        let stored = repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.status, SourceIngestionStatus::Failed);
        assert_eq!(stored.successful_generation, 2);
        assert!(repo
            .get_note(&record_id_to_string(partial.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(current.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn retrieval_hides_unpromoted_source_generations_after_interruption() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        repo.create_note(
            Note::new("visible generation one")
                .with_embedding(vec![0.0; 1024])
                .with_source(source_id.clone())
                .with_source_generation(first.source.generation),
        )
        .await
        .unwrap();

        // The first process can be interrupted before promotion. Its staged
        // note must not be returned by either retrieval path after restart.
        assert!(repo
            .fulltext_search("visible generation", 10)
            .await
            .unwrap()
            .is_empty());
        assert!(repo
            .vector_search(vec![0.0; 1024], 10)
            .await
            .unwrap()
            .is_empty());

        repo.complete_file_import(&mut first.source).await.unwrap();
        assert_eq!(
            repo.fulltext_search("visible generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            repo.vector_search(vec![0.0; 1024], 10).await.unwrap().len(),
            1
        );

        let second = begin_markdown(&repo, "second", false).await;
        assert_eq!(second.source.generation, 2);
        repo.create_note(
            Note::new("pending generation two")
                .with_embedding(vec![0.0; 1024])
                .with_source(source_id)
                .with_source_generation(second.source.generation),
        )
        .await
        .unwrap();
        assert!(repo
            .fulltext_search("pending generation", 10)
            .await
            .unwrap()
            .is_empty());
        assert_eq!(
            repo.fulltext_search("visible generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn related_notes_hide_staged_source_generations() {
        let repo = Repository::new(init_memory().await.unwrap());
        let anchor = repo.create_note(Note::new("manual anchor")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        let anchor_id = anchor.id.as_ref().unwrap();
        let staged_id = staged.id.as_ref().unwrap();
        // Cover both graph directions for every relationship projection.
        for edge_type in [
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::RelatedTo,
        ] {
            repo.create_edge(anchor_id, staged_id, edge_type.clone(), None)
                .await
                .unwrap();
            repo.create_edge(staged_id, anchor_id, edge_type, None)
                .await
                .unwrap();
        }

        let related = repo.get_related_notes(anchor_id).await.unwrap();
        assert!(related.supporting.is_empty());
        assert!(related.supported_by.is_empty());
        assert!(related.contradicting.is_empty());
        assert!(related.contradicted_by.is_empty());
        assert!(related.related.is_empty());
        assert!(related.related_from.is_empty());
    }

    #[tokio::test]
    async fn orphan_notes_ignore_hidden_source_generation_neighbors() {
        let repo = Repository::new(init_memory().await.unwrap());
        let anchor = repo.create_note(Note::new("manual anchor")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        let anchor_id = anchor.id.as_ref().unwrap();
        let staged_id = staged.id.as_ref().unwrap();
        // A staged source generation is invisible in every graph direction,
        // so it cannot prevent a visible manual note from being an orphan.
        for edge_type in [
            EdgeType::Supports,
            EdgeType::Contradicts,
            EdgeType::RelatedTo,
        ] {
            repo.create_edge(anchor_id, staged_id, edge_type.clone(), None)
                .await
                .unwrap();
            repo.create_edge(staged_id, anchor_id, edge_type, None)
                .await
                .unwrap();
        }

        let orphans = repo.find_orphan_notes().await.unwrap();
        assert_eq!(orphans.len(), 1);
        assert_eq!(orphans[0].id.as_ref(), anchor.id.as_ref());
    }

    #[tokio::test]
    async fn note_edge_lists_hide_edges_with_hidden_source_generation_endpoints() {
        let repo = Repository::new(init_memory().await.unwrap());
        let manual_left = repo.create_note(Note::new("manual left")).await.unwrap();
        let manual_right = repo.create_note(Note::new("manual right")).await.unwrap();
        let plan = begin_markdown(&repo, "staged", false).await;
        let staged = repo
            .create_note(
                Note::new("staged source note")
                    .with_source(plan.source.id.as_ref().unwrap().clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        repo.create_edge(
            manual_left.id.as_ref().unwrap(),
            manual_right.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();
        repo.create_edge(
            manual_left.id.as_ref().unwrap(),
            staged.id.as_ref().unwrap(),
            EdgeType::Supports,
            None,
        )
        .await
        .unwrap();

        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
        assert_eq!(
            repo.get_note_edges(&record_id_to_string(manual_left.id.as_ref().unwrap()))
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(repo
            .get_note_edges(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn source_owned_notes_are_created_and_updated_with_ownership_intact() {
        let repo = Repository::new(init_memory().await.unwrap());
        let plan = begin_markdown(&repo, "content", false).await;
        let source_id = plan.source.id.as_ref().unwrap().clone();
        let created = repo
            .create_note(
                Note::new("owned content")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();

        // Creation writes both ownership fields in the single CREATE command;
        // there is no unowned persisted state to leak if import work stops.
        assert_eq!(created.source_id.as_ref(), Some(&source_id));
        assert_eq!(created.source_generation, Some(plan.source.generation));

        let mut edited = created.clone();
        edited.content = "edited content".into();
        // Callers that do not repeat source ownership must not accidentally
        // detach a source-owned note during a content update.
        edited.source_id = None;
        edited.source_generation = None;
        let updated = repo
            .update_note(&record_id_to_string(created.id.as_ref().unwrap()), edited)
            .await
            .unwrap();
        assert_eq!(updated.source_id.as_ref(), Some(&source_id));
        assert_eq!(updated.source_generation, Some(plan.source.generation));
    }

    #[tokio::test]
    async fn update_note_rebuilds_markdown_search_content_without_stale_terms() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut markdown_note = Note::new("obsolete-search-token is removed");
        markdown_note.chunk_heading_path = vec!["Roadmap".into()];
        markdown_note.search_content = Some("Roadmap\n\nobsolete-search-token is removed".into());
        let created = repo.create_note(markdown_note).await.unwrap();
        let created_id = created.id.as_ref().unwrap().clone();

        let mut edited = created.clone();
        edited.content = "current-search-token is indexed".into();
        // Model a caller changing only `content`; this stale field used to
        // keep removed body terms in the highest-weight FTS column.
        edited.search_content = created.search_content.clone();
        let updated = repo
            .update_note(&record_id_to_string(&created_id), edited)
            .await
            .unwrap();

        assert_eq!(
            updated.search_content.as_deref(),
            Some("Roadmap\n\ncurrent-search-token is indexed")
        );
        assert!(repo
            .fulltext_search("obsolete-search-token", 10)
            .await
            .unwrap()
            .iter()
            .all(|result| result.id != created_id));
        let current = repo
            .fulltext_search("current-search-token", 10)
            .await
            .unwrap();
        assert!(current
            .iter()
            .any(|result| { result.id == created_id && result.fts_score.is_some() }));
    }

    #[tokio::test]
    async fn update_note_preserves_custom_search_aliases_and_replacements() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut note = Note::new("original body text");
        note.search_content = Some("legacyalias".into());
        let created = repo.create_note(note).await.unwrap();
        let note_id = created.id.as_ref().unwrap().clone();

        // A metadata-only update must not discard an intentional alias just
        // because it does not equal the default body-derived search value.
        let mut metadata_only = created.clone();
        metadata_only.title = Some("Reference note".into());
        let metadata_updated = repo
            .update_note(&record_id_to_string(&note_id), metadata_only)
            .await
            .unwrap();
        assert_eq!(
            metadata_updated.search_content.as_deref(),
            Some("legacyalias")
        );
        assert_eq!(metadata_updated.title.as_deref(), Some("Reference note"));
        assert!(repo
            .fulltext_search("legacyalias", 10)
            .await
            .unwrap()
            .iter()
            .any(|result| result.id == note_id));

        // When callers intentionally replace search text alongside a content
        // edit, retain that replacement instead of overriding it with the
        // generic body-derived value.
        let mut content_edit = metadata_updated;
        content_edit.content = "replacement body text".into();
        content_edit.search_content = Some("replacementalias".into());
        let replaced = repo
            .update_note(&record_id_to_string(&note_id), content_edit)
            .await
            .unwrap();
        assert_eq!(replaced.search_content.as_deref(), Some("replacementalias"));
        assert!(repo
            .fulltext_search("legacyalias", 10)
            .await
            .unwrap()
            .iter()
            .all(|result| result.id != note_id));
        assert!(repo
            .fulltext_search("replacementalias", 10)
            .await
            .unwrap()
            .iter()
            .any(|result| result.id == note_id));
    }

    #[tokio::test]
    async fn atomic_note_update_replaces_mentions_with_the_new_content() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo.create_note(Note::new("old body")).await.unwrap();
        let note_id = record_id_to_string(note.id.as_ref().unwrap());
        let mut old_entity = Entity::new("Old Entity", EntityType::Concept);
        old_entity.metadata = serde_json::json!({});
        let old_entity = repo.upsert_entity(old_entity).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), old_entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let mut replacement = note.clone();
        replacement.content = "new body".into();
        let mut new_entity = Entity::new("New Entity", EntityType::Concept);
        new_entity.metadata = serde_json::json!({});
        let updated = repo
            .update_note_and_replace_entities(&note_id, replacement, vec![new_entity])
            .await
            .unwrap();

        assert_eq!(updated.content, "new body");
        let linked = repo.get_entities_for_note(&note_id).await.unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "New Entity");
    }

    #[tokio::test]
    async fn atomic_note_update_rolls_back_before_content_change_on_entity_error() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo.create_note(Note::new("old body")).await.unwrap();
        let note_id = record_id_to_string(note.id.as_ref().unwrap());
        let mut old_entity = Entity::new("Prior Entity", EntityType::Concept);
        old_entity.metadata = serde_json::json!({});
        let old_entity = repo.upsert_entity(old_entity).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), old_entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let mut replacement = note.clone();
        replacement.content = "new body must not persist".into();
        let mut malformed = Entity::new("Malformed Entity", EntityType::Concept);
        malformed.metadata = serde_json::json!("not an object");
        assert!(repo
            .update_note_and_replace_entities(&note_id, replacement, vec![malformed])
            .await
            .is_err());

        assert_eq!(
            repo.get_note(&note_id).await.unwrap().unwrap().content,
            "old body"
        );
        let linked = repo.get_entities_for_note(&note_id).await.unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Prior Entity");
    }

    #[tokio::test]
    async fn atomic_note_create_with_mentions_rolls_back_and_retries_without_duplicates() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut malformed = Entity::new("Malformed Entity", EntityType::Concept);
        malformed.metadata = serde_json::json!("not an object");
        assert!(repo
            .create_note_and_replace_entities(Note::new("detached copy"), vec![malformed])
            .await
            .is_err());
        assert!(repo.list_notes(10).await.unwrap().is_empty());

        let mut entity = Entity::new("Copied Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let created = repo
            .create_note_and_replace_entities(Note::new("detached copy"), vec![entity])
            .await
            .unwrap();
        assert_eq!(repo.list_notes(10).await.unwrap().len(), 1);
        let linked = repo
            .get_entities_for_note(&record_id_to_string(created.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Copied Entity");
    }

    #[tokio::test]
    async fn promotion_selects_the_new_generation_before_old_cleanup() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let first_note = repo
            .create_note(
                Note::new("first generation")
                    .with_embedding(vec![1.0; 1024])
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let old_generation_partner = repo
            .create_note(
                Note::new("first generation partner")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_generation_proposal = repo
            .upsert_gardener_proposal(
                first_note.id.as_ref().unwrap(),
                old_generation_partner.id.as_ref().unwrap(),
                0.9,
                "old generation appears related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let old_generation_proposal_id = old_generation_proposal.id.unwrap();
        let old_generation_accepted_partner = repo
            .create_note(
                Note::new("first generation accepted partner")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_generation_accepted_proposal = repo
            .upsert_gardener_proposal(
                first_note.id.as_ref().unwrap(),
                old_generation_accepted_partner.id.as_ref().unwrap(),
                0.9,
                "old generation accepted relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let old_generation_accepted_proposal_id = old_generation_accepted_proposal.id.unwrap();
        let old_generation_accepted_edge_id = repo
            .accept_edge_proposal(
                &old_generation_accepted_proposal_id,
                Some("reviewer".into()),
                Some("approved before reimport".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let second_note = repo
            .create_note(
                Note::new("second generation")
                    .with_embedding(vec![1.0; 1024])
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        // Simulate a process stopping after durable promotion but before the
        // best-effort destructive cleanup. The new complete generation is
        // immediately searchable; the old one is merely hidden/recoverable.
        repo.promote_file_import(&mut second.source).await.unwrap();
        assert_eq!(
            second.source.successful_generation,
            second.source.generation
        );
        assert_eq!(
            repo.get_edge_proposal(&old_generation_proposal_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
        assert_eq!(
            repo.fulltext_search("second generation", 10)
                .await
                .unwrap()
                .len(),
            1
        );
        assert!(repo
            .fulltext_search("first generation", 10)
            .await
            .unwrap()
            .is_empty());
        // Other unfiltered scans must honor the same visibility rule while
        // cleanup is deferred after an interruption.
        assert_eq!(repo.list_notes(10).await.unwrap().len(), 1);
        assert_eq!(repo.find_orphan_notes().await.unwrap().len(), 1);
        assert_eq!(repo.get_notes_page(10, 0).await.unwrap().len(), 1);
        let second_key = record_id_to_string(second_note.id.as_ref().unwrap())
            .strip_prefix("note:")
            .unwrap()
            .to_string();
        assert!(repo
            .find_similar_notes(&second_key, vec![1.0; 1024], 0.0, 10)
            .await
            .unwrap()
            .is_empty());

        // The unchanged-hash path doubles as durable recovery: it retries
        // cleanup instead of leaving hidden old generations forever.
        let retry = begin_markdown(&repo, "second", false).await;
        assert_eq!(retry.action, SourceImportAction::Unchanged);
        assert_eq!(retry.cleanup.notes, 3);
        assert_eq!(retry.cleanup.proposals, 1);
        assert!(repo
            .get_note(&record_id_to_string(first_note.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        let accepted_proposal = repo
            .get_edge_proposal(&old_generation_accepted_proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(accepted_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted_proposal.resulting_edge_id, None);
        assert!(!repo
            .note_edge_exists(&old_generation_accepted_edge_id)
            .await
            .unwrap());
    }

    #[tokio::test]
    async fn promotion_serializes_hidden_generation_retirement_with_acceptance() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "old generation race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();

        let second = begin_markdown(&repo, "second", false).await;
        // Queue promotion ahead of acceptance while the shared lock is held.
        // Tokio's mutex queues waiters fairly, so acceptance must observe the
        // proposal retirement performed at the visibility boundary.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let promotion_repo = repo.clone();
        let mut second_source = second.source;
        let promotion =
            tokio::spawn(
                async move { promotion_repo.promote_file_import(&mut second_source).await },
            );
        tokio::task::yield_now().await;
        let acceptance_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let acceptance = tokio::spawn(async move {
            acceptance_repo
                .accept_edge_proposal(&accepting_id, Some("reviewer".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        promotion.await.unwrap().unwrap();
        assert!(acceptance.await.unwrap().is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn completion_serializes_proposal_retarget_before_old_edge_undo() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "retarget race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let old_edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let new_left = repo
            .create_note(
                Note::new("second generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let new_right = repo
            .create_note(
                Note::new("second generation right")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.copy_note_dependents_to_successors(&[
            (
                old_left.id.as_ref().unwrap().clone(),
                new_left.id.as_ref().unwrap().clone(),
                true,
            ),
            (
                old_right.id.as_ref().unwrap().clone(),
                new_right.id.as_ref().unwrap().clone(),
                true,
            ),
        ])
        .await
        .unwrap();

        // Queue complete before undo while the lifecycle lock is held. The
        // completion transition must promote, retarget the accepted proposal
        // to the staged edge, and retire the old edge before undo may act.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let completion_repo = repo.clone();
        let completion = tokio::spawn(async move {
            completion_repo
                .complete_file_import(&mut second.source)
                .await
        });
        tokio::task::yield_now().await;
        let undo_repo = repo.clone();
        let undo_edge_id = old_edge_id.clone();
        let undo = tokio::spawn(async move {
            undo_repo
                .undo_edge(&undo_edge_id, Some("concurrent undo".into()))
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        completion.await.unwrap().unwrap();
        assert!(!undo.await.unwrap().unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Accepted);
        let replacement_edge_id = proposal.resulting_edge_id.unwrap();
        assert_ne!(replacement_edge_id, old_edge_id);
        assert!(repo.note_edge_exists(&replacement_edge_id).await.unwrap());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_graph_writes_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // Queue reconciliation before a manual graph write. The shared lock
        // makes the write observe the finished cleanup, where its old endpoint
        // is absent, instead of allowing it to succeed and be silently swept
        // away after the dependent snapshot.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let mutation_repo = repo.clone();
        let mutation_old_id = old.id.as_ref().unwrap().clone();
        let mutation_manual_id = manual.id.as_ref().unwrap().clone();
        let mutation = tokio::spawn(async move {
            mutation_repo
                .create_edge(
                    &mutation_old_id,
                    &mutation_manual_id,
                    EdgeType::Supports,
                    Some(0.9),
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(mutation.await.unwrap().is_err());
        assert!(repo
            .get_note(&record_id_to_string(old.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_mentions_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let mut entity = Entity::new("Concurrent Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // Queue reconciliation before the mention write. The write must see
        // that the old endpoint is gone after cleanup rather than succeeding
        // in the snapshot/cleanup window and being silently discarded.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let mention_repo = repo.clone();
        let old_note_id = old.id.as_ref().unwrap().clone();
        let entity_id = entity.id.as_ref().unwrap().clone();
        let mention = tokio::spawn(async move {
            mention_repo
                .link_note_to_entity(&old_note_id, &entity_id)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(mention.await.unwrap().is_err());
        assert!(repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn reconciliation_snapshots_a_complete_batched_entity_extraction() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let entities = ["First Extracted Entity", "Second Extracted Entity"]
            .into_iter()
            .map(|name| {
                let mut entity = Entity::new(name, EntityType::Concept);
                entity.metadata = serde_json::json!({});
                entity
            })
            .collect();

        // Queue the entire extraction result before reconciliation. The
        // shared lifecycle lock makes reconciliation snapshot both links,
        // rather than seeing an arbitrary prefix between per-entity writes.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let extraction_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let extraction = tokio::spawn(async move {
            extraction_repo
                .upsert_entities_and_link_note(&old_id, entities)
                .await
        });
        tokio::task::yield_now().await;
        let reconciliation_repo = repo.clone();
        let reconcile_old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(
                    &mut second.source,
                    &[(reconcile_old_id, replacement_id, true)],
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert_eq!(extraction.await.unwrap().unwrap(), 2);
        reconciliation.await.unwrap().unwrap();
        let entities = repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(entities.len(), 2);
        assert!(entities
            .iter()
            .any(|entity| entity.name == "First Extracted Entity"));
        assert!(entities
            .iter()
            .any(|entity| entity.name == "Second Extracted Entity"));
    }

    #[tokio::test]
    async fn failed_entity_link_batch_rolls_back_only_its_partial_mentions() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("entity batch rollback target"))
            .await
            .unwrap();
        let mut existing = Entity::new("Preexisting link survives", EntityType::Concept);
        existing.metadata = serde_json::json!({});
        let existing = repo.upsert_entity(existing).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), existing.id.as_ref().unwrap())
            .await
            .unwrap();
        let mut valid = Entity::new("Link must be rolled back", EntityType::Concept);
        valid.metadata = serde_json::json!({});
        let mut invalid = Entity::new("Entity write fails after first link", EntityType::Concept);
        // The entity schema requires an object or NONE. This reliably fails
        // the second upsert after the first link has been staged.
        invalid.metadata = serde_json::json!("not an object");

        assert!(repo
            .upsert_entities_and_link_note(note.id.as_ref().unwrap(), vec![valid, invalid])
            .await
            .is_err());
        let linked = repo
            .get_entities_for_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Preexisting link survives");
    }

    #[tokio::test]
    async fn failed_entity_replacement_preserves_the_prior_complete_mention_set() {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("entity replacement rollback target"))
            .await
            .unwrap();
        let mut existing = Entity::new("Prior extraction survives", EntityType::Concept);
        existing.metadata = serde_json::json!({});
        let existing = repo.upsert_entity(existing).await.unwrap();
        repo.link_note_to_entity(note.id.as_ref().unwrap(), existing.id.as_ref().unwrap())
            .await
            .unwrap();
        let mut invalid = Entity::new("Malformed replacement", EntityType::Concept);
        invalid.metadata = serde_json::json!("not an object");

        assert!(repo
            .replace_note_entities(note.id.as_ref().unwrap(), vec![invalid])
            .await
            .is_err());
        let linked = repo
            .get_entities_for_note(&record_id_to_string(note.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 1);
        assert_eq!(linked[0].name, "Prior extraction survives");
    }

    #[tokio::test]
    async fn reconciliation_snapshots_a_complete_replaced_entity_set() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let mut stale = Entity::new("Stale extraction", EntityType::Concept);
        stale.metadata = serde_json::json!({});
        let stale = repo.upsert_entity(stale).await.unwrap();
        repo.link_note_to_entity(old.id.as_ref().unwrap(), stale.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let entities = ["Fresh extraction one", "Fresh extraction two"]
            .into_iter()
            .map(|name| {
                let mut entity = Entity::new(name, EntityType::Concept);
                entity.metadata = serde_json::json!({});
                entity
            })
            .collect();

        // Inference has completed before this point. Queue the atomic mention
        // replacement ahead of reconciliation to prove the source transition
        // copies the full fresh set, never a transient cleared set.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let replacement_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let refresh = tokio::spawn(async move {
            replacement_repo
                .replace_note_entities(&old_id, entities)
                .await
        });
        tokio::task::yield_now().await;
        let reconciliation_repo = repo.clone();
        let reconcile_old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(
                    &mut second.source,
                    &[(reconcile_old_id, replacement_id, true)],
                )
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert_eq!(refresh.await.unwrap().unwrap(), 2);
        reconciliation.await.unwrap().unwrap();
        let linked = repo
            .get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
            .await
            .unwrap();
        assert_eq!(linked.len(), 2);
        assert!(linked
            .iter()
            .any(|entity| entity.name == "Fresh extraction one"));
        assert!(linked
            .iter()
            .any(|entity| entity.name == "Fresh extraction two"));
    }

    #[tokio::test]
    async fn reconciliation_serializes_provenance_writes_and_copies_them_without_deadlock() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let (conversation_id, message_id) = create_provenance_records(&repo).await;
        repo.link_note_to_conversation(old.id.as_ref().unwrap(), &conversation_id)
            .await
            .unwrap();
        repo.link_note_to_message(old.id.as_ref().unwrap(), &message_id)
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        // This also exercises the lock-free internal helpers used while the
        // reconciliation task already owns the lifecycle lock.
        repo.reconcile_file_import(
            &mut second.source,
            &[(
                old.id.as_ref().unwrap().clone(),
                replacement.id.as_ref().unwrap().clone(),
                true,
            )],
        )
        .await
        .unwrap();
        assert!(repo
            .conversation_has_note_links(&conversation_id)
            .await
            .unwrap());
        let copied_messages: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(copied_messages, vec![message_id]);

        // The now-hidden old endpoint must reject later provenance writes;
        // this prevents a link from being inserted after the copy snapshot
        // and then lost during cleanup.
        assert!(repo
            .link_note_to_conversation(old.id.as_ref().unwrap(), &conversation_id)
            .await
            .is_err());
        assert!(repo
            .link_note_to_message(old.id.as_ref().unwrap(), &copied_messages[0])
            .await
            .is_err());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_provenance_writes_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let (conversation_id, message_id) = create_provenance_records(&repo).await;

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let conversation_repo = repo.clone();
        let conversation_note_id = old.id.as_ref().unwrap().clone();
        let conversation_link = tokio::spawn(async move {
            conversation_repo
                .link_note_to_conversation(&conversation_note_id, &conversation_id)
                .await
        });
        tokio::task::yield_now().await;
        let message_repo = repo.clone();
        let message_note_id = old.id.as_ref().unwrap().clone();
        let message_link = tokio::spawn(async move {
            message_repo
                .link_note_to_message(&message_note_id, &message_id)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(conversation_link.await.unwrap().is_err());
        assert!(message_link.await.unwrap().is_err());
        let copied_conversations: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_conversation WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let copied_messages: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE out FROM note_from_message WHERE in = $note_id")
            .bind(("note_id", replacement.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(copied_conversations.is_empty());
        assert!(copied_messages.is_empty());
    }

    #[tokio::test]
    async fn reconciliation_prevents_post_snapshot_mention_removals_from_being_lost() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("first generation chunk")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let mut entity = Entity::new("Copied Entity", EntityType::Concept);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        repo.link_note_to_entity(old.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let replacement = repo
            .create_note(
                Note::new("second generation chunk")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();

        let guard = repo.proposal_acceptance_lock.lock().await;
        let reconciliation_repo = repo.clone();
        let old_id = old.id.as_ref().unwrap().clone();
        let replacement_id = replacement.id.as_ref().unwrap().clone();
        let reconciliation = tokio::spawn(async move {
            reconciliation_repo
                .reconcile_file_import(&mut second.source, &[(old_id, replacement_id, true)])
                .await
        });
        tokio::task::yield_now().await;
        let removal_repo = repo.clone();
        let old_note_id = old.id.as_ref().unwrap().clone();
        let removal =
            tokio::spawn(async move { removal_repo.delete_mentions_for_note(&old_note_id).await });
        tokio::task::yield_now().await;
        drop(guard);

        reconciliation.await.unwrap().unwrap();
        assert!(removal.await.unwrap().is_err());
        assert_eq!(
            repo.get_entities_for_note(&record_id_to_string(replacement.id.as_ref().unwrap()))
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn graph_writes_reject_hidden_generations_but_allow_current_pending() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old = repo
            .create_note(
                Note::new("old generation")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let current_pending = repo
            .create_note(
                Note::new("current pending generation")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.create_edge(
            current_pending.id.as_ref().unwrap(),
            manual.id.as_ref().unwrap(),
            EdgeType::Supports,
            Some(0.9),
        )
        .await
        .unwrap();

        // Simulate a crash after durable promotion and before physical old
        // generation cleanup. The old note remains stored but is hidden.
        repo.promote_file_import(&mut second.source).await.unwrap();
        assert!(repo
            .create_edge(
                old.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                EdgeType::RelatedTo,
                Some(0.9),
            )
            .await
            .is_err());
        repo.create_edge(
            current_pending.id.as_ref().unwrap(),
            manual.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.9),
        )
        .await
        .unwrap();
    }

    #[tokio::test]
    async fn recovery_drops_copied_edge_when_its_proposal_was_undone() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "undo before retarget".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let old_edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let mut second = begin_markdown(&repo, "second", false).await;
        let new_left = repo
            .create_note(
                Note::new("second generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        let new_right = repo
            .create_note(
                Note::new("second generation right")
                    .with_source(source_id)
                    .with_source_generation(second.source.generation),
            )
            .await
            .unwrap();
        repo.copy_note_dependents_to_successors(&[
            (
                old_left.id.as_ref().unwrap().clone(),
                new_left.id.as_ref().unwrap().clone(),
                true,
            ),
            (
                old_right.id.as_ref().unwrap().clone(),
                new_right.id.as_ref().unwrap().clone(),
                true,
            ),
        ])
        .await
        .unwrap();
        let copied_edges: Vec<RecordId> = repo
            .db
            .query("SELECT VALUE id FROM related_to WHERE in = $note OR out = $note")
            .bind(("note", new_left.id.as_ref().unwrap().clone()))
            .await
            .unwrap()
            .take(0)
            .unwrap();
        let copied_edge_id = copied_edges.into_iter().next().unwrap();

        // Reconstruct a crash window from older callers that copied staged
        // dependents before completion: the original edge was undone before
        // the copied edge could retarget the proposal audit.
        assert!(repo
            .undo_edge(&old_edge_id, Some("undone before retarget".into()))
            .await
            .unwrap());
        repo.complete_file_import(&mut second.source).await.unwrap();

        assert!(!repo.note_edge_exists(&copied_edge_id).await.unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn post_promotion_failure_keeps_new_generation_for_recovery() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut import = begin_markdown(&repo, "durably promoted", false).await;
        let source_id = import.source.id.as_ref().unwrap().clone();
        let staged = repo
            .create_note(
                Note::new("new generation remains visible")
                    .with_source(source_id.clone())
                    .with_source_generation(import.source.generation),
            )
            .await
            .unwrap();
        let manual = repo
            .create_note(Note::new("manual endpoint"))
            .await
            .unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                staged.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                0.9,
                "force a post-promotion retarget failure".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let (edge_id, _) = repo
            .create_audited_edge(
                staged.id.as_ref().unwrap(),
                manual.id.as_ref().unwrap(),
                EdgeType::RelatedTo,
                Some(0.9),
                Some("staged proposal-backed edge"),
                "test",
                Some(&proposal_id),
                false,
            )
            .await
            .unwrap();

        // The proposal is deliberately still pending, so retargeting fails
        // after `replace_source` durably promoted this generation.
        assert!(repo.complete_file_import(&mut import.source).await.is_err());
        let stored = repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.status, SourceIngestionStatus::Ready);
        assert_eq!(stored.successful_generation, stored.generation);
        assert!(repo
            .get_note(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());

        // Repair the injected inconsistency, then exercise the unchanged-hash
        // recovery path. It retries retargeting/cleanup without deleting the
        // already promoted current generation.
        repo.db
            .query(
                "UPDATE $proposal SET status = 'accepted', resulting_edge_id = $edge, updated_at = time::now()",
            )
            .bind(("proposal", proposal_id.clone()))
            .bind(("edge", edge_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let recovered = begin_markdown(&repo, "durably promoted", false).await;
        assert_eq!(recovered.action, SourceImportAction::Unchanged);
        assert!(repo.note_edge_exists(&edge_id).await.unwrap());
        assert!(repo
            .get_note(&record_id_to_string(staged.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
    }

    #[tokio::test]
    async fn interrupted_promotion_cannot_resume_an_acceptance_for_hidden_notes() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut first = begin_markdown(&repo, "first", false).await;
        let source_id = first.source.id.as_ref().unwrap().clone();
        let old_left = repo
            .create_note(
                Note::new("first generation left")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        let old_right = repo
            .create_note(
                Note::new("first generation right")
                    .with_source(source_id.clone())
                    .with_source_generation(first.source.generation),
            )
            .await
            .unwrap();
        repo.complete_file_import(&mut first.source).await.unwrap();
        let proposal_id = repo
            .upsert_gardener_proposal(
                old_left.id.as_ref().unwrap(),
                old_right.id.as_ref().unwrap(),
                0.9,
                "interrupted promotion race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        // Persist an acceptance claim, then simulate a different process
        // being interrupted after promotion updates source visibility but
        // before it can retire old-generation proposals.
        assert!(repo
            .claim_pending_proposal(
                &proposal_id,
                ProposedEdgeStatus::Accepting,
                Some("reviewer".into()),
                Some("approved before interruption".into()),
                Some(true),
            )
            .await
            .unwrap());
        let mut second = begin_markdown(&repo, "second", false).await;
        second.source.successful_generation = second.source.generation;
        second.source.status = SourceIngestionStatus::Ready;
        second.source.last_error = None;
        second.source.updated_at = chrono::Utc::now();
        second.source.last_ingested_at = Some(second.source.updated_at);
        repo.replace_source(&second.source).await.unwrap();

        // Resuming must treat now-hidden endpoints as stale, rather than
        // materializing an edge during the interruption window.
        assert!(repo
            .accept_edge_proposal(&proposal_id, Some("retry".into()), None, true)
            .await
            .is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("proposal endpoint is no longer visible")
        );
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());

        // The normal unchanged import recovery then deletes the hidden old
        // generation without finding a dangling accepted edge/proposal.
        let recovery = begin_markdown(&repo, "second", false).await;
        assert_eq!(recovery.action, SourceImportAction::Unchanged);
        assert_eq!(recovery.cleanup.notes, 2);
        assert!(repo
            .get_note(&record_id_to_string(old_left.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
    }

    #[tokio::test]
    async fn source_delete_preview_matches_confirmed_derived_cascade() {
        let repo = Repository::new(init_memory().await.unwrap());
        let mut plan = begin_markdown(&repo, "content", false).await;
        let source_id = plan.source.id.as_ref().unwrap().clone();
        // Garden acceptance only operates on notes in the active source
        // generation, so promote this source before creating its fixture
        // relationships.
        repo.complete_file_import(&mut plan.source).await.unwrap();
        let derived = repo
            .create_note(
                Note::new("derived")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();
        let derived_second = repo
            .create_note(
                Note::new("derived second")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();
        let unrelated = repo.create_note(Note::new("manual")).await.unwrap();
        // This internal edge is reachable through two source-owned notes but
        // must count once in the exact dry-run/delete summary.
        repo.create_edge(
            derived.id.as_ref().unwrap(),
            derived_second.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.5),
        )
        .await
        .unwrap();
        repo.create_edge(
            derived.id.as_ref().unwrap(),
            unrelated.id.as_ref().unwrap(),
            EdgeType::RelatedTo,
            Some(0.5),
        )
        .await
        .unwrap();
        let proposal = repo
            .upsert_gardener_proposal(
                derived.id.as_ref().unwrap(),
                unrelated.id.as_ref().unwrap(),
                0.9,
                "source-derived note looks related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let accepted_proposal = repo
            .upsert_gardener_proposal(
                derived_second.id.as_ref().unwrap(),
                unrelated.id.as_ref().unwrap(),
                0.9,
                "accepted source-derived note looks related".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let accepted_proposal_id = accepted_proposal.id.unwrap();
        let accepted_edge_id = repo
            .accept_edge_proposal(
                &accepted_proposal_id,
                Some("reviewer".into()),
                Some("approved before source removal".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();
        let mut retained_entity = Entity::new("Retained entity", EntityType::Concept);
        retained_entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(retained_entity).await.unwrap();
        repo.link_note_to_entity(derived.id.as_ref().unwrap(), entity.id.as_ref().unwrap())
            .await
            .unwrap();

        let preview = repo.preview_source_delete(&plan.source).await.unwrap();
        assert_eq!(preview.notes, 2);
        assert_eq!(preview.mentions, 1);
        assert_eq!(preview.note_edges, 3);
        assert_eq!(preview.proposals, 2);
        let confirmed = repo.delete_source(&plan.source).await.unwrap();
        assert_eq!(confirmed, preview);
        assert!(repo
            .get_note(&record_id_to_string(derived.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_none());
        assert!(repo
            .get_note(&record_id_to_string(unrelated.id.as_ref().unwrap()))
            .await
            .unwrap()
            .is_some());
        assert!(repo
            .get_source(&record_id_to_string(&source_id))
            .await
            .unwrap()
            .is_none());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("proposal endpoint removed by source lifecycle")
        );
        let accepted_proposal = repo
            .get_edge_proposal(&accepted_proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(accepted_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted_proposal.resulting_edge_id, None);
        assert!(!repo.note_edge_exists(&accepted_edge_id).await.unwrap());
        // Source cleanup must retire the pending proposal before policy batch
        // acceptance sees it, rather than failing on its missing endpoint.
        assert_eq!(
            repo.accept_gardener_proposals_above(0.8, Some("policy".into()))
                .await
                .unwrap(),
            0
        );
        let mut retained_entity = Entity::new("Retained entity", EntityType::Concept);
        retained_entity.metadata = serde_json::json!({});
        assert!(repo.upsert_entity(retained_entity).await.is_ok());
    }

    #[tokio::test]
    async fn test_create_and_get_note() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let note = Note::new("Test content").with_title("Test Title");
        let created = repo.create_note(note).await.unwrap();

        assert!(created.id.is_some());
        assert_eq!(created.content, "Test content");
    }

    #[tokio::test]
    async fn test_list_notes() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        // Create a few notes
        for i in 0..3 {
            let note = Note::new(format!("Content {}", i));
            repo.create_note(note).await.unwrap();
        }

        let notes = repo.list_notes(10).await.unwrap();
        assert_eq!(notes.len(), 3);
    }

    async fn two_notes(repo: &Repository) -> (RecordId, RecordId) {
        let first = repo
            .create_note(Note::new("first"))
            .await
            .unwrap()
            .id
            .unwrap();
        let second = repo
            .create_note(Note::new("second"))
            .await
            .unwrap()
            .id
            .unwrap();
        (first, second)
    }

    #[tokio::test]
    async fn gardener_proposals_are_canonical_and_idempotent() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;

        let original = repo
            .upsert_gardener_proposal(
                &second,
                &first,
                0.81,
                "similar notes".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let updated = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.93,
                "newer similarity".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();

        assert_eq!(original.id, updated.id);
        assert_eq!(updated.confidence, 0.93);
        assert_eq!(updated.reason, "newer similarity");
        assert_eq!(updated.status, ProposedEdgeStatus::Pending);
        assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
        assert!(record_id_to_string(&updated.from_id) < record_id_to_string(&updated.to_id));
        let proposal_id = updated.id.as_ref().unwrap();
        assert_eq!(
            parse_record_id(&record_id_to_string(proposal_id), Some("proposed_edge")).unwrap(),
            *proposal_id
        );
    }

    #[tokio::test]
    async fn proposal_refresh_preserves_terminal_decision_audit() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.8,
                "original scan".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let rejected = repo
            .reject_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("not appropriate".into()),
            )
            .await
            .unwrap();
        let refreshed = repo
            .upsert_gardener_proposal(
                &second,
                &first,
                0.99,
                "later scan must not overwrite the decision".into(),
                Some("new-test".into()),
                None,
            )
            .await
            .unwrap();
        assert_eq!(refreshed.id, Some(proposal_id));
        assert_eq!(refreshed.status, ProposedEdgeStatus::Rejected);
        assert_eq!(refreshed.reason, "original scan");
        assert_eq!(refreshed.reviewer, rejected.reviewer);
        assert_eq!(refreshed.action_reason, rejected.action_reason);
    }

    #[tokio::test]
    async fn concurrent_proposal_upserts_reload_the_unique_index_winner() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let left_repo = repo.clone();
        let right_repo = repo.clone();
        let left_from = first.clone();
        let left_to = second.clone();
        let right_from = second;
        let right_to = first;

        let (left, right) = tokio::join!(
            left_repo.upsert_gardener_proposal(
                &left_from,
                &left_to,
                0.81,
                "left scan".into(),
                Some("test".into()),
                None,
            ),
            right_repo.upsert_gardener_proposal(
                &right_from,
                &right_to,
                0.93,
                "right scan".into(),
                Some("test".into()),
                None,
            ),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.id, right.id);
        assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn interrupted_acceptance_claims_are_recoverable_by_retry_and_batch() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let retry = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "retry after interruption".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let retry_id = retry.id.unwrap();
        // Simulate a process stopping after its durable acceptance claim but
        // before edge creation/finalization.
        repo.db
            .query(
                "UPDATE $id SET status = 'accepting', reviewer = 'first reviewer', action_reason = 'first decision', acceptance_is_manual = true, reviewed_at = time::now()",
            )
            .bind(("id", retry_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        let recovered = repo
            .accept_edge_proposal(
                &retry_id,
                Some("retry reviewer".into()),
                Some("retry decision".into()),
                false,
            )
            .await
            .unwrap();
        assert_eq!(recovered.status, ProposedEdgeStatus::Accepted);
        assert_eq!(recovered.action_reason.as_deref(), Some("first decision"));
        let retry_edge_id = recovered.resulting_edge_id.unwrap();
        assert!(repo.note_edge_exists(&retry_edge_id).await.unwrap());
        assert!(repo.list_note_edges(10).await.unwrap()[0].is_manual);

        let third = repo
            .create_note(Note::new("third"))
            .await
            .unwrap()
            .id
            .unwrap();
        let batch = repo
            .upsert_gardener_proposal(
                &first,
                &third,
                0.9,
                "batch recovery".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let batch_id = batch.id.unwrap();
        // Simulate the legacy poisoned shape produced before recovery support:
        // accepted without a resulting edge id. Batch completion must repair it.
        repo.db
            .query(
                "UPDATE $id SET status = 'accepted', reviewer = 'policy', action_reason = 'policy decision', acceptance_is_manual = false, resulting_edge_id = NONE, reviewed_at = time::now()",
            )
            .bind(("id", batch_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();
        assert_eq!(
            repo.accept_gardener_proposals_above(0.8, Some("policy retry".into()))
                .await
                .unwrap(),
            1
        );
        let batch = repo.get_edge_proposal(&batch_id).await.unwrap().unwrap();
        assert_eq!(batch.status, ProposedEdgeStatus::Accepted);
        assert!(batch.resulting_edge_id.is_some());
        assert_eq!(batch.action_reason.as_deref(), Some("policy decision"));
    }

    #[tokio::test]
    async fn concurrent_accepts_share_one_stable_edge_and_completion() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "same proposal".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let left_repo = repo.clone();
        let right_repo = repo.clone();
        let left_id = proposal_id.clone();
        let right_id = proposal_id.clone();

        let (left, right) = tokio::join!(
            left_repo.accept_edge_proposal(
                &left_id,
                Some("left reviewer".into()),
                Some("left acceptance".into()),
                true,
            ),
            right_repo.accept_edge_proposal(
                &right_id,
                Some("right reviewer".into()),
                Some("right acceptance".into()),
                false,
            ),
        );
        let left = left.unwrap();
        let right = right.unwrap();
        assert_eq!(left.status, ProposedEdgeStatus::Accepted);
        assert_eq!(left.resulting_edge_id, right.resulting_edge_id);
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Accepted);
        assert!(proposal.resulting_edge_id.is_some());
    }

    #[tokio::test]
    async fn acceptance_never_adopts_an_independent_manual_edge() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        repo.create_edge(&first, &second, EdgeType::RelatedTo, Some(0.7))
            .await
            .unwrap();
        let manual_edge = repo.list_note_edges(10).await.unwrap().pop().unwrap();
        assert_eq!(manual_edge.provenance.as_deref(), Some("manual_api"));

        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "would duplicate manual edge".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        assert!(repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .is_err());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("equivalent edge already materialized independently")
        );
        assert!(repo.note_edge_exists(&manual_edge.id).await.unwrap());
    }

    #[tokio::test]
    async fn endpoint_deletion_and_acceptance_leave_no_dangling_edge() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (removed, retained) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &removed,
                &retained,
                0.9,
                "race with deletion".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();

        // Queue both operations behind the shared lifecycle lock, then let
        // them race for it. Either ordering is valid, but the final graph may
        // not retain an edge to the removed endpoint.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let accepting_repo = repo.clone();
        let deleting_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let removed_id = record_id_to_string(&removed);
        let acceptance = async move {
            accepting_repo
                .accept_edge_proposal(&accepting_id, Some("reviewer".into()), None, true)
                .await
        };
        let deletion = async move { deleting_repo.delete_note(&removed_id).await };
        drop(guard);
        let (_acceptance, deletion) = tokio::join!(acceptance, deletion);
        deletion.unwrap();

        assert!(repo
            .get_note(&record_id_to_string(&removed))
            .await
            .unwrap()
            .is_none());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn proposal_accept_reject_and_undo_are_auditable_and_idempotent() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let accepted = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "semantic overlap".into(),
                Some("test".into()),
                Some("fixture".into()),
            )
            .await
            .unwrap();
        let accepted_id = accepted.id.unwrap();
        let accepted = repo
            .accept_edge_proposal(
                &accepted_id,
                Some("reviewer".into()),
                Some("looks related".into()),
                true,
            )
            .await
            .unwrap();
        assert_eq!(accepted.status, ProposedEdgeStatus::Accepted);
        let accepted_reviewed_at = accepted.reviewed_at;
        assert_eq!(accepted.action_reason.as_deref(), Some("looks related"));
        let edge_id = accepted.resulting_edge_id.clone().unwrap();
        assert_eq!(
            repo.accept_edge_proposal(&accepted_id, None, None, true)
                .await
                .unwrap()
                .resulting_edge_id,
            Some(edge_id.clone())
        );

        let edge = repo.list_note_edges(10).await.unwrap().pop().unwrap();
        assert_eq!(edge.id, edge_id);
        assert_eq!(edge.reason.as_deref(), Some("semantic overlap"));
        assert_eq!(edge.provenance.as_deref(), Some("gardener-similarity"));
        assert!(edge.is_manual);
        assert!(repo.note_edge_exists(&edge_id).await.unwrap());

        assert!(repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        assert!(!repo.note_edge_exists(&edge_id).await.unwrap());
        assert!(!repo
            .undo_edge(&edge_id, Some("reversed".into()))
            .await
            .unwrap());
        let undone_proposal = repo.get_edge_proposal(&accepted_id).await.unwrap().unwrap();
        assert_eq!(undone_proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(undone_proposal.resulting_edge_id, None);
        assert_eq!(
            undone_proposal.action_reason.as_deref(),
            Some("looks related")
        );
        assert_eq!(undone_proposal.reviewed_at, accepted_reviewed_at);
        assert_eq!(
            undone_proposal.supersession_reason.as_deref(),
            Some("reversed")
        );
        assert!(undone_proposal.superseded_at.is_some());

        let rejected = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "same pair after undo stays terminal".into(),
                None,
                None,
            )
            .await
            .unwrap();
        assert_eq!(rejected.status, ProposedEdgeStatus::Superseded);
    }

    #[tokio::test]
    async fn undo_repairs_a_proposal_after_a_prior_edge_only_delete() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "accepted then interrupted undo".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let edge_id = repo
            .accept_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("accepted".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();
        // Simulate an interruption after undo's physical deletion but before
        // it could supersede the proposal record.
        repo.db
            .query("DELETE $id")
            .bind(("id", edge_id.clone()))
            .await
            .unwrap()
            .check()
            .unwrap();

        assert!(repo
            .undo_edge(&edge_id, Some("recovered undo".into()))
            .await
            .unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
        assert_eq!(
            proposal.supersession_reason.as_deref(),
            Some("recovered undo")
        );
        assert!(!repo.undo_edge(&edge_id, None).await.unwrap());
    }

    #[tokio::test]
    async fn undo_serializes_with_acceptance_completion() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal_id = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "undo and retry race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let edge_id = repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        // Queue undo before a concurrent acceptance retry. Without the shared
        // lock, the retry can read `accepted` before undo retires its audit
        // record and race finalization against edge deletion.
        let guard = repo.proposal_acceptance_lock.lock().await;
        let undo_repo = repo.clone();
        let undo_id = edge_id.clone();
        let undo = tokio::spawn(async move { undo_repo.undo_edge(&undo_id, None).await });
        tokio::task::yield_now().await;
        let acceptance_repo = repo.clone();
        let accepting_id = proposal_id.clone();
        let acceptance = tokio::spawn(async move {
            acceptance_repo
                .accept_edge_proposal(&accepting_id, Some("retry".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert!(undo.await.unwrap().unwrap());
        assert!(acceptance.await.unwrap().is_err());
        assert!(!repo.note_edge_exists(&edge_id).await.unwrap());
        let proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn independently_constructed_repositories_share_lifecycle_serialization() {
        let db = init_memory().await.unwrap();
        let accepting_repo = Repository::new(db.clone());
        let undoing_repo = Repository::new(db);
        assert!(Arc::ptr_eq(
            &accepting_repo.proposal_acceptance_lock,
            &undoing_repo.proposal_acceptance_lock
        ));
        let (first, second) = two_notes(&accepting_repo).await;
        let proposal_id = accepting_repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "independent repository race".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap()
            .id
            .unwrap();
        let edge_id = accepting_repo
            .accept_edge_proposal(&proposal_id, Some("reviewer".into()), None, true)
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        // Queue the competing operations through separate Repository values.
        // Undo wins the shared lifecycle lock, so a later acceptance retry
        // must see a superseded proposal rather than recreate/finalize it.
        let guard = accepting_repo.proposal_acceptance_lock.lock().await;
        let undo_repo = undoing_repo.clone();
        let undo_id = edge_id.clone();
        let undo = tokio::spawn(async move { undo_repo.undo_edge(&undo_id, None).await });
        tokio::task::yield_now().await;
        let retry_repo = accepting_repo.clone();
        let retry_id = proposal_id.clone();
        let retry = tokio::spawn(async move {
            retry_repo
                .accept_edge_proposal(&retry_id, Some("retry".into()), None, true)
                .await
        });
        tokio::task::yield_now().await;
        drop(guard);

        assert!(undo.await.unwrap().unwrap());
        assert!(retry.await.unwrap().is_err());
        assert!(!undoing_repo.note_edge_exists(&edge_id).await.unwrap());
        let proposal = undoing_repo
            .get_edge_proposal(&proposal_id)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(proposal.status, ProposedEdgeStatus::Superseded);
        assert_eq!(proposal.resulting_edge_id, None);
    }

    #[tokio::test]
    async fn independent_memory_stores_do_not_share_lifecycle_serialization() {
        let first_repo = Repository::new(init_memory().await.unwrap());
        let second_repo = Repository::new(init_memory().await.unwrap());
        assert!(!Arc::ptr_eq(
            &first_repo.proposal_acceptance_lock,
            &second_repo.proposal_acceptance_lock
        ));

        // Holding a lifecycle transition for one logical store must not block
        // a repository backed by a separately initialized in-memory store.
        let first_guard = first_repo.proposal_acceptance_lock.lock().await;
        let second_guard = second_repo
            .proposal_acceptance_lock
            .try_lock()
            .expect("independent store lifecycle lock is not held");
        drop(second_guard);
        drop(first_guard);
    }

    #[tokio::test]
    async fn proposal_reject_is_idempotent_and_stale_endpoints_are_superseded() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(&first, &second, 0.8, "similar".into(), None, None)
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();
        let rejected = repo
            .reject_edge_proposal(
                &proposal_id,
                Some("reviewer".into()),
                Some("not useful".into()),
            )
            .await
            .unwrap();
        assert_eq!(rejected.status, ProposedEdgeStatus::Rejected);
        assert_eq!(
            repo.reject_edge_proposal(&proposal_id, None, None)
                .await
                .unwrap()
                .status,
            ProposedEdgeStatus::Rejected
        );

        let (third, fourth) = two_notes(&repo).await;
        let stale = repo
            .upsert_gardener_proposal(&third, &fourth, 0.8, "similar".into(), None, None)
            .await
            .unwrap();
        let stale_id = stale.id.unwrap();
        let _: Option<Note> = repo.db.delete(fourth).await.unwrap();
        assert!(repo
            .accept_edge_proposal(&stale_id, None, None, true)
            .await
            .is_err());
        assert_eq!(
            repo.get_edge_proposal(&stale_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
    }

    #[tokio::test]
    async fn delete_note_retires_proposals_and_their_accepted_edges() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (removed, pending_target) = two_notes(&repo).await;
        let accepted_target = repo
            .create_note(Note::new("accepted target"))
            .await
            .unwrap()
            .id
            .unwrap();
        let pending = repo
            .upsert_gardener_proposal(
                &removed,
                &pending_target,
                0.8,
                "pending relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let pending_id = pending.id.unwrap();
        let accepted = repo
            .upsert_gardener_proposal(
                &removed,
                &accepted_target,
                0.9,
                "accepted relationship".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let accepted_id = accepted.id.unwrap();
        let accepted_edge_id = repo
            .accept_edge_proposal(
                &accepted_id,
                Some("reviewer".into()),
                Some("approved".into()),
                true,
            )
            .await
            .unwrap()
            .resulting_edge_id
            .unwrap();

        let removed_id = record_id_to_string(&removed);
        let preview = repo.preview_note_delete(&removed_id).await.unwrap();
        assert_eq!(preview.notes, 1);
        assert_eq!(preview.note_edges, 1);
        assert_eq!(preview.proposals, 2);
        let deleted = repo.delete_note_with_summary(&removed_id).await.unwrap();
        assert_eq!(deleted, preview);

        assert!(repo.get_note(&removed_id).await.unwrap().is_none());
        assert_eq!(
            repo.get_edge_proposal(&pending_id)
                .await
                .unwrap()
                .unwrap()
                .status,
            ProposedEdgeStatus::Superseded
        );
        let accepted = repo.get_edge_proposal(&accepted_id).await.unwrap().unwrap();
        assert_eq!(accepted.status, ProposedEdgeStatus::Superseded);
        assert_eq!(accepted.resulting_edge_id, None);
        assert!(!repo.note_edge_exists(&accepted_edge_id).await.unwrap());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn competing_accept_and_reject_claim_exactly_one_terminal_decision() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let proposal = repo
            .upsert_gardener_proposal(
                &first,
                &second,
                0.9,
                "similarly scoped notes".into(),
                Some("test".into()),
                None,
            )
            .await
            .unwrap();
        let proposal_id = proposal.id.unwrap();

        let (accept, reject) = tokio::join!(
            repo.accept_edge_proposal(
                &proposal_id,
                Some("acceptor".into()),
                Some("accept decision".into()),
                true,
            ),
            repo.reject_edge_proposal(
                &proposal_id,
                Some("rejector".into()),
                Some("reject decision".into()),
            ),
        );
        assert_ne!(accept.is_ok(), reject.is_ok());

        let final_proposal = repo.get_edge_proposal(&proposal_id).await.unwrap().unwrap();
        match final_proposal.status {
            ProposedEdgeStatus::Accepted => {
                assert!(final_proposal.resulting_edge_id.is_some());
                assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);
            }
            ProposedEdgeStatus::Rejected => {
                assert!(final_proposal.resulting_edge_id.is_none());
                assert!(repo.list_note_edges(10).await.unwrap().is_empty());
            }
            status => panic!("unexpected terminal status: {status}"),
        }
    }

    #[tokio::test]
    async fn bulk_gardener_acceptance_pages_every_match_and_propagates_failures() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        let third = repo
            .create_note(Note::new("third"))
            .await
            .unwrap()
            .id
            .unwrap();
        for (from, to) in [(&first, &second), (&first, &third), (&second, &third)] {
            repo.upsert_gardener_proposal(from, to, 0.9, "similar".into(), None, None)
                .await
                .unwrap();
        }
        assert_eq!(
            repo.accept_gardener_proposals_above_in_pages(
                0.8,
                Some("cli batch acceptance".into()),
                "reviewed as a related note".into(),
                true,
                1,
            )
            .await
            .unwrap(),
            3
        );
        assert_eq!(
            repo.list_edge_proposals(Some(ProposedEdgeStatus::Accepted), 10)
                .await
                .unwrap()
                .len(),
            3
        );
        let accepted = repo
            .list_edge_proposals(Some(ProposedEdgeStatus::Accepted), 10)
            .await
            .unwrap();
        assert!(accepted.iter().all(|proposal| {
            proposal.action_reason.as_deref() == Some("reviewed as a related note")
        }));
        assert!(repo
            .list_note_edges(10)
            .await
            .unwrap()
            .iter()
            .all(|edge| edge.is_manual));

        let failing_repo = Repository::new(init_memory().await.unwrap());
        let (from, stale_endpoint) = two_notes(&failing_repo).await;
        failing_repo
            .upsert_gardener_proposal(&from, &stale_endpoint, 0.9, "similar".into(), None, None)
            .await
            .unwrap();
        let _: Option<Note> = failing_repo.db.delete(stale_endpoint).await.unwrap();
        assert!(failing_repo
            .accept_gardener_proposals_above_in_pages(
                0.8,
                None,
                "automatic policy".into(),
                false,
                1,
            )
            .await
            .is_err());
    }

    #[tokio::test]
    async fn self_edges_and_reverse_symmetric_duplicates_are_rejected() {
        let repo = Repository::new(init_memory().await.unwrap());
        let (first, second) = two_notes(&repo).await;
        assert!(repo
            .upsert_gardener_proposal(&first, &first, 0.8, "self".into(), None, None)
            .await
            .is_err());

        repo.create_edge(&first, &second, EdgeType::RelatedTo, Some(0.8))
            .await
            .unwrap();
        repo.create_edge(&second, &first, EdgeType::RelatedTo, Some(0.8))
            .await
            .unwrap();
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 1);

        repo.create_edge(&first, &second, EdgeType::Supports, Some(0.8))
            .await
            .unwrap();
        repo.create_edge(&second, &first, EdgeType::Supports, Some(0.8))
            .await
            .unwrap();
        assert_eq!(repo.list_note_edges(10).await.unwrap().len(), 3);
    }

    #[tokio::test]
    async fn similar_note_search_excludes_only_the_query_note() {
        let repo = Repository::new(init_memory().await.unwrap());
        let embedding = vec![1.0; 1024];
        let first = repo
            .create_note(Note::new("first").with_embedding(embedding.clone()))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_embedding(embedding.clone()))
            .await
            .unwrap();
        let similar = repo
            .find_similar_notes(
                &record_id_to_string(first.id.as_ref().unwrap()),
                embedding,
                0.7,
                5,
            )
            .await
            .unwrap();
        assert_eq!(similar.len(), 1);
    }

    #[tokio::test]
    async fn get_entities_for_note_uses_a_bound_record_id() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let note = repo
            .create_note(Note::new("Entity-linked note"))
            .await
            .unwrap();
        let note_id = note.id.unwrap();
        let mut entity = Entity::new("SurrealDB", EntityType::Technology);
        entity.metadata = serde_json::json!({});
        let entity = repo.upsert_entity(entity).await.unwrap();
        let entity_id = entity.id.unwrap();
        repo.link_note_to_entity(&note_id, &entity_id)
            .await
            .unwrap();

        let note_key = record_id_to_string(&note_id)
            .strip_prefix("note:")
            .unwrap()
            .to_string();

        for note_reference in [note_key.clone(), format!("note:{note_key}")] {
            let entities = repo.get_entities_for_note(&note_reference).await.unwrap();

            assert_eq!(entities.len(), 1);
            assert_eq!(entities[0].id.as_ref(), Some(&entity_id));
        }
    }

    #[tokio::test]
    async fn test_hybrid_search_small_limit_keeps_relevant_note() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db);

        let rust_embedding = vec![1.0_f32; 1024];
        let distractor_embedding = vec![0.05_f32; 1024];

        let rust_note = Note::new("Rust is memory-safe and fast")
            .with_title("Rust note")
            .with_embedding(rust_embedding.clone());
        repo.create_note(rust_note).await.unwrap();

        for i in 0..60 {
            let note = Note::new(format!("Distractor content {}", i))
                .with_title(format!("Distractor {}", i))
                .with_embedding(distractor_embedding.clone());
            repo.create_note(note).await.unwrap();
        }

        let results = repo
            .hybrid_search_notes("Rust note", rust_embedding, 3, None, None)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        assert_eq!(results[0].title.as_deref(), Some("Rust note"));
    }

    #[tokio::test]
    async fn candidate_cutoffs_break_equal_component_scores_by_record_id() {
        let db = init_memory().await.unwrap();
        let repo = Repository::new(db.clone());
        let embedding = vec![1.0_f32; 1024];
        for id in ["zulu", "alpha"] {
            let note = Note::new("identical deterministic candidate")
                .with_title("identical deterministic candidate")
                .with_embedding(embedding.clone());
            let _: Option<Note> = db.create(("note", id)).content(note).await.unwrap();
        }

        let vector = repo
            .vector_search_notes(embedding, 1, None, None)
            .await
            .unwrap();
        let fulltext = repo
            .fulltext_search_notes("identical deterministic candidate", 1, None, None)
            .await
            .unwrap();

        assert_eq!(record_id_to_string(&vector[0].id), "note:alpha");
        assert_eq!(record_id_to_string(&fulltext[0].id), "note:alpha");
    }

    #[tokio::test]
    async fn processing_job_checkpoint_cancel_and_resume_are_durable() {
        let repo = Repository::new(init_memory().await.unwrap());
        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::Embedding,
                Some("source:7/2".into()),
                3,
                Some("missing_embeddings".into()),
                vec!["note:one".into(), "note:two".into(), "note:three".into()],
            )
            .await
            .unwrap();
        let id = job.id.clone().unwrap();
        assert_eq!(job.scope.as_deref(), Some("missing_embeddings"));
        assert_eq!(job.item_ids.len(), 3);
        let updated = repo
            .update_processing_job(
                &id,
                ProcessingJobUpdate {
                    completed_count: Some(1),
                    checkpoint: Some(Some("note:one".into())),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(updated.completed_count, 1);
        assert_eq!(updated.checkpoint.as_deref(), Some("note:one"));
        let cancelled = repo.cancel_processing_job(&id).await.unwrap();
        assert_eq!(cancelled.status, ProcessingJobStatus::Cancelled.as_str());
        let resumed = repo.resume_processing_job(&id).await.unwrap();
        assert_eq!(resumed.status, ProcessingJobStatus::Running.as_str());
        assert_eq!(repo.list_processing_jobs(10).await.unwrap().len(), 1);
    }
}
