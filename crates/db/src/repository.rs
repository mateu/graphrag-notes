//! Repository pattern for database operations

use crate::{DbConnection, Result, DbError};
use graphrag_core::{Note, Entity, Source, EdgeType};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;
use tracing::instrument;

/// Repository for all database operations
#[derive(Clone)]
pub struct Repository {
    db: DbConnection,
}

impl Repository {
    /// Create a new repository
    pub fn new(db: DbConnection) -> Self {
        Self { db }
    }
    
    // ==========================================
    // NOTE OPERATIONS
    // ==========================================
    
    /// Create a new note
    #[instrument(skip(self, note))]
    pub async fn create_note(&self, note: Note) -> Result<Note> {
        // Use SurrealDB's high-level create API so we get back the stored
        // record including its generated `id`.
        let created: Option<Note> = self.db
            .create("note")
            .content(note)
            .await?;

        created.ok_or_else(|| DbError::QueryFailed("create_note".into()))
    }
    
    /// Get a note by ID
    #[instrument(skip(self))]
    pub async fn get_note(&self, id: &str) -> Result<Option<Note>> {
        let note: Option<Note> = self.db.select(("note", id)).await?;
        Ok(note)
    }
    
    /// Update a note
    #[instrument(skip(self, note))]
    pub async fn update_note(&self, id: &str, note: Note) -> Result<Note> {
        let updated: Option<Note> = self.db
            .update(("note", id))
            .content(note)
            .await?;
        
        updated.ok_or_else(|| DbError::NotFound("note".into(), id.into()))
    }
    
    /// Delete a note
    #[instrument(skip(self))]
    pub async fn delete_note(&self, id: &str) -> Result<()> {
        let _: Option<Note> = self.db.delete(("note", id)).await?;
        Ok(())
    }
    
    /// List recent notes (basic fields only, for CLI)
    #[instrument(skip(self))]
    pub async fn list_notes(&self, limit: usize) -> Result<Vec<SearchResult>> {
        let mut notes: Vec<SearchResult> = self.db
            .select("note")
            .await?;

        // Sort by creation time descending and apply limit in Rust to avoid
        // SurrealDB multi-result `take` issues and deserialization problems
        // with full `Note` records.
        notes.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        if notes.len() > limit {
            notes.truncate(limit);
        }

        Ok(notes)
    }
    
    /// Get notes without embeddings (for processing)
    #[instrument(skip(self))]
    pub async fn get_notes_without_embeddings(&self) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self.db
            .query("SELECT * FROM note WHERE embedding IS NONE OR array::len(embedding) = 0")
            .await?
            .take(0)?;
        
        Ok(notes)
    }

    /// Get notes without entity links (for extraction)
    #[instrument(skip(self))]
    pub async fn get_notes_without_entities(&self, limit: usize) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self.db
            .query("SELECT * FROM note WHERE id NOT IN (SELECT in FROM mentions) LIMIT $limit")
            .bind(("limit", limit))
            .await?
            .take(0)?;

        Ok(notes)
    }
    
    /// Update note embedding
    #[instrument(skip(self, embedding))]
    pub async fn update_note_embedding(&self, id: &surrealdb::RecordId, embedding: Vec<f32>) -> Result<()> {
        self.db
            .query("UPDATE note SET embedding = $embedding, updated_at = time::now() WHERE id = $id")
            .bind(("id", id.clone()))
            .bind(("embedding", embedding))
            .await?;
        
        Ok(())
    }
    
    // ==========================================
    // SEARCH OPERATIONS  
    // ==========================================
    
    /// Hybrid search combining vector similarity and full-text
    #[instrument(skip(self, embedding))]
    pub async fn hybrid_search(
        &self,
        query_text: &str,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        // Run vector search
        let vec_results = self.vector_search(embedding.clone(), limit).await?;
        
        // Run fulltext search
        let fts_results = self.fulltext_search(query_text, limit).await?;
        
        // Merge results using HashMap to deduplicate and combine scores
        use std::collections::hash_map::Entry;
        let mut map = std::collections::HashMap::new();
        
        for r in vec_results {
            map.insert(r.id.clone(), r);
        }
        
        for r in fts_results {
            match map.entry(r.id.clone()) {
                Entry::Occupied(mut e) => {
                    // Update existing with fts_score if present
                    if let Some(score) = r.fts_score {
                         e.get_mut().fts_score = Some(score);
                    }
                }
                Entry::Vacant(e) => {
                    e.insert(r);
                }
            }
        }
        
        let results: Vec<SearchResult> = map.into_values().collect();
        Ok(results)
    }
    #[instrument(skip(self, embedding))]
    pub async fn vector_search(
        &self,
        embedding: Vec<f32>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let results: Vec<SearchResult> = self.db
            .query(r#"
                SELECT 
                    id,
                    title,
                    content,
                    note_type,
                    tags,
                    created_at,
                    vector::distance::knn() AS vec_distance
                FROM note
                WHERE embedding <|100,COSINE|> $embedding
                LIMIT $limit
            "#)
            .bind(("embedding", embedding))
            .bind(("limit", limit))
            .await?
            .take(0)?;
        
        // let results = results.unwrap_or_default();
        Ok(results)
    }
    
    /// Full-text search only
    #[instrument(skip(self))]
    pub async fn fulltext_search(
        &self,
        query: &str,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        let results: Vec<SearchResult> = self.db
            .query(r#"
                SELECT 
                    id,
                    title,
                    content,
                    note_type,
                    tags,
                    created_at,
                    search::score(0) + search::score(1) AS fts_score
                FROM note
                WHERE content @0@ $query OR title @1@ $query
                ORDER BY fts_score DESC
                LIMIT $limit
            "#)
            .bind(("query", query.to_string()))
            .bind(("limit", limit))
            .await?
            .take(0)?;
        
        // let results = results.unwrap_or_default();
        Ok(results)
    }
    
    // ==========================================
    // GRAPH OPERATIONS
    // ==========================================
    
    /// Create a relationship between notes
    #[instrument(skip(self))]
    pub async fn create_edge(
        &self,
        from_id: &surrealdb::RecordId,
        to_id: &surrealdb::RecordId,
        edge_type: EdgeType,
        confidence: Option<f32>,
    ) -> Result<()> {
        // Edge tables are regular SCHEMAFULL tables with 'in' and 'out' fields
        // Use regular INSERT INTO (INSERT RELATION INTO requires TYPE RELATION tables)
        let table = edge_type.to_string();
        let query = format!(
             "INSERT INTO {table} (in, out, confidence, created_at) VALUES ($from, $to, $confidence, time::now())"
        );

        let result = self.db
            .query(&query)
            .bind(("from", from_id.clone()))
            .bind(("to", to_id.clone()))
            .bind(("confidence", confidence))
            .await?;

        // Check for errors in the query result
        result.check()?;

        Ok(())
    }
    
    /// Get notes related to a given note (any direction)
    #[instrument(skip(self))]
    pub async fn get_related_notes(&self, note_id: &surrealdb::RecordId) -> Result<RelatedNotes> {
        let result: Vec<RelatedNotes> = self.db
            .query(r#"
                SELECT 
                    (SELECT * FROM ->supports->note) AS supporting,
                    (SELECT * FROM <-supports<-note) AS supported_by,
                    (SELECT * FROM ->contradicts->note) AS contradicting,
                    (SELECT * FROM <-contradicts<-note) AS contradicted_by,
                    (SELECT * FROM ->related_to->note) AS related,
                    (SELECT * FROM <-related_to<-note) AS related_from
                FROM note
                WHERE id = $id
            "#)
            .bind(("id", note_id.clone()))
            .await?
            .take(0)?;
        
        result.into_iter().next().ok_or_else(|| {
            DbError::NotFound("note".into(), note_id.to_string())
        })
    }
    
    /// Find orphan notes (no connections)
    #[instrument(skip(self))]
    pub async fn find_orphan_notes(&self) -> Result<Vec<Note>> {
        let notes: Vec<Note> = self.db
            .query(r#"
                SELECT * FROM note 
                WHERE 
                    array::len(->supports->note) = 0 AND
                    array::len(<-supports<-note) = 0 AND
                    array::len(->contradicts->note) = 0 AND
                    array::len(<-contradicts<-note) = 0 AND
                    array::len(->related_to->note) = 0 AND
                    array::len(<-related_to<-note) = 0
            "#)
            .await?
            .take(0)?;
        
        Ok(notes)
    }
    
    /// Find potentially related notes (for gardener suggestions)
    #[instrument(skip(self, embedding))]
    pub async fn find_similar_notes(
        &self,
        note_id: &str,
        embedding: Vec<f32>,
        threshold: f32,
        limit: usize,
    ) -> Result<Vec<SimilarNote>> {
        let results: Vec<SimilarNote> = self.db
            .query(r#"
                SELECT
                    id,
                    title,
                    content,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM note
                WHERE
                    <string>id != $note_id AND
                    embedding IS NOT NONE AND
                    vector::similarity::cosine(embedding, $embedding) > $threshold
                ORDER BY similarity DESC
                LIMIT $limit
            "#)
            .bind(("note_id", format!("note:{}", note_id)))
            .bind(("embedding", embedding))
            .bind(("threshold", threshold))
            .bind(("limit", limit))
            .await?
            .take(0)?;

        Ok(results)
    }
    
    // ==========================================
    // ENTITY OPERATIONS
    // ==========================================
    
    /// Create or get existing entity by canonical name
    #[instrument(skip(self))]
    pub async fn upsert_entity(&self, entity: Entity) -> Result<Entity> {
        let Entity {
            id: _,
            entity_type,
            name,
            canonical_name,
            embedding,
            metadata,
            created_at: _,
        } = entity;

        let result: Option<Entity> = self.db
            .query(r#"
                INSERT INTO entity (entity_type, name, canonical_name, embedding, metadata, created_at)
                VALUES ($entity_type, $name, $canonical_name, $embedding, $metadata, time::now())
                ON DUPLICATE KEY UPDATE 
                    name = $name,
                    embedding = $embedding
            "#)
            .bind(("entity_type", entity_type.clone()))
            .bind(("name", name.clone()))
            .bind(("canonical_name", canonical_name.clone()))
            .bind(("embedding", embedding.clone()))
            .bind(("metadata", metadata.clone()))
            .await?
            .take(0)?;

        if let Some(entity) = result {
            return Ok(entity);
        }

        // If SurrealDB doesn't return the id on upsert, look it up by canonical name
        let fetched: Option<Entity> = self.db
            .query("SELECT * FROM entity WHERE canonical_name = $canonical_name LIMIT 1")
            .bind(("canonical_name", canonical_name))
            .await?
            .take(0)?;

        fetched.ok_or_else(|| DbError::CreateFailed("entity".into()))
    }
    
    /// Link a note to an entity
    #[instrument(skip(self))]
    pub async fn link_note_to_entity(&self, note_id: &surrealdb::RecordId, entity_id: &surrealdb::RecordId) -> Result<()> {
        self.db
            .query("RELATE $note_id->mentions->$entity_id")
            .bind(("note_id", note_id.clone()))
            .bind(("entity_id", entity_id.clone()))
            .await?;
        
        Ok(())
    }

    /// Get entities linked to a note
    #[instrument(skip(self))]
    pub async fn get_entities_for_note(&self, note_id: &str) -> Result<Vec<Entity>> {
        let note_id = if note_id.starts_with("note:") {
            note_id.to_string()
        } else {
            format!("note:{}", note_id)
        };
        let entities: Vec<Entity> = self.db
            .query("SELECT out.* FROM mentions WHERE in = $note_id")
            .bind(("note_id", note_id))
            .await?
            .take(0)?;

        Ok(entities)
    }

    /// List note-to-note edges across all edge tables
    #[instrument(skip(self))]
    pub async fn list_note_edges(&self, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let mut edges: Vec<NoteEdgeRow> = Vec::new();
        let limit = limit.max(1);

        edges.extend(self.query_edges_table("supports", limit).await?);
        edges.extend(self.query_edges_table("contradicts", limit).await?);
        edges.extend(self.query_edges_table("related_to", limit).await?);
        edges.extend(self.query_edges_table("derived_from", limit).await?);

        Ok(edges)
    }

    /// Get note-to-note edges for a specific note id (in or out)
    #[instrument(skip(self))]
    pub async fn get_note_edges(&self, note_id: &str) -> Result<Vec<NoteEdgeRow>> {
        let note_id = normalize_note_id(note_id);
        let mut edges: Vec<NoteEdgeRow> = Vec::new();

        edges.extend(self.query_edges_for_note("supports", &note_id).await?);
        edges.extend(self.query_edges_for_note("contradicts", &note_id).await?);
        edges.extend(self.query_edges_for_note("related_to", &note_id).await?);
        edges.extend(self.query_edges_for_note("derived_from", &note_id).await?);

        Ok(edges)
    }
    
    // ==========================================
    // SOURCE OPERATIONS
    // ==========================================
    
    /// Create a source
    #[instrument(skip(self, source))]
    pub async fn create_source(&self, source: Source) -> Result<Source> {
        // Insert source via a raw query and ignore the returned record; we only
        // need the data persisted, not the generated SurrealDB `id` value.
        self.db
            .query("CREATE source CONTENT $source")
            .bind(("source", source.clone()))
            .await?;

        Ok(source)
    }
    
    // ==========================================
    // STATS
    // ==========================================
    
    /// Get database statistics
    #[instrument(skip(self))]
    pub async fn get_stats(&self) -> Result<DbStats> {
        let stats: Vec<DbStats> = self.db
            .query(r#"
                RETURN {
                    note_count: (SELECT count() FROM note GROUP ALL)[0].count,
                    entity_count: (SELECT count() FROM entity GROUP ALL)[0].count,
                    source_count: (SELECT count() FROM source GROUP ALL)[0].count,
                    mention_count: (SELECT count() FROM mentions GROUP ALL)[0].count,
                    edge_count: (
                        (SELECT count() FROM supports GROUP ALL)[0].count +
                        (SELECT count() FROM contradicts GROUP ALL)[0].count +
                        (SELECT count() FROM related_to GROUP ALL)[0].count
                    )
                }
            "#)
            .await?
            .take(0)?;
        
        stats.into_iter().next().ok_or_else(|| DbError::QueryFailed("stats".into()))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteEdgeRow {
    pub edge_type: String,
    pub in_id: RecordId,
    pub out_id: RecordId,
    #[serde(default)]
    pub confidence: Option<f32>,
    #[serde(default)]
    pub reason: Option<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

fn normalize_note_id(note_id: &str) -> String {
    if note_id.starts_with("note:") {
        note_id.to_string()
    } else {
        format!("note:{}", note_id)
    }
}

impl Repository {
    async fn query_edges_table(&self, table: &str, limit: usize) -> Result<Vec<NoteEdgeRow>> {
        let query = format!(
            "SELECT '{table}' AS edge_type, in AS in_id, out AS out_id, confidence, reason, created_at FROM {table} LIMIT $limit"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("limit", limit))
            .await?
            .take(0)?;
        Ok(edges)
    }

    async fn query_edges_for_note(
        &self,
        table: &str,
        note_id: &str,
    ) -> Result<Vec<NoteEdgeRow>> {
        let note_id = note_id.to_string();
        let query = format!(
            "SELECT '{table}' AS edge_type, in AS in_id, out AS out_id, confidence, reason, created_at \
             FROM {table} WHERE in = $note_id OR out = $note_id"
        );
        let edges: Vec<NoteEdgeRow> = self
            .db
            .query(&query)
            .bind(("note_id", note_id))
            .await?
            .take(0)?;
        Ok(edges)
    }
}

// ==========================================
// RESULT TYPES
// ==========================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub note_type: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RelatedNotes {
    #[serde(default)]
    pub supporting: Vec<Note>,
    #[serde(default)]
    pub supported_by: Vec<Note>,
    #[serde(default)]
    pub contradicting: Vec<Note>,
    #[serde(default)]
    pub contradicted_by: Vec<Note>,
    #[serde(default)]
    pub related: Vec<Note>,
    #[serde(default)]
    pub related_from: Vec<Note>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarNote {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub similarity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DbStats {
    #[serde(default)]
    pub note_count: i64,
    #[serde(default)]
    pub entity_count: i64,
    #[serde(default)]
    pub source_count: i64,
    #[serde(default)]
    pub mention_count: i64,
    #[serde(default)]
    pub edge_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::init_memory;
    
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
}
