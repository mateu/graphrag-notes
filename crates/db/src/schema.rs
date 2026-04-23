//! SurrealDB schema definitions

use crate::{DbConnection, Result};
use tracing::info;

/// Embedding dimension (Jina v3 default: 1024)
pub const EMBEDDING_DIMENSION: usize = 1024;

/// Initialize the database schema
pub async fn initialize_schema(db: &DbConnection) -> Result<()> {
    info!("Initializing database schema...");

    // Define tables and fields
    db.query(SCHEMA_DEFINITION).await?;

    info!("Schema initialized successfully");
    Ok(())
}

const SCHEMA_DEFINITION: &str = r#"
-- ============================================
-- TABLES
-- ============================================

-- Notes table
DEFINE TABLE IF NOT EXISTS note SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS note_type ON note TYPE string DEFAULT 'raw';
DEFINE FIELD IF NOT EXISTS title ON note TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content ON note TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON note TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS source_id ON note TYPE option<record<source>>;
DEFINE FIELD IF NOT EXISTS tags ON note TYPE array<string> DEFAULT [];
-- entity_ids: kept in the schema with DEFAULT [] so SurrealDB never returns NONE for this
-- field (serde `default` only handles absent fields, not explicit nulls). The Rust model marks
-- it `skip_serializing` because entity–note links live in the `mentions` edge table rather than
-- on the note record itself.
DEFINE FIELD IF NOT EXISTS entity_ids ON note TYPE array<string> DEFAULT [];
DEFINE FIELD IF NOT EXISTS created_at ON note TYPE datetime DEFAULT time::now();
DEFINE FIELD IF NOT EXISTS updated_at ON note TYPE datetime DEFAULT time::now();

-- Entities table
DEFINE TABLE IF NOT EXISTS entity SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS entity_type ON entity TYPE string DEFAULT 'other';
DEFINE FIELD IF NOT EXISTS name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS canonical_name ON entity TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON entity TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS metadata ON entity TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON entity TYPE datetime DEFAULT time::now();

-- Sources table  
DEFINE TABLE IF NOT EXISTS source SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS source_type ON source TYPE string DEFAULT 'manual';
DEFINE FIELD IF NOT EXISTS title ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS uri ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS content ON source TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON source TYPE option<object>;
DEFINE FIELD IF NOT EXISTS created_at ON source TYPE datetime DEFAULT time::now();

-- Conversations imported from chat exports
DEFINE TABLE IF NOT EXISTS conversation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS uuid ON conversation TYPE string;
DEFINE FIELD IF NOT EXISTS title ON conversation TYPE option<string>;
DEFINE FIELD IF NOT EXISTS summary ON conversation TYPE option<string>;
DEFINE FIELD IF NOT EXISTS source_uri ON conversation TYPE option<string>;
DEFINE FIELD IF NOT EXISTS account_uuid ON conversation TYPE option<string>;
DEFINE FIELD IF NOT EXISTS metadata ON conversation TYPE option<object>;
DEFINE FIELD IF NOT EXISTS summary_embedding ON conversation TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS created_at ON conversation TYPE datetime;
DEFINE FIELD IF NOT EXISTS updated_at ON conversation TYPE datetime;
DEFINE FIELD IF NOT EXISTS ingested_at ON conversation TYPE datetime DEFAULT time::now();

-- Messages imported from chat exports
DEFINE TABLE IF NOT EXISTS message SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS message_key ON message TYPE string;
DEFINE FIELD IF NOT EXISTS message_uuid ON message TYPE option<string>;
DEFINE FIELD IF NOT EXISTS conversation_id ON message TYPE record<conversation>;
DEFINE FIELD IF NOT EXISTS conversation_uuid ON message TYPE string;
DEFINE FIELD IF NOT EXISTS message_index ON message TYPE int;
DEFINE FIELD IF NOT EXISTS role ON message TYPE string;
DEFINE FIELD IF NOT EXISTS content ON message TYPE string;
DEFINE FIELD IF NOT EXISTS embedding ON message TYPE option<array<float>>;
DEFINE FIELD IF NOT EXISTS content_blocks ON message TYPE option<array<object>>;
DEFINE FIELD IF NOT EXISTS attachments ON message TYPE option<array<object>>;
DEFINE FIELD IF NOT EXISTS files ON message TYPE option<array<object>>;
DEFINE FIELD IF NOT EXISTS has_files ON message TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON message TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS updated_at ON message TYPE option<datetime>;
DEFINE FIELD IF NOT EXISTS ingested_at ON message TYPE datetime DEFAULT time::now();

-- ============================================
-- GRAPH EDGE TABLES
-- ============================================

-- Note relationships
DEFINE TABLE IF NOT EXISTS supports SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON supports TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON supports TYPE record<note>;
DEFINE FIELD IF NOT EXISTS confidence ON supports TYPE option<float>;
DEFINE FIELD IF NOT EXISTS is_manual ON supports TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON supports TYPE datetime DEFAULT time::now();

DEFINE TABLE IF NOT EXISTS contradicts SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON contradicts TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON contradicts TYPE record<note>;
DEFINE FIELD IF NOT EXISTS confidence ON contradicts TYPE option<float>;
DEFINE FIELD IF NOT EXISTS is_manual ON contradicts TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON contradicts TYPE datetime DEFAULT time::now();

DEFINE TABLE IF NOT EXISTS derived_from SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON derived_from TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON derived_from TYPE record<note>;
DEFINE FIELD IF NOT EXISTS created_at ON derived_from TYPE datetime DEFAULT time::now();

DEFINE TABLE IF NOT EXISTS related_to SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON related_to TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON related_to TYPE record<note>;
DEFINE FIELD IF NOT EXISTS confidence ON related_to TYPE option<float>;
DEFINE FIELD IF NOT EXISTS reason ON related_to TYPE option<string>;
DEFINE FIELD IF NOT EXISTS is_manual ON related_to TYPE bool DEFAULT false;
DEFINE FIELD IF NOT EXISTS created_at ON related_to TYPE datetime DEFAULT time::now();

-- Note to entity relationships
DEFINE TABLE IF NOT EXISTS mentions SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON mentions TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON mentions TYPE record<entity>;
DEFINE FIELD IF NOT EXISTS created_at ON mentions TYPE datetime DEFAULT time::now();

-- Note provenance relationships
DEFINE TABLE IF NOT EXISTS note_from_conversation SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON note_from_conversation TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON note_from_conversation TYPE record<conversation>;
DEFINE FIELD IF NOT EXISTS created_at ON note_from_conversation TYPE datetime DEFAULT time::now();

DEFINE TABLE IF NOT EXISTS note_from_message SCHEMAFULL;
DEFINE FIELD IF NOT EXISTS in ON note_from_message TYPE record<note>;
DEFINE FIELD IF NOT EXISTS out ON note_from_message TYPE record<message>;
DEFINE FIELD IF NOT EXISTS created_at ON note_from_message TYPE datetime DEFAULT time::now();

-- ============================================
-- INDEXES
-- ============================================

-- Search analyzers
DEFINE ANALYZER IF NOT EXISTS ascii
    TOKENIZERS class
    FILTERS lowercase, ascii;

-- Full-text search on note content
DEFINE INDEX IF NOT EXISTS idx_note_content ON note FIELDS content 
    FULLTEXT ANALYZER ascii BM25;

-- Full-text search on note title
DEFINE INDEX IF NOT EXISTS idx_note_title ON note FIELDS title 
    FULLTEXT ANALYZER ascii BM25;

-- Full-text search on chat messages and conversation summaries
DEFINE INDEX IF NOT EXISTS idx_message_content ON message FIELDS content
    FULLTEXT ANALYZER ascii BM25;
DEFINE INDEX IF NOT EXISTS idx_conversation_summary ON conversation FIELDS summary
    FULLTEXT ANALYZER ascii BM25;
DEFINE INDEX IF NOT EXISTS idx_conversation_title ON conversation FIELDS title
    FULLTEXT ANALYZER ascii BM25;

-- Vector index for semantic search (HNSW for performance)
DEFINE INDEX IF NOT EXISTS idx_note_embedding ON note FIELDS embedding 
    HNSW DIMENSION 1024 DIST COSINE;

-- Entity lookups
DEFINE INDEX IF NOT EXISTS idx_entity_canonical ON entity FIELDS canonical_name UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_entity_type ON entity FIELDS entity_type;

-- Source lookups
DEFINE INDEX IF NOT EXISTS idx_source_uri ON source FIELDS uri;

-- Conversation/message lookups
DEFINE INDEX IF NOT EXISTS idx_conversation_uuid ON conversation FIELDS uuid UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_message_key ON message FIELDS message_key UNIQUE;
DEFINE INDEX IF NOT EXISTS idx_message_conversation ON message FIELDS conversation_id;
DEFINE INDEX IF NOT EXISTS idx_message_uuid ON message FIELDS message_uuid;
DEFINE INDEX IF NOT EXISTS idx_message_embedding ON message FIELDS embedding HNSW DIMENSION 1024 DIST COSINE;
DEFINE INDEX IF NOT EXISTS idx_conversation_summary_embedding ON conversation FIELDS summary_embedding HNSW DIMENSION 1024 DIST COSINE;

-- Note type filtering
DEFINE INDEX IF NOT EXISTS idx_note_type ON note FIELDS note_type;

-- Tag filtering
DEFINE INDEX IF NOT EXISTS idx_note_tags ON note FIELDS tags;
"#;

#[cfg(test)]
mod tests {
    use crate::init_memory;

    #[tokio::test]
    async fn test_schema_initialization() {
        let db = init_memory().await.expect("Failed to init db");

        // Verify tables exist by selecting from them
        let notes: Vec<serde_json::Value> = db.select("note").await.unwrap();
        assert!(notes.is_empty());

        let entities: Vec<serde_json::Value> = db.select("entity").await.unwrap();
        assert!(entities.is_empty());
    }
}
