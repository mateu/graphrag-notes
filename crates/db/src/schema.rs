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
DEFINE TABLE note SCHEMAFULL;
DEFINE FIELD note_type ON note TYPE string DEFAULT 'raw';
DEFINE FIELD title ON note TYPE option<string>;
DEFINE FIELD content ON note TYPE string;
DEFINE FIELD embedding ON note TYPE option<array<float>>;
DEFINE FIELD source_id ON note TYPE option<record<source>>;
DEFINE FIELD tags ON note TYPE array<string> DEFAULT [];
DEFINE FIELD created_at ON note TYPE datetime DEFAULT time::now();
DEFINE FIELD updated_at ON note TYPE datetime DEFAULT time::now();

-- Entities table
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD entity_type ON entity TYPE string DEFAULT 'other';
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD canonical_name ON entity TYPE string;
DEFINE FIELD embedding ON entity TYPE option<array<float>>;
DEFINE FIELD metadata ON entity TYPE option<object>;
DEFINE FIELD created_at ON entity TYPE datetime DEFAULT time::now();

-- Sources table  
DEFINE TABLE source SCHEMAFULL;
DEFINE FIELD source_type ON source TYPE string DEFAULT 'manual';
DEFINE FIELD title ON source TYPE option<string>;
DEFINE FIELD uri ON source TYPE option<string>;
DEFINE FIELD content ON source TYPE option<string>;
DEFINE FIELD metadata ON source TYPE option<object>;
DEFINE FIELD created_at ON source TYPE datetime DEFAULT time::now();

-- ============================================
-- GRAPH EDGE TABLES
-- ============================================

-- Note relationships
DEFINE TABLE supports SCHEMAFULL;
DEFINE FIELD in ON supports TYPE record<note>;
DEFINE FIELD out ON supports TYPE record<note>;
DEFINE FIELD confidence ON supports TYPE option<float>;
DEFINE FIELD is_manual ON supports TYPE bool DEFAULT false;
DEFINE FIELD created_at ON supports TYPE datetime DEFAULT time::now();

DEFINE TABLE contradicts SCHEMAFULL;
DEFINE FIELD in ON contradicts TYPE record<note>;
DEFINE FIELD out ON contradicts TYPE record<note>;
DEFINE FIELD confidence ON contradicts TYPE option<float>;
DEFINE FIELD is_manual ON contradicts TYPE bool DEFAULT false;
DEFINE FIELD created_at ON contradicts TYPE datetime DEFAULT time::now();

DEFINE TABLE derived_from SCHEMAFULL;
DEFINE FIELD in ON derived_from TYPE record<note>;
DEFINE FIELD out ON derived_from TYPE record<note>;
DEFINE FIELD created_at ON derived_from TYPE datetime DEFAULT time::now();

DEFINE TABLE related_to SCHEMAFULL;
DEFINE FIELD in ON related_to TYPE record<note>;
DEFINE FIELD out ON related_to TYPE record<note>;
DEFINE FIELD confidence ON related_to TYPE option<float>;
DEFINE FIELD reason ON related_to TYPE option<string>;
DEFINE FIELD is_manual ON related_to TYPE bool DEFAULT false;
DEFINE FIELD created_at ON related_to TYPE datetime DEFAULT time::now();

-- Note to entity relationships
DEFINE TABLE mentions SCHEMAFULL;
DEFINE FIELD in ON mentions TYPE record<note>;
DEFINE FIELD out ON mentions TYPE record<entity>;
DEFINE FIELD created_at ON mentions TYPE datetime DEFAULT time::now();

-- ============================================
-- INDEXES
-- ============================================

-- Search analyzers
DEFINE ANALYZER IF NOT EXISTS ascii
    TOKENIZERS class
    FILTERS lowercase, ascii;

-- Full-text search on note content
DEFINE INDEX idx_note_content ON note FIELDS content 
    SEARCH ANALYZER ascii BM25;

-- Full-text search on note title
DEFINE INDEX idx_note_title ON note FIELDS title 
    SEARCH ANALYZER ascii BM25;

-- Vector index for semantic search (HNSW for performance)
DEFINE INDEX idx_note_embedding ON note FIELDS embedding 
    HNSW DIMENSION 1024 DIST COSINE;

-- Entity lookups
DEFINE INDEX idx_entity_canonical ON entity FIELDS canonical_name UNIQUE;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;

-- Source lookups
DEFINE INDEX idx_source_uri ON source FIELDS uri;

-- Note type filtering
DEFINE INDEX idx_note_type ON note FIELDS note_type;

-- Tag filtering
DEFINE INDEX idx_note_tags ON note FIELDS tags;
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
