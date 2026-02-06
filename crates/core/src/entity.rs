//! Entity types - people, concepts, projects, etc.

use chrono::{DateTime, Utc};
use surrealdb::RecordId;
use serde::{Deserialize, Serialize};

/// The type/classification of an entity
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EntityType {
    /// A person
    Person,
    /// An organization or company
    Organization,
    /// A concept or idea
    Concept,
    /// A project or initiative
    Project,
    /// A technology or tool
    Technology,
    /// A location
    Location,
    /// A date or time period
    Date,
    /// Generic/other
    Other,
}

impl Default for EntityType {
    fn default() -> Self {
        Self::Other
    }
}

/// An entity extracted from notes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Unique identifier
    pub id: Option<RecordId>,
    
    /// The type of entity
    #[serde(default)]
    pub entity_type: EntityType,
    
    /// Display name
    #[serde(default)]
    pub name: String,
    
    /// Canonical/normalized name for deduplication
    #[serde(default)]
    pub canonical_name: String,
    
    /// Vector embedding of the entity
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub embedding: Vec<f32>,
    
    /// Additional metadata
    #[serde(default)]
    pub metadata: serde_json::Value,
    
    /// When first seen
    #[serde(skip_serializing)]
    pub created_at: DateTime<Utc>,
}

impl Entity {
    /// Create a new entity
    pub fn new(name: impl Into<String>, entity_type: EntityType) -> Self {
        let name = name.into();
        let canonical = Self::canonicalize(&name);
        Self {
            id: None,
            entity_type,
            name,
            canonical_name: canonical,
            embedding: Vec::new(),
            metadata: serde_json::Value::Null,
            created_at: Utc::now(),
        }
    }
    
    /// Canonicalize a name for deduplication
    pub fn canonicalize(name: &str) -> String {
        name.to_lowercase()
            .trim()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }
    
    /// Builder: set embedding
    pub fn with_embedding(mut self, embedding: Vec<f32>) -> Self {
        self.embedding = embedding;
        self
    }
}

/// Result of entity extraction from text
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedEntity {
    pub name: String,
    pub entity_type: EntityType,
    /// Character offset in source text
    pub start: usize,
    pub end: usize,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_entity_creation() {
        let entity = Entity::new("John Doe", EntityType::Person);
        
        assert_eq!(entity.name, "John Doe");
        assert_eq!(entity.canonical_name, "john doe");
        assert_eq!(entity.entity_type, EntityType::Person);
    }
    
    #[test]
    fn test_canonicalization() {
        assert_eq!(Entity::canonicalize("  John   DOE  "), "john doe");
        assert_eq!(Entity::canonicalize("Machine Learning"), "machine learning");
    }
}
