//! Edge types - relationships between notes

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use surrealdb::RecordId;

/// Types of relationships between notes
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// One note supports/confirms another
    Supports,
    /// One note contradicts another
    Contradicts,
    /// One note is derived from another
    DerivedFrom,
    /// One note references/mentions another
    References,
    /// One note is related to another (generic)
    RelatedTo,
    /// Note mentions an entity
    Mentions,
    /// Note is tagged with something
    TaggedWith,
}

impl std::fmt::Display for EdgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EdgeType::Supports => write!(f, "supports"),
            EdgeType::Contradicts => write!(f, "contradicts"),
            EdgeType::DerivedFrom => write!(f, "derived_from"),
            EdgeType::References => write!(f, "references"),
            EdgeType::RelatedTo => write!(f, "related_to"),
            EdgeType::Mentions => write!(f, "mentions"),
            EdgeType::TaggedWith => write!(f, "tagged_with"),
        }
    }
}

/// An edge/relationship in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// Unique identifier (SurrealDB generates this)
    pub id: Option<RecordId>,

    /// Source node ID (the "from" node)
    pub from_id: String,

    /// Target node ID (the "to" node)
    pub to_id: String,

    /// Type of relationship
    pub edge_type: EdgeType,

    /// Confidence score (0.0 - 1.0), for auto-generated edges
    pub confidence: Option<f32>,

    /// Additional metadata/context about the relationship
    #[serde(default)]
    pub metadata: serde_json::Value,

    /// When this edge was created
    #[serde(skip_serializing)]
    pub created_at: DateTime<Utc>,

    /// Whether this edge was manually created or auto-generated
    pub is_manual: bool,
}

impl Edge {
    /// Create a new edge
    pub fn new(from_id: impl Into<String>, to_id: impl Into<String>, edge_type: EdgeType) -> Self {
        Self {
            id: None,
            from_id: from_id.into(),
            to_id: to_id.into(),
            edge_type,
            confidence: None,
            metadata: serde_json::Value::Null,
            created_at: Utc::now(),
            is_manual: false,
        }
    }

    /// Builder: set confidence
    pub fn with_confidence(mut self, confidence: f32) -> Self {
        self.confidence = Some(confidence.clamp(0.0, 1.0));
        self
    }

    /// Builder: mark as manual
    pub fn manual(mut self) -> Self {
        self.is_manual = true;
        self
    }
}

/// A suggested edge that needs user confirmation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuggestedEdge {
    pub edge: Edge,
    /// Why this edge was suggested
    pub reason: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new("note:1", "note:2", EdgeType::Supports).with_confidence(0.85);

        assert_eq!(edge.from_id, "note:1");
        assert_eq!(edge.to_id, "note:2");
        assert_eq!(edge.edge_type, EdgeType::Supports);
        assert_eq!(edge.confidence, Some(0.85));
        assert!(!edge.is_manual);
    }

    #[test]
    fn test_edge_type_display() {
        assert_eq!(EdgeType::Supports.to_string(), "supports");
        assert_eq!(EdgeType::Contradicts.to_string(), "contradicts");
    }
}
