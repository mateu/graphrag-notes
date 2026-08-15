//! Shared repository query rows and result models.
//!
//! These types are deliberately data-only. Domain modules own their SQL while
//! callers continue importing the stable re-exports from `repository`.

use super::*;

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SearchResult {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub note_type: String,
    pub tags: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct MessageSearchResult {
    pub id: RecordId,
    pub conversation_id: RecordId,
    pub conversation_uuid: String,
    pub message_index: i64,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub created_at: Option<chrono::DateTime<chrono::Utc>>,
    #[serde(default)]
    pub source_uri: Option<String>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct ConversationSearchResult {
    pub id: RecordId,
    pub uuid: String,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub summary: Option<String>,
    #[serde(default)]
    pub source_uri: Option<String>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    #[serde(default)]
    pub vec_distance: Option<f32>,
    #[serde(default)]
    pub fts_score: Option<f32>,
    #[serde(skip, default)]
    #[surreal(default)]
    pub fusion: FusionEvidence,
}

impl FusionRecord for SearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

impl FusionRecord for MessageSearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

impl FusionRecord for ConversationSearchResult {
    fn fusion_id(&self) -> String {
        record_id_to_string(&self.id)
    }

    fn vector_distance(&self) -> Option<f32> {
        self.vec_distance
    }

    fn fulltext_score(&self) -> Option<f32> {
        self.fts_score
    }

    fn set_fusion_evidence(&mut self, evidence: FusionEvidence) {
        self.fusion = evidence;
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, SurrealValue)]
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

#[derive(Debug, Clone, Serialize, Deserialize, SurrealValue)]
pub struct SimilarNote {
    pub id: RecordId,
    pub title: Option<String>,
    pub content: String,
    pub similarity: f32,
}
