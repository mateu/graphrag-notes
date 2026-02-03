//! Gardener Agent - Maintains graph connections

use crate::{MlClient, Result};
use graphrag_core::{Note, EdgeType};
use graphrag_db::Repository;
use tracing::{info, debug, instrument};

/// A suggested connection between notes
#[derive(Debug)]
pub struct SuggestedConnection {
    pub from_note: Note,
    pub to_note: Note,
    pub edge_type: EdgeType,
    pub similarity: f32,
    pub reason: String,
}

/// The Gardener agent maintains graph health
pub struct GardenerAgent {
    repo: Repository,
    #[allow(dead_code)]
    ml: MlClient,
    /// Minimum similarity threshold for suggesting connections
    similarity_threshold: f32,
}

impl GardenerAgent {
    /// Create a new Gardener agent
    pub fn new(repo: Repository, ml: MlClient) -> Self {
        Self {
            repo,
            ml,
            similarity_threshold: 0.7,
        }
    }
    
    /// Set the similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }
    
    /// Find orphan notes (no connections)
    #[instrument(skip(self))]
    pub async fn find_orphans(&self) -> Result<Vec<Note>> {
        info!("Finding orphan notes...");
        
        let orphans = self.repo.find_orphan_notes().await?;
        
        info!("Found {} orphan notes", orphans.len());
        
        Ok(orphans)
    }
    
    /// Suggest connections for orphan notes
    #[instrument(skip(self))]
    pub async fn suggest_connections(&self) -> Result<Vec<SuggestedConnection>> {
        let orphans = self.find_orphans().await?;
        
        if orphans.is_empty() {
            info!("No orphan notes to process");
            return Ok(Vec::new());
        }
        
        info!("Finding connections for {} orphan notes", orphans.len());
        
        let mut suggestions = Vec::new();
        
        for orphan in orphans {
            if orphan.embedding.is_empty() {
                debug!("Skipping orphan without embedding: {:?}", orphan.id);
                continue;
            }
            
            let note_id = match orphan.id.as_ref() {
                Some(id) => id.key().to_string(),
                None => {
                    debug!("Skipping orphan without id");
                    continue;
                }
            };

            // Find similar notes
            let similar = self.repo
                .find_similar_notes(
                    &note_id,
                    orphan.embedding.clone(),
                    self.similarity_threshold,
                    5,
                )
                .await?;
            
            for sim in similar {
                // Get the full target note
                let target_id = sim.id.key().to_string();
                if let Some(target_note) = self.repo.get_note(&target_id).await? {
                    suggestions.push(SuggestedConnection {
                        from_note: orphan.clone(),
                        to_note: target_note,
                        edge_type: EdgeType::RelatedTo,
                        similarity: sim.similarity,
                        reason: format!(
                            "High semantic similarity ({:.1}%)",
                            sim.similarity * 100.0
                        ),
                    });
                }
            }
        }
        
        info!("Generated {} connection suggestions", suggestions.len());
        
        Ok(suggestions)
    }
    
    /// Apply a suggested connection
    #[instrument(skip(self, suggestion))]
    pub async fn apply_connection(&self, suggestion: &SuggestedConnection) -> Result<()> {
        let from_id = suggestion.from_note.id.as_ref()
            .ok_or_else(|| crate::AgentError::NotFound("from note id".into()))?;
        
        let to_id = suggestion.to_note.id.as_ref()
            .ok_or_else(|| crate::AgentError::NotFound("to note id".into()))?;
        
        self.repo.create_edge(
            from_id,
            to_id,
            suggestion.edge_type.clone(),
            Some(suggestion.similarity),
        ).await?;
        
        info!(
            "Created {:?} edge from {} to {}",
            suggestion.edge_type, from_id, to_id
        );
        
        Ok(())
    }
    
    /// Run full maintenance cycle
    #[instrument(skip(self))]
    pub async fn run_maintenance(&self) -> Result<MaintenanceReport> {
        info!("Starting maintenance cycle...");
        
        let orphans_before = self.find_orphans().await?.len();
        let suggestions = self.suggest_connections().await?;
        let suggestions_count = suggestions.len();
        
        // Auto-apply high-confidence suggestions
        let mut applied = 0;
        for suggestion in &suggestions {
            if suggestion.similarity > 0.85 {
                if self.apply_connection(suggestion).await.is_ok() {
                    applied += 1;
                }
            }
        }
        
        let orphans_after = self.find_orphans().await?.len();
        
        let report = MaintenanceReport {
            orphans_found: orphans_before,
            suggestions_generated: suggestions_count,
            connections_applied: applied,
            orphans_remaining: orphans_after,
        };
        
        info!("Maintenance complete: {:?}", report);
        
        Ok(report)
    }
}

/// Report from a maintenance run
#[derive(Debug)]
pub struct MaintenanceReport {
    pub orphans_found: usize,
    pub suggestions_generated: usize,
    pub connections_applied: usize,
    pub orphans_remaining: usize,
}

#[cfg(test)]
mod tests {
    // Integration tests require the ML worker running
}
