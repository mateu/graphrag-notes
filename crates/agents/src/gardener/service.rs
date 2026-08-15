//! Gardener Agent - Maintains graph connections

use crate::Result;
use graphrag_core::{record_id_to_string, EdgeType, Note, ProposedEdge};
use graphrag_db::Repository;
use std::collections::BTreeSet;
use tracing::{debug, info, instrument};

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
    /// Minimum similarity threshold for suggesting connections
    similarity_threshold: f32,
    auto_apply_threshold: f32,
    auto_apply_enabled: bool,
    max_suggestions: usize,
}

impl GardenerAgent {
    /// Create a new Gardener agent
    pub fn new(repo: Repository) -> Self {
        Self {
            repo,
            similarity_threshold: 0.7,
            auto_apply_threshold: 0.85,
            // Mutating accepted-edge tables is always opt-in. A configured
            // threshold selects a policy boundary but never enables it alone.
            auto_apply_enabled: false,
            max_suggestions: 50,
        }
    }

    /// Set the similarity threshold
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold;
        self
    }

    /// Set the minimum confidence required to automatically create a
    /// suggested connection during maintenance.
    pub fn with_auto_apply_threshold(mut self, threshold: f32) -> Self {
        self.auto_apply_threshold = threshold;
        self
    }

    /// Explicitly opt into automatic acceptance of similarity proposals.
    /// The repository still restricts the policy to `related_to` proposals.
    pub fn with_auto_apply_policy(mut self, enabled: bool, threshold: f32) -> Self {
        self.auto_apply_enabled = enabled;
        self.auto_apply_threshold = threshold;
        self
    }

    /// Limit how many candidate connections a maintenance run produces.
    pub fn with_max_suggestions(mut self, max_suggestions: usize) -> Self {
        self.max_suggestions = max_suggestions;
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
        // `related_to` is symmetric. Orphan A can nominate B and orphan B
        // can nominate A in the same scan, but both map to one canonical
        // proposal. Keep this scan's output one-to-one with that proposal.
        let mut seen_pairs = BTreeSet::new();

        for orphan in orphans {
            if suggestions.len() >= self.max_suggestions {
                break;
            }
            if orphan.embedding.is_empty() {
                debug!("Skipping orphan without embedding: {:?}", orphan.id);
                continue;
            }

            let note_id = match orphan.id.as_ref() {
                Some(id) => record_id_to_string(id),
                None => {
                    debug!("Skipping orphan without id");
                    continue;
                }
            };

            // Find similar notes
            let similar = self
                .repo
                .find_similar_notes(
                    &note_id,
                    orphan.embedding.clone(),
                    self.similarity_threshold,
                    5,
                )
                .await?;

            for sim in similar {
                // Get the full target note
                let target_id = record_id_to_string(&sim.id);
                if let Some(target_note) = self.repo.get_note(&target_id).await? {
                    let pair = canonical_related_pair(&note_id, &target_id);
                    if !seen_pairs.insert(pair) {
                        continue;
                    }
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
                    if suggestions.len() >= self.max_suggestions {
                        break;
                    }
                }
            }
        }

        info!("Generated {} connection suggestions", suggestions.len());

        Ok(suggestions)
    }

    /// Apply a suggested connection
    #[instrument(skip(self, suggestion))]
    pub async fn apply_connection(&self, suggestion: &SuggestedConnection) -> Result<()> {
        let from_id = suggestion
            .from_note
            .id
            .as_ref()
            .ok_or_else(|| crate::AgentError::NotFound("from note id".into()))?;

        let to_id = suggestion
            .to_note
            .id
            .as_ref()
            .ok_or_else(|| crate::AgentError::NotFound("to note id".into()))?;

        if suggestion.edge_type != EdgeType::RelatedTo {
            return Err(crate::AgentError::Processing(
                "similarity suggestions may only create related_to proposals".into(),
            ));
        }
        let proposal = self
            .repo
            .upsert_gardener_proposal(
                from_id,
                to_id,
                suggestion.similarity,
                suggestion.reason.clone(),
                Some(env!("CARGO_PKG_VERSION").into()),
                None,
            )
            .await?;
        let proposal_id = proposal
            .id
            .ok_or_else(|| crate::AgentError::Processing("persisted proposal has no id".into()))?;
        self.repo
            .accept_edge_proposal(
                &proposal_id,
                Some("explicit gardener apply_connection call".into()),
                Some("explicit manual acceptance".into()),
                true,
            )
            .await?;

        info!(
            "Created {:?} edge from {:?} to {:?}",
            suggestion.edge_type, from_id, to_id
        );

        Ok(())
    }

    /// Run full maintenance cycle
    #[instrument(skip(self))]
    pub async fn run_maintenance(&self) -> Result<MaintenanceReport> {
        info!("Starting maintenance cycle...");

        let scan = self.scan(false).await?;
        let mut applied = 0;
        if self.auto_apply_enabled {
            applied = self
                .repo
                .accept_gardener_proposals_above(
                    self.auto_apply_threshold,
                    Some("configured gardener auto-apply policy".into()),
                )
                .await?;
        }

        let orphans_after = self.find_orphans().await?.len();

        let report = MaintenanceReport {
            orphans_found: scan.orphans_found,
            suggestions_generated: scan.suggestions_generated,
            connections_applied: applied,
            orphans_remaining: orphans_after,
        };

        info!("Maintenance complete: {:?}", report);

        Ok(report)
    }

    /// Produce and optionally persist reviewable proposals. A dry run is fully
    /// non-mutating; a normal scan changes only `proposed_edge`, never an
    /// accepted edge table.
    #[instrument(skip(self))]
    pub async fn scan(&self, dry_run: bool) -> Result<ScanReport> {
        let orphans_found = self.find_orphans().await?.len();
        let suggestions = self.suggest_connections().await?;
        let suggestions_generated = suggestions.len();
        let mut proposals = Vec::with_capacity(suggestions_generated);

        if !dry_run {
            for suggestion in suggestions {
                let from_id = suggestion
                    .from_note
                    .id
                    .as_ref()
                    .ok_or_else(|| crate::AgentError::NotFound("from note id".into()))?;
                let to_id = suggestion
                    .to_note
                    .id
                    .as_ref()
                    .ok_or_else(|| crate::AgentError::NotFound("to note id".into()))?;
                proposals.push(
                    self.repo
                        .upsert_gardener_proposal(
                            from_id,
                            to_id,
                            suggestion.similarity,
                            suggestion.reason,
                            Some(env!("CARGO_PKG_VERSION").into()),
                            None,
                        )
                        .await?,
                );
            }
        }

        Ok(ScanReport {
            orphans_found,
            suggestions_generated,
            proposals,
            dry_run,
        })
    }
}

fn canonical_related_pair(first: &str, second: &str) -> (String, String) {
    if first <= second {
        (first.to_owned(), second.to_owned())
    } else {
        (second.to_owned(), first.to_owned())
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

/// Outcome of one Gardener scan.
#[derive(Debug)]
pub struct ScanReport {
    pub orphans_found: usize,
    pub suggestions_generated: usize,
    pub proposals: Vec<ProposedEdge>,
    pub dry_run: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::ProposedEdgeStatus;
    use graphrag_db::init_memory;

    #[tokio::test]
    async fn scan_dry_run_and_default_maintenance_do_not_mutate_accepted_edges() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("first").with_embedding(vec![1.0; 1024]))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_embedding(vec![1.0; 1024]))
            .await
            .unwrap();
        let gardener = GardenerAgent::new(repo.clone()).with_threshold(0.7);

        let preview = gardener.scan(true).await.unwrap();
        assert!(preview.dry_run);
        // Equal embeddings make each orphan nominate the other. A symmetric
        // `related_to` proposal must still be surfaced once per scan.
        assert_eq!(preview.suggestions_generated, 1);
        assert!(repo.list_edge_proposals(None, 10).await.unwrap().is_empty());
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());

        let scan = gardener.scan(false).await.unwrap();
        assert!(!scan.dry_run);
        assert_eq!(repo.list_edge_proposals(None, 10).await.unwrap().len(), 1);
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());

        let report = gardener.run_maintenance().await.unwrap();
        assert_eq!(report.connections_applied, 0);
        assert!(repo.list_note_edges(10).await.unwrap().is_empty());
        assert_eq!(
            repo.list_edge_proposals(Some(ProposedEdgeStatus::Pending), 10)
                .await
                .unwrap()
                .len(),
            1
        );
    }

    #[tokio::test]
    async fn explicit_auto_apply_policy_accepts_only_related_to_proposals() {
        let repo = Repository::new(init_memory().await.unwrap());
        repo.create_note(Note::new("first").with_embedding(vec![1.0; 1024]))
            .await
            .unwrap();
        repo.create_note(Note::new("second").with_embedding(vec![1.0; 1024]))
            .await
            .unwrap();
        let gardener = GardenerAgent::new(repo.clone())
            .with_threshold(0.7)
            .with_auto_apply_policy(true, 0.8);

        let report = gardener.run_maintenance().await.unwrap();
        assert_eq!(report.connections_applied, 1);
        let edges = repo.list_note_edges(10).await.unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].edge_type, "related_to");
        assert!(!edges[0].is_manual);
    }
}
