//! Request-local, closed-vocabulary retrieval observations. No user data is accepted.
use serde::Serialize;
use std::{cell::RefCell, collections::BTreeMap, future::Future, time::Instant};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GraphStatus {
    Off,
    Ok,
    TimedOut,
    #[default]
    Skipped,
}
impl GraphStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Ok => "ok",
            Self::TimedOut => "timed_out",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct RetrievalUsage {
    pub phases: BTreeMap<&'static str, f64>,
    pub graph_status: GraphStatus,
}
tokio::task_local! { static USAGE: RefCell<RetrievalUsage>; }

/// Scope observations to this future, including errors, without sharing across RPCs.
pub async fn capture<T>(future: impl Future<Output = T>) -> (T, RetrievalUsage) {
    USAGE
        .scope(RefCell::new(RetrievalUsage::default()), async {
            let result = future.await;
            (result, current())
        })
        .await
}
pub fn current() -> RetrievalUsage {
    USAGE
        .try_with(|usage| usage.borrow().clone())
        .unwrap_or_default()
}
pub fn graph_status(status: GraphStatus) {
    let _ = USAGE.try_with(|usage| usage.borrow_mut().graph_status = status);
}

/// Only these phases can be recorded, never a query/ID/free-form label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Embedding,
    Lexical,
    VectorFusion,
    GraphQueryCandidates,
    GraphEntityMatching,
    GraphMentions,
    GraphSeedEvidence,
    GraphEdgeExpansion,
    GraphNoteHydration,
    GraphProvenance,
    ContextPacking,
}
impl Phase {
    pub fn field(self) -> &'static str {
        match self {
            Self::Embedding => "embedding_ms",
            Self::Lexical => "lexical_ms",
            Self::VectorFusion => "vector_fusion_ms",
            Self::GraphQueryCandidates => "graph_query_candidates_ms",
            Self::GraphEntityMatching => "graph_entity_matching_ms",
            Self::GraphMentions => "graph_mentions_ms",
            Self::GraphSeedEvidence => "graph_seed_evidence_ms",
            Self::GraphEdgeExpansion => "graph_edge_expansion_ms",
            Self::GraphNoteHydration => "graph_note_hydration_ms",
            Self::GraphProvenance => "graph_provenance_ms",
            Self::ContextPacking => "context_packing_ms",
        }
    }
}

/// The same clock feeds debug events and usage. Drop records partial elapsed
/// time when a phase errors or is cancelled; repeated phases are summed.
pub struct PhaseTimer {
    started: Instant,
    phase: Phase,
}
impl PhaseTimer {
    pub fn start(phase: Phase) -> Self {
        Self {
            started: Instant::now(),
            phase,
        }
    }
    pub fn elapsed(&self) -> std::time::Duration {
        self.started.elapsed()
    }
}
impl Drop for PhaseTimer {
    fn drop(&mut self) {
        let elapsed = self.elapsed().as_secs_f64() * 1000.0;
        let _ = USAGE.try_with(|usage| {
            *usage
                .borrow_mut()
                .phases
                .entry(self.phase.field())
                .or_default() += elapsed;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// Pins cancellation ownership of the locked SurrealDB buffered local
    /// path. A detached engine task would execute the late write after timeout.
    #[tokio::test]
    async fn dropping_buffered_local_query_stops_execution_and_releases_transaction() {
        let db = crate::init_memory().await.unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_millis(10), async {
            db.query("SLEEP 100ms; CREATE note:late_after_cancel SET content = 'must not execute';")
                .await
        })
        .await;
        assert!(result.is_err());
        // Subsequent writers/readers succeed immediately on the same connection.
        db.query("CREATE note:after_cancel SET content = 'recovered';")
            .await
            .unwrap()
            .check()
            .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        let rows: Vec<surrealdb::types::Value> = db
            .query("SELECT * FROM note:late_after_cancel")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert!(
            rows.is_empty(),
            "query continued executing after its future was dropped"
        );
        let rows: Vec<surrealdb::types::Value> = db
            .query("SELECT * FROM note:after_cancel")
            .await
            .unwrap()
            .take(0)
            .unwrap();
        assert_eq!(rows.len(), 1);
    }

    #[tokio::test]
    async fn observations_are_request_local_and_record_partial_work() {
        let (_, usage) = capture(async {
            {
                let _timer = PhaseTimer::start(Phase::Embedding);
                tokio::task::yield_now().await;
            }
            {
                let _timer = PhaseTimer::start(Phase::Embedding);
            }
            graph_status(GraphStatus::TimedOut);
        })
        .await;
        assert_eq!(usage.phases.len(), 1);
        assert!(usage.phases["embedding_ms"] >= 0.0);
        assert_eq!(usage.graph_status, GraphStatus::TimedOut);
        assert!(current().phases.is_empty());
        let (a, b) = tokio::join!(
            capture(async {
                let _timer = PhaseTimer::start(Phase::Lexical);
                tokio::task::yield_now().await;
            }),
            capture(async {
                let _timer = PhaseTimer::start(Phase::ContextPacking);
                tokio::task::yield_now().await;
            })
        );
        assert_eq!(
            a.1.phases.keys().copied().collect::<Vec<_>>(),
            ["lexical_ms"]
        );
        assert_eq!(
            b.1.phases.keys().copied().collect::<Vec<_>>(),
            ["context_packing_ms"]
        );
    }
}
