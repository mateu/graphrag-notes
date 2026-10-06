//! Bounded diagnostics over resources already owned by the service.
use crate::{ApplicationResult, EmbeddedApplication};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ReadinessState {
    Ready,
    Unavailable,
    Unknown,
    NotConfigured,
    Stale,
    Partial,
    Forbidden,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Readiness {
    pub state: ReadinessState,
    pub summary: String,
    pub next_action: Option<String>,
    /// Unix seconds of a previous observation, never an implicit active probe.
    pub observed_at: Option<u64>,
}
impl Readiness {
    pub fn new(state: ReadinessState, summary: &str, next_action: Option<&str>) -> Self {
        Self {
            state,
            summary: summary.into(),
            next_action: next_action.map(str::to_owned),
            observed_at: None,
        }
    }
    pub fn unknown(summary: &str, next_action: &str) -> Self {
        Self::new(ReadinessState::Unknown, summary, Some(next_action))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ApplicationReadiness {
    pub storage: Readiness,
    pub embeddings: Readiness,
    pub extraction: Readiness,
    pub sources: Readiness,
    pub backup: Readiness,
}
impl Default for ApplicationReadiness {
    fn default() -> Self {
        Self {
            storage: Readiness::unknown("Storage readiness evidence is unavailable.", "Ask the service owner to check the corpus."),
            embeddings: Readiness::unknown("No cached embedding-provider health observation.", "Use explicit keyword search, or ask the owner for a provider check."),
            extraction: Readiness::unknown("No cached extraction-provider health observation.", "Ask the owner to check extraction before starting a job."),
            sources: Readiness::unknown("No registered collection refresh evidence.", "Supply a bound client refresh status file or run the collection refresh status command."),
            backup: Readiness::unknown("No recorded backup evidence is available through this service.", "Ask the owner to verify the latest backup and restore evidence."),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct OwnedJobReadiness {
    pub readiness: Readiness,
    pub sampled: usize,
    pub limit: usize,
    /// A bounded recent sample, never a whole-corpus job count.
    pub counts: std::collections::BTreeMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ServiceReadiness {
    pub schema_version: u32,
    pub read_only: bool,
    pub inference_probed: bool,
    pub instance_id: String,
    pub application: ApplicationReadiness,
    pub jobs: OwnedJobReadiness,
}

#[derive(Default)]
pub(crate) struct ProviderObservations {
    pub embeddings: Option<(bool, SystemTime)>,
    pub extraction: Option<(bool, SystemTime)>,
}

fn cached(observation: Option<(bool, SystemTime)>, configured: bool) -> Readiness {
    if !configured {
        return Readiness::new(
            ReadinessState::NotConfigured,
            "Provider configuration is absent.",
            Some("Ask the owner to configure this provider."),
        );
    }
    let Some((available, when)) = observation else {
        return Readiness::unknown("Provider is configured; current availability is unknown.", "Routine status does not contact providers. Use keyword search or request an explicit owner-side provider check.");
    };
    let state = if when.elapsed().unwrap_or(Duration::MAX) > Duration::from_secs(60) {
        ReadinessState::Stale
    } else if available {
        ReadinessState::Ready
    } else {
        ReadinessState::Unavailable
    };
    let mut result = Readiness::new(state, "Cached provider health observation; inference and model compatibility are not verified by this report.", Some("Use keyword search when embedding availability is uncertain; ask the owner to check providers."));
    result.observed_at = when
        .duration_since(UNIX_EPOCH)
        .ok()
        .map(|duration| duration.as_secs());
    result
}

impl EmbeddedApplication {
    pub(crate) async fn readonly_readiness(&self) -> ApplicationResult<ApplicationReadiness> {
        let storage =
            match tokio::time::timeout(Duration::from_secs(3), self.repo.check_readiness()).await {
                Ok(Ok(())) => Readiness::new(
                    ReadinessState::Ready,
                    "The owning service completed a bounded corpus read.",
                    None,
                ),
                _ => Readiness::new(
                    ReadinessState::Unavailable,
                    "The owning service could not complete the corpus read.",
                    Some("Ask the owner to inspect storage; do not open another database process."),
                ),
            };
        let observations = self
            .provider_observations
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        Ok(ApplicationReadiness {
            storage,
            embeddings: cached(
                observations.embeddings,
                !self.embedder.capabilities().model.is_empty(),
            ),
            extraction: cached(
                observations.extraction,
                !self.extractor.capabilities().model.is_empty(),
            ),
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn configured_is_not_ready_and_old_observations_are_stale() {
        assert_eq!(cached(None, true).state, ReadinessState::Unknown);
        assert_eq!(cached(None, false).state, ReadinessState::NotConfigured);
        assert_eq!(
            cached(Some((false, SystemTime::now())), true).state,
            ReadinessState::Unavailable
        );
        assert_eq!(
            cached(
                Some((true, SystemTime::now() - Duration::from_secs(61))),
                true
            )
            .state,
            ReadinessState::Stale
        );
    }
}
