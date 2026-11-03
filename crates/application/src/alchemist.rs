//! Opt-in, read-only Alchemist synthesis behind the application boundary.
//!
//! Requests are validated against operator ceilings (requests may only
//! tighten them), run under one whole-request deadline and return a receipt
//! that echoes the resolved budgets and policy. Nothing is persisted.

use crate::{ActionCancellation, ApplicationError, EmbeddedApplication, GraphPolicy, Scope};
use graphrag_agents::alchemist::{
    self, AlchemistError, CitationPolicy, SharedSynthesisGenerator, Synthesis, SynthesisBudgets,
    SynthesisInput, SynthesisPlan, SynthesisUsage, VerificationReport, MIN_QUOTE_CHARS,
    SYNTHESIS_CONTRACT,
};
use graphrag_agents::{AugmentOptions, GraphMode, SearchAgent, SearchScope};
use graphrag_db::Repository;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Maximum characters in a seed record ID.
pub const MAX_SYNTHESIS_SEED_CHARS: usize = 512;

fn default_scope() -> Scope {
    Scope::Notes
}

fn default_graph() -> GraphPolicy {
    GraphPolicy::Auto
}

/// One read-only synthesis. Supply exactly one of `query` or `seed`. Omitted
/// budgets use the server's configured values, which are also ceilings.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SynthesisRequest {
    /// Question answered from `build_context` retrieval (1–1024 characters).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub query: Option<String>,
    /// Canonical `note:` ID whose accepted-edge neighbourhood is explained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<String>,
    /// Query mode only (default `notes`).
    #[serde(default = "default_scope")]
    pub scope: Scope,
    /// Query mode only (default `auto`).
    #[serde(default = "default_graph")]
    pub graph: GraphPolicy,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 32))]
    pub max_chunks: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 32768))]
    pub max_context_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 8192))]
    pub max_output_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 300000))]
    pub deadline_ms: Option<u64>,
    /// Seed mode only: accepted-edge hops, at most 2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(range(min = 1, max = 2))]
    pub max_hops: Option<usize>,
    /// `drop` (default) or `strict`; a server configured strict stays strict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub citation_policy: Option<CitationPolicy>,
}

impl SynthesisRequest {
    fn empty() -> Self {
        Self {
            query: None,
            seed: None,
            scope: default_scope(),
            graph: default_graph(),
            max_chunks: None,
            max_context_tokens: None,
            max_output_tokens: None,
            deadline_ms: None,
            max_hops: None,
            citation_policy: None,
        }
    }

    pub fn for_query(query: impl Into<String>) -> Self {
        Self {
            query: Some(query.into()),
            ..Self::empty()
        }
    }

    pub fn for_seed(seed: impl Into<String>) -> Self {
        Self {
            seed: Some(seed.into()),
            ..Self::empty()
        }
    }
}

/// The fixed guarantees of this contract, echoed so a client can check them.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SynthesisPolicy {
    pub citation_policy: CitationPolicy,
    pub min_quote_chars: usize,
    /// Only accepted (reviewed) edges are read; proposals never are.
    pub relations: String,
    pub writes: String,
    pub persisted: bool,
    pub feeds_retrieval: bool,
    pub evidence_assessment: String,
    pub confidence: String,
}

impl SynthesisPolicy {
    fn new(citation_policy: CitationPolicy) -> Self {
        Self {
            citation_policy,
            min_quote_chars: MIN_QUOTE_CHARS,
            relations: "accepted_only".into(),
            writes: "none".into(),
            persisted: false,
            feeds_retrieval: false,
            evidence_assessment: "absent: accepted edges are reported as unassessed".into(),
            confidence:
                "none: strength is a lineage-deduplicated origin count, never a probability".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SynthesisReceipt {
    pub contract: String,
    pub mode: String,
    pub query: Option<String>,
    pub seed: Option<String>,
    pub scope: Option<Scope>,
    pub graph: Option<GraphPolicy>,
    /// Budgets this request actually ran under.
    pub budgets: SynthesisBudgets,
    /// Operator ceilings (and defaults) from `[alchemist]`.
    pub ceilings: SynthesisBudgets,
    pub policy: SynthesisPolicy,
    pub usage: SynthesisUsage,
    pub verification: VerificationReport,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct SynthesisResponse {
    pub synthesis: Synthesis,
    pub receipt: SynthesisReceipt,
}

#[derive(Debug, thiserror::Error)]
pub enum SynthesisError {
    #[error("Alchemist synthesis is disabled on this server; the owner must opt in with [alchemist] enabled = true.")]
    Disabled,
    #[error("Synthesis exceeded its {deadline_ms} ms deadline; nothing was returned or stored.")]
    DeadlineExceeded { deadline_ms: u64 },
    #[error("{reason}")]
    Unverifiable {
        reason: String,
        verification: Box<VerificationReport>,
    },
    #[error(transparent)]
    Application(#[from] ApplicationError),
}

impl SynthesisError {
    /// Stable machine-readable category for transports.
    pub fn code(&self) -> &'static str {
        match self {
            Self::Disabled => "alchemist_disabled",
            Self::DeadlineExceeded { .. } => "deadline_exceeded",
            Self::Unverifiable { .. } => "unverifiable_synthesis",
            Self::Application(error) => error.code(),
        }
    }
}

impl From<AlchemistError> for SynthesisError {
    fn from(error: AlchemistError) -> Self {
        match error {
            AlchemistError::Invalid(message) => ApplicationError::Validation(message).into(),
            AlchemistError::NotFound(message) => ApplicationError::NotFound(message).into(),
            AlchemistError::Unverifiable { reason, report } => Self::Unverifiable {
                reason,
                verification: report,
            },
            AlchemistError::Agent(error) => ApplicationError::from(error).into(),
        }
    }
}

/// Enabled synthesis: operator ceilings, default policy and the provider.
#[derive(Clone)]
pub struct AlchemistRuntime {
    pub ceilings: SynthesisBudgets,
    pub citation_policy: CitationPolicy,
    pub generator: SharedSynthesisGenerator,
}

/// Borrowed read-side dependencies of one synthesis.
pub struct SynthesisContext<'a> {
    pub repo: &'a Repository,
    pub search: &'a SearchAgent,
    pub augment: &'a AugmentOptions,
    /// `None` means the Alchemist is disabled (the default).
    pub runtime: Option<&'a AlchemistRuntime>,
}

fn pick<T: Copy + PartialOrd + From<u8> + std::fmt::Display>(
    name: &str,
    requested: Option<T>,
    ceiling: T,
) -> Result<T, ApplicationError> {
    match requested {
        None => Ok(ceiling),
        Some(value) if value >= T::from(1) && value <= ceiling => Ok(value),
        Some(_) => Err(ApplicationError::Validation(format!(
            "{name} must be between 1 and the server ceiling {ceiling}"
        ))),
    }
}

/// Validate a request and resolve it against the operator's ceilings.
pub fn resolve_synthesis(
    runtime: &AlchemistRuntime,
    request: &SynthesisRequest,
) -> Result<SynthesisPlan, ApplicationError> {
    let input = match (&request.query, &request.seed) {
        (Some(query), None) => {
            if query.trim().is_empty()
                || query.chars().count() > crate::MAX_REMOTE_QUERY_CHARS
                || query.contains('\0')
            {
                return Err(ApplicationError::Validation(
                    "query must contain 1–1024 characters".into(),
                ));
            }
            SynthesisInput::Query {
                query: query.clone(),
                scope: match request.scope {
                    Scope::Notes => SearchScope::Notes,
                    Scope::Messages => SearchScope::Messages,
                    Scope::All => SearchScope::All,
                },
                graph: match request.graph {
                    GraphPolicy::Off => GraphMode::Off,
                    GraphPolicy::Auto => GraphMode::Auto,
                    GraphPolicy::On => GraphMode::On,
                },
            }
        }
        (None, Some(seed)) => {
            let seed = seed.trim();
            if seed.is_empty() || seed.chars().count() > MAX_SYNTHESIS_SEED_CHARS {
                return Err(ApplicationError::Validation(
                    "seed must be a canonical note:ID of at most 512 characters".into(),
                ));
            }
            SynthesisInput::Seed { id: seed.into() }
        }
        _ => {
            return Err(ApplicationError::Validation(
                "supply exactly one of query or seed".into(),
            ))
        }
    };
    let ceilings = runtime.ceilings;
    let budgets = SynthesisBudgets {
        max_chunks: pick("max_chunks", request.max_chunks, ceilings.max_chunks)?,
        max_context_tokens: pick(
            "max_context_tokens",
            request.max_context_tokens,
            ceilings.max_context_tokens,
        )?,
        max_output_tokens: pick(
            "max_output_tokens",
            request.max_output_tokens,
            ceilings.max_output_tokens,
        )?,
        deadline_ms: pick("deadline_ms", request.deadline_ms, ceilings.deadline_ms)?,
        max_hops: pick("max_hops", request.max_hops, ceilings.max_hops)?,
    };
    budgets.validate().map_err(ApplicationError::Validation)?;
    let policy = match (runtime.citation_policy, request.citation_policy) {
        (CitationPolicy::Strict, Some(CitationPolicy::Drop)) => {
            return Err(ApplicationError::Validation(
                "this server requires the strict citation policy".into(),
            ))
        }
        (_, Some(requested)) => requested,
        (configured, None) => configured,
    };
    Ok(SynthesisPlan {
        input,
        budgets,
        policy,
    })
}

/// Run one read-only synthesis under its whole-request deadline. Shared by
/// the CLI (local) and the service (`synthesize_notes`).
pub async fn synthesize_notes(
    context: SynthesisContext<'_>,
    request: SynthesisRequest,
    cancellation: ActionCancellation,
) -> Result<SynthesisResponse, SynthesisError> {
    let runtime = context.runtime.ok_or(SynthesisError::Disabled)?;
    let plan = resolve_synthesis(runtime, &request)?;
    if cancellation.is_cancelled() {
        return Err(ApplicationError::Cancelled.into());
    }
    let deadline_ms = plan.budgets.deadline_ms;
    let work = alchemist::synthesize(
        context.repo,
        context.search,
        runtime.generator.as_ref(),
        context.augment,
        &plan,
    );
    // Dropping the future on expiry or cancellation is safe: nothing is
    // written, so there is no partial state to roll back.
    let outcome = tokio::select! {
        biased;
        _ = cancellation.cancelled() => return Err(ApplicationError::Cancelled.into()),
        result = tokio::time::timeout(Duration::from_millis(deadline_ms), work) => match result {
            Err(_) => return Err(SynthesisError::DeadlineExceeded { deadline_ms }),
            Ok(result) => result?,
        },
    };
    let (mode, query, seed, scope, graph) = match &plan.input {
        SynthesisInput::Query { query, .. } => (
            "query",
            Some(query.clone()),
            None,
            Some(request.scope),
            Some(request.graph),
        ),
        SynthesisInput::Seed { id } => ("seed", None, Some(id.clone()), None, None),
    };
    Ok(SynthesisResponse {
        synthesis: outcome.synthesis,
        receipt: SynthesisReceipt {
            contract: SYNTHESIS_CONTRACT.into(),
            mode: mode.into(),
            query,
            seed,
            scope,
            graph,
            budgets: plan.budgets,
            ceilings: runtime.ceilings,
            policy: SynthesisPolicy::new(plan.policy),
            usage: outcome.usage,
            verification: outcome.verification,
        },
    })
}

impl EmbeddedApplication {
    /// Enable read-only synthesis. Without this the Alchemist is disabled.
    pub fn with_alchemist(mut self, runtime: AlchemistRuntime) -> Self {
        self.alchemist = Some(runtime);
        self
    }

    pub(crate) async fn synthesize_notes_impl(
        &self,
        request: SynthesisRequest,
        cancellation: ActionCancellation,
    ) -> Result<SynthesisResponse, SynthesisError> {
        let runtime = self.alchemist.as_ref().ok_or(SynthesisError::Disabled)?;
        let plan = resolve_synthesis(runtime, &request)?;
        if matches!(plan.input, SynthesisInput::Query { .. }) {
            self.require_providers(false, &cancellation).await?;
        }
        synthesize_notes(
            SynthesisContext {
                repo: &self.repo,
                search: &self.search,
                augment: &self.augment_options,
                runtime: Some(runtime),
            },
            request,
            cancellation,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RemoteApplicationOperations;
    use graphrag_agents::alchemist::FixtureSynthesisGenerator;
    use graphrag_agents::{DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig};
    use std::sync::Arc;

    fn runtime(generator: FixtureSynthesisGenerator, policy: CitationPolicy) -> AlchemistRuntime {
        AlchemistRuntime {
            ceilings: SynthesisBudgets {
                max_chunks: 4,
                max_context_tokens: 2_000,
                max_output_tokens: 512,
                deadline_ms: 2_000,
                max_hops: 1,
            },
            citation_policy: policy,
            generator: Arc::new(generator),
        }
    }

    async fn application() -> (EmbeddedApplication, String) {
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        let note = repo
            .create_note(graphrag_core::Note::new(
                "Fictional beehive log: the north hive swarmed on a warm afternoon.",
            ))
            .await
            .unwrap();
        let embedder: graphrag_agents::SharedEmbedder = Arc::new(DeterministicEmbedder::default());
        let application = EmbeddedApplication::new(
            repo.clone(),
            SearchAgent::new(repo, embedder.clone()),
            embedder,
            Arc::new(FixtureEntityExtractor::default()),
            LibrarianRuntimeConfig::default(),
        );
        (
            application,
            graphrag_core::record_id_to_string(note.id.as_ref().unwrap()),
        )
    }

    #[test]
    fn requests_may_only_tighten_operator_ceilings() {
        let drop = runtime(FixtureSynthesisGenerator::fixed("{}"), CitationPolicy::Drop);
        let plan = resolve_synthesis(&drop, &SynthesisRequest::for_seed("note:a")).unwrap();
        assert_eq!(plan.budgets, drop.ceilings);
        assert_eq!(plan.policy, CitationPolicy::Drop);
        let mut request = SynthesisRequest::for_seed("note:a");
        request.max_chunks = Some(2);
        request.deadline_ms = Some(50);
        request.citation_policy = Some(CitationPolicy::Strict);
        let plan = resolve_synthesis(&drop, &request).unwrap();
        assert_eq!((plan.budgets.max_chunks, plan.budgets.deadline_ms), (2, 50));
        assert_eq!(plan.policy, CitationPolicy::Strict);
        for invalid in [
            SynthesisRequest {
                max_chunks: Some(5),
                ..SynthesisRequest::for_seed("note:a")
            },
            SynthesisRequest {
                max_hops: Some(2),
                ..SynthesisRequest::for_seed("note:a")
            },
            SynthesisRequest {
                deadline_ms: Some(0),
                ..SynthesisRequest::for_seed("note:a")
            },
            SynthesisRequest {
                max_context_tokens: Some(2_001),
                ..SynthesisRequest::for_seed("note:a")
            },
            SynthesisRequest {
                seed: Some("note:a".into()),
                ..SynthesisRequest::for_query("both")
            },
            SynthesisRequest::for_query("   "),
            SynthesisRequest::for_seed(""),
            SynthesisRequest {
                query: None,
                ..SynthesisRequest::for_query("neither")
            },
        ] {
            assert!(
                matches!(
                    resolve_synthesis(&drop, &invalid),
                    Err(ApplicationError::Validation(_))
                ),
                "{invalid:?}"
            );
        }
        let strict = runtime(
            FixtureSynthesisGenerator::fixed("{}"),
            CitationPolicy::Strict,
        );
        let loosen = SynthesisRequest {
            citation_policy: Some(CitationPolicy::Drop),
            ..SynthesisRequest::for_seed("note:a")
        };
        assert!(resolve_synthesis(&strict, &loosen).is_err());
        assert_eq!(
            resolve_synthesis(&strict, &SynthesisRequest::for_seed("note:a"))
                .unwrap()
                .policy,
            CitationPolicy::Strict
        );
    }

    #[tokio::test]
    async fn disabled_by_default_without_any_provider_call() {
        let (application, seed) = application().await;
        let result = application
            .synthesize_notes(SynthesisRequest::for_seed(seed), ActionCancellation::new())
            .await;
        assert!(matches!(result, Err(SynthesisError::Disabled)));
        assert_eq!(
            SynthesisError::Disabled.code(),
            "alchemist_disabled",
            "transports report a stable category"
        );
    }

    #[tokio::test]
    async fn the_deadline_bounds_the_whole_request_and_returns_nothing() {
        let (application, seed) = application().await;
        let draft = serde_json::json!({"claims": [], "conflicts": [], "open_questions": []});
        let slow =
            FixtureSynthesisGenerator::fixed(draft.to_string()).with_delay(Duration::from_secs(5));
        let application = application.with_alchemist(runtime(slow.clone(), CitationPolicy::Drop));
        let started = std::time::Instant::now();
        let request = SynthesisRequest {
            deadline_ms: Some(60),
            ..SynthesisRequest::for_seed(seed.clone())
        };
        let result = application
            .synthesize_notes(request, ActionCancellation::new())
            .await;
        assert!(matches!(
            result,
            Err(SynthesisError::DeadlineExceeded { deadline_ms: 60 })
        ));
        assert!(started.elapsed() < Duration::from_secs(2));
        assert_eq!(slow.requests().len(), 1);

        let cancelled = ActionCancellation::new();
        cancelled.cancel();
        assert!(matches!(
            application
                .synthesize_notes(SynthesisRequest::for_seed(seed), cancelled)
                .await,
            Err(SynthesisError::Application(ApplicationError::Cancelled))
        ));
    }
}
