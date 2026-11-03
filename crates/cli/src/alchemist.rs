//! `graphrag synthesize`: opt-in, read-only Alchemist synthesis, run locally
//! or through a server's `synthesize_notes` tool. Nothing is written.

use crate::cli::{Cli, Commands, GraphModeArg, SearchScopeArg};
use crate::output::{self, safe_text, OutputFormat};
use anyhow::{Context, Result};
use clap::ValueEnum;
use graphrag_agents::alchemist::{
    CitationPolicy, ConfiguredSynthesisGenerator, ConflictKind, SynthesisBudgets,
};
use graphrag_agents::SharedEmbedder;
use graphrag_application::{
    ActionCancellation, AlchemistRuntime, ApplicationError, GraphPolicy, Scope, SynthesisContext,
    SynthesisError, SynthesisPreflight, SynthesisRequest, SynthesisResponse,
};
use graphrag_config::{AlchemistCitationPolicy, RuntimeConfig};
use graphrag_db::Repository;
use std::io::{self, Write};
use std::sync::Arc;

pub(crate) const TOOL: &str = "synthesize_notes";

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
pub(crate) enum CitationPolicyArg {
    /// Drop claims whose citations do not verify, and count them
    Drop,
    /// Fail if any citation does not verify
    Strict,
}

/// The runtime of an enabled `[alchemist]` section, or `None` (the default).
pub(crate) fn runtime(config: &RuntimeConfig) -> Option<AlchemistRuntime> {
    let settings = &config.alchemist;
    settings.enabled.then(|| AlchemistRuntime {
        ceilings: SynthesisBudgets {
            max_chunks: settings.max_chunks,
            max_context_tokens: settings.max_context_tokens,
            max_output_tokens: settings.max_output_tokens,
            deadline_ms: settings.deadline_ms,
            max_hops: settings.max_hops,
        },
        citation_policy: match settings.citation_policy {
            AlchemistCitationPolicy::Drop => CitationPolicy::Drop,
            AlchemistCitationPolicy::Strict => CitationPolicy::Strict,
        },
        generator: Arc::new(ConfiguredSynthesisGenerator::from_config(
            &crate::app::inference_provider_config(config),
        )),
    })
}

/// The service request for `graphrag synthesize`, shared by both modes.
pub(crate) fn request(command: &Commands) -> Option<(SynthesisRequest, OutputFormat)> {
    let Commands::Synthesize {
        query,
        seed,
        scope,
        graph,
        max_chunks,
        max_tokens,
        max_output_tokens,
        deadline_ms,
        hops,
        policy,
        format,
    } = command
    else {
        return None;
    };
    // Clap admits exactly one; the application re-validates either way.
    let mut request = SynthesisRequest::for_query(String::new());
    request.query = query.clone();
    request.seed = seed.clone();
    request.scope = match scope {
        SearchScopeArg::Notes => Scope::Notes,
        SearchScopeArg::Messages => Scope::Messages,
        SearchScopeArg::All => Scope::All,
    };
    request.graph = match graph {
        GraphModeArg::Off => GraphPolicy::Off,
        GraphModeArg::Auto => GraphPolicy::Auto,
        GraphModeArg::On => GraphPolicy::On,
    };
    request.max_chunks = *max_chunks;
    request.max_context_tokens = *max_tokens;
    request.max_output_tokens = *max_output_tokens;
    request.deadline_ms = *deadline_ms;
    request.max_hops = *hops;
    request.citation_policy = policy.map(|policy| match policy {
        CitationPolicyArg::Drop => CitationPolicy::Drop,
        CitationPolicyArg::Strict => CitationPolicy::Strict,
    });
    Some((request, *format))
}

fn into_error(error: SynthesisError) -> anyhow::Error {
    let message = error.to_string();
    match error {
        SynthesisError::Application(error) => error.into(),
        SynthesisError::Disabled => ApplicationError::Compatibility(message).into(),
        SynthesisError::DeadlineExceeded { .. } => {
            ApplicationError::ProviderUnavailable(message).into()
        }
        SynthesisError::Unverifiable { .. } => ApplicationError::Internal(message).into(),
    }
}

pub(crate) async fn run_local(
    repo: Repository,
    embedder: SharedEmbedder,
    config: &RuntimeConfig,
    command: &Commands,
) -> Result<()> {
    let (request, format) = request(command).context("not a synthesize command")?;
    let runtime = runtime(config);
    let search =
        crate::app::configured_search_agent(repo.clone(), embedder.clone(), &config.search);
    // Query mode embeds the question. Its readiness check runs inside the
    // synthesis deadline, never during bootstrap, and only once enabled.
    let preflight = request.query.is_some().then(|| {
        Box::pin(async move {
            if embedder.health().await.unwrap_or(false) {
                Ok(())
            } else {
                Err(ApplicationError::ProviderUnavailable(
                    "Embeddings service unavailable; seed mode (--seed) needs no embeddings."
                        .into(),
                ))
            }
        }) as SynthesisPreflight<'static>
    });
    let augment = crate::app::augment_options(
        config.augment.default_limit,
        config.augment.max_tokens,
        config.augment.max_chunk_tokens,
        &config.augment,
    );
    let response = graphrag_application::synthesize_notes(
        SynthesisContext {
            repo: &repo,
            search: &search,
            augment: &augment,
            runtime: runtime.as_ref(),
            preflight,
        },
        request,
        ActionCancellation::new(),
    )
    .await
    .map_err(into_error)?;
    print(format, &response)
}

pub(crate) async fn run_remote(cli: &Cli, server: &str) -> Result<()> {
    let (request, format) = request(&cli.command).context("not a synthesize command")?;
    let value =
        crate::remote::call_tool(cli, server, TOOL, serde_json::to_value(&request)?).await?;
    let response: SynthesisResponse =
        serde_json::from_value(value["data"].clone()).map_err(|_| {
            ApplicationError::Compatibility("incompatible synthesize_notes result contract".into())
        })?;
    print(format, &response)
}

fn print(format: OutputFormat, response: &SynthesisResponse) -> Result<()> {
    output::print(format, "synthesize", response, |writer| {
        render(writer, response)
    })
}

fn short(revision: &str) -> &str {
    revision.get(..12).unwrap_or(revision)
}

/// Human rendering. Every corpus- or model-supplied string is escaped.
pub(crate) fn render(writer: &mut dyn Write, response: &SynthesisResponse) -> io::Result<()> {
    let (synthesis, receipt) = (&response.synthesis, &response.receipt);
    let subject = match (&receipt.query, &receipt.seed) {
        (Some(query), _) => format!("query {:?}", safe_text(query, false)),
        (_, Some(seed)) => format!("seed {}", safe_text(seed, false)),
        _ => String::new(),
    };
    writeln!(
        writer,
        "Alchemist synthesis ({subject}): read-only; not stored; not evidence"
    )?;
    writeln!(writer, "\nClaims")?;
    if synthesis.claims.is_empty() {
        writeln!(writer, "  (none verified)")?;
    }
    for (index, claim) in synthesis.claims.iter().enumerate() {
        writeln!(writer, "  {}. {}", index + 1, safe_text(&claim.text, false))?;
        writeln!(writer, "     {}", safe_text(&claim.strength.wording, false))?;
        for citation in &claim.citations {
            writeln!(
                writer,
                "     [{}] {} @ {} \"{}\"",
                citation.label,
                safe_text(&citation.chunk_id, false),
                short(&citation.revision),
                safe_text(&citation.span, false)
            )?;
        }
    }
    if !synthesis.conflicts.is_empty() {
        writeln!(writer, "\nConflicts (surfaced, not resolved)")?;
        for conflict in &synthesis.conflicts {
            let claims: Vec<String> = conflict
                .claims
                .iter()
                .map(|index| (index + 1).to_string())
                .collect();
            let chunks: Vec<String> = conflict
                .chunks
                .iter()
                .map(|chunk| safe_text(chunk, false))
                .collect();
            match conflict.kind {
                ConflictKind::AcceptedContradictsEdge => writeln!(
                    writer,
                    "  - accepted contradicts edge {} between {}; claims [{}]",
                    safe_text(conflict.edge_id.as_deref().unwrap_or("?"), false),
                    chunks.join(" and "),
                    claims.join(", ")
                )?,
                ConflictKind::ConflictingClaims => writeln!(
                    writer,
                    "  - conflicting claims [{}]: {}",
                    claims.join(", "),
                    safe_text(conflict.topic.as_deref().unwrap_or(""), false)
                )?,
            }
        }
    }
    if !synthesis.relation_chains.is_empty() {
        writeln!(
            writer,
            "\nRelation chains (accepted edges; derived_from lineage collapsed)"
        )?;
        for chain in &synthesis.relation_chains {
            let steps: Vec<String> = chain
                .steps
                .iter()
                .map(|step| {
                    format!(
                        "{} -{}-> {} [{}]",
                        step.from_origin, step.edge_type, step.to_origin, step.evidence
                    )
                })
                .collect();
            writeln!(
                writer,
                "  to {}: {} ({} lineage step(s) collapsed)",
                safe_text(&chain.target_chunk, false),
                safe_text(&steps.join("; "), false),
                chain.collapsed_lineage_steps
            )?;
        }
    }
    if !synthesis.timeline.is_empty() {
        writeln!(writer, "\nTimeline (note dates, not event dates)")?;
        for entry in &synthesis.timeline {
            writeln!(
                writer,
                "  {}  claim {}  {}",
                safe_text(&entry.note_date, false),
                entry.claim + 1,
                safe_text(&entry.chunk_id, false)
            )?;
        }
    }
    if !synthesis.open_questions.is_empty() {
        writeln!(writer, "\nOpen questions (model-generated, unverified)")?;
        for question in &synthesis.open_questions {
            writeln!(writer, "  - {}", safe_text(question, false))?;
        }
    }
    if !synthesis.gaps.is_empty() {
        writeln!(writer, "\nGaps")?;
        for gap in &synthesis.gaps {
            writeln!(writer, "  - {}", safe_text(&gap.detail, false))?;
        }
    }
    let (budgets, ceilings) = (&receipt.budgets, &receipt.ceilings);
    let (usage, verification) = (&receipt.usage, &receipt.verification);
    writeln!(writer, "\nReceipt")?;
    writeln!(
        writer,
        "  contract {}, mode {}, citation policy {}, relations {}, writes {}, persisted {}, feeds retrieval {}",
        receipt.contract,
        receipt.mode,
        receipt.policy.citation_policy.as_str(),
        receipt.policy.relations,
        receipt.policy.writes,
        receipt.policy.persisted,
        receipt.policy.feeds_retrieval
    )?;
    writeln!(
        writer,
        "  budgets: max_chunks={} max_context_tokens={} max_output_tokens={} deadline_ms={} max_hops={} (ceilings {}/{}/{}/{}/{})",
        budgets.max_chunks,
        budgets.max_context_tokens,
        budgets.max_output_tokens,
        budgets.deadline_ms,
        budgets.max_hops,
        ceilings.max_chunks,
        ceilings.max_context_tokens,
        ceilings.max_output_tokens,
        ceilings.deadline_ms,
        ceilings.max_hops
    )?;
    writeln!(
        writer,
        "  usage: {} chunk(s) used of {} considered, {} context tokens, {} relation(s), provider {}/{}, {} ms",
        usage.chunks_used,
        usage.chunks_considered,
        usage.context_tokens,
        usage.relations_considered,
        safe_text(&usage.provider, false),
        safe_text(&usage.model, false),
        usage.elapsed_ms
    )?;
    let rejections: Vec<String> = verification
        .rejections
        .iter()
        .map(|(reason, count)| format!("{reason}={count}"))
        .collect();
    writeln!(
        writer,
        "  verification: {} of {} claim(s) kept, {} dropped ({} uncited); {} citation(s) rejected [{}]; {} chunk(s) re-checked at their pinned revision",
        verification.claims_kept,
        verification.claims_proposed,
        verification.claims_dropped,
        verification.uncited_claims,
        verification.citations_rejected,
        rejections.join(", "),
        verification.chunks_rechecked
    )?;
    writeln!(writer, "  evidence: {}", receipt.policy.evidence_assessment)?;
    writeln!(writer, "  strength: {}", receipt.policy.confidence)
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use graphrag_agents::alchemist::FixtureSynthesisGenerator;
    use graphrag_agents::DeterministicEmbedder;

    fn parse(args: &[&str]) -> Commands {
        Cli::try_parse_from(args).unwrap().command
    }

    #[test]
    fn exactly_one_of_query_or_seed_is_accepted() {
        assert!(Cli::try_parse_from(["graphrag", "synthesize"]).is_err());
        assert!(Cli::try_parse_from(["graphrag", "synthesize", "q", "--seed", "note:a"]).is_err());
        let (request, format) = request(&parse(&[
            "graphrag",
            "synthesize",
            "--seed",
            "note:a",
            "--hops",
            "1",
            "--max-chunks",
            "4",
            "--policy",
            "strict",
            "--format",
            "json",
        ]))
        .unwrap();
        assert_eq!(request.seed.as_deref(), Some("note:a"));
        assert_eq!(request.query, None);
        assert_eq!((request.max_hops, request.max_chunks), (Some(1), Some(4)));
        assert_eq!(request.citation_policy, Some(CitationPolicy::Strict));
        assert_eq!(format, OutputFormat::Json);
        let (request, _) = request_for(&["graphrag", "synthesize", "what changed?"]);
        assert_eq!(request.query.as_deref(), Some("what changed?"));
        assert_eq!(
            (request.scope, request.graph),
            (Scope::Notes, GraphPolicy::Auto)
        );
    }

    fn request_for(args: &[&str]) -> (SynthesisRequest, OutputFormat) {
        request(&parse(args)).unwrap()
    }

    #[tokio::test]
    async fn local_synthesis_is_disabled_by_default_and_contacts_no_provider() {
        let config = RuntimeConfig::default();
        assert!(runtime(&config).is_none());
        let command = parse(&["graphrag", "synthesize", "orchard pressure"]);
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        let error = run_local(
            repo,
            Arc::new(DeterministicEmbedder::default().unhealthy()),
            &config,
            &command,
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("[alchemist] enabled = true"));
        assert_eq!(
            crate::app::exit_code_for(&error),
            output::ExitCode::Compatibility
        );
        let mut enabled = RuntimeConfig::default();
        enabled.alchemist.enabled = true;
        assert!(runtime(&enabled).is_some());
    }

    /// An embeddings provider whose readiness probe never answers in time.
    struct SlowHealth;

    #[async_trait::async_trait]
    impl graphrag_agents::Embedder for SlowHealth {
        async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
            unreachable!("readiness must time out first")
        }
        async fn embed_batch(
            &self,
            _: &[String],
            _: bool,
        ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
            unreachable!("readiness must time out first")
        }
        async fn health(&self) -> graphrag_agents::Result<bool> {
            tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            Ok(true)
        }
        fn capabilities(&self) -> graphrag_agents::InferenceCapabilities {
            DeterministicEmbedder::default().capabilities()
        }
    }

    #[tokio::test]
    async fn local_query_readiness_runs_inside_the_deadline() {
        let mut config = RuntimeConfig::default();
        config.alchemist.enabled = true;
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        let command = parse(&["graphrag", "synthesize", "orchard?", "--deadline-ms", "60"]);
        let started = std::time::Instant::now();
        let error = run_local(repo.clone(), Arc::new(SlowHealth), &config, &command)
            .await
            .unwrap_err();
        assert!(error.to_string().contains("60 ms deadline"), "{error:#}");
        assert!(started.elapsed() < std::time::Duration::from_secs(2));
        let unhealthy = run_local(
            repo,
            Arc::new(DeterministicEmbedder::default().unhealthy()),
            &config,
            &command,
        )
        .await
        .unwrap_err();
        assert!(unhealthy
            .to_string()
            .contains("Embeddings service unavailable"));
    }

    #[tokio::test]
    async fn human_rendering_marks_the_output_read_only_and_echoes_the_receipt() {
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        let note = repo
            .create_note(graphrag_core::Note::new(
                "Fictional field note: the orchard gauge read 2.1 bar at dawn.",
            ))
            .await
            .unwrap();
        let id = graphrag_core::record_id_to_string(note.id.as_ref().unwrap());
        let draft = serde_json::json!({
            "claims": [{"text": "The gauge read 2.1 bar.", "citations": [{"chunk": "C1", "quote": "the orchard gauge read 2.1 bar"}]}],
            "conflicts": [],
            "open_questions": ["Which gauge?\u{1b}[31m"]
        });
        let runtime = AlchemistRuntime {
            ceilings: SynthesisBudgets::default(),
            citation_policy: CitationPolicy::Drop,
            generator: Arc::new(FixtureSynthesisGenerator::fixed(draft.to_string())),
        };
        let search = graphrag_agents::SearchAgent::new(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
        );
        let augment = graphrag_agents::AugmentOptions::default();
        let response = graphrag_application::synthesize_notes(
            SynthesisContext {
                repo: &repo,
                search: &search,
                augment: &augment,
                runtime: Some(&runtime),
                preflight: None,
            },
            SynthesisRequest::for_seed(id.clone()),
            ActionCancellation::new(),
        )
        .await
        .unwrap();
        let mut rendered = Vec::new();
        render(&mut rendered, &response).unwrap();
        let rendered = String::from_utf8(rendered).unwrap();
        assert!(rendered.starts_with(&format!(
            "Alchemist synthesis (seed {id}): read-only; not stored; not evidence"
        )));
        assert!(rendered.contains("\"the orchard gauge read 2.1 bar\""));
        assert!(rendered.contains("Open questions (model-generated, unverified)"));
        assert!(!rendered.contains('\u{1b}'));
        assert!(rendered.contains("writes none, persisted false, feeds retrieval false"));
        assert!(rendered.contains("max_chunks=8 max_context_tokens=4000"));
        assert!(rendered.contains("1 of 1 claim(s) kept"));
    }
}
