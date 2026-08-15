//! GraphRAG Notes CLI
//!
//! A command-line interface for the GraphRAG Notes system.

use crate::{backup, commands, doctor, output};
use anyhow::{Context, Result};
use clap::Parser;
use graphrag_agents::{
    AugmentDiagnostics, AugmentOptions, GraphRetrievalConfig, InferenceProviderConfig,
    InferenceProviders, LibrarianRuntimeConfig, ProcessingConfig, ResilientEmbedder,
    ResilientEntityExtractor, SearchAgent, SearchScope, SharedEmbedder, SharedEntityExtractor,
    TokenCountMode,
};
use graphrag_config::{AugmentConfig, CliOverrides, RuntimeConfig, SearchConfig};
use graphrag_db::{
    fusion::{FusionConfig, FusionStrategy},
    init_memory, init_persistent, migrations, Repository,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;
use tracing::info;

/// Resources initialized for the selected command after bootstrap has applied
/// lazy database and provider requirements. This stays deliberately small: it
/// carries concrete interfaces, not a service locator.
pub(crate) struct AppContext {
    pub(crate) repo: Repository,
    pub(crate) config: RuntimeConfig,
    pub(crate) tei: SharedEmbedder,
    pub(crate) tgi: SharedEntityExtractor,
    pub(crate) librarian_config: LibrarianRuntimeConfig,
    pub(crate) cancellation_requested: Option<Arc<AtomicBool>>,
    pub(crate) prepared_notes_edit: Option<commands::notes::PreparedEdit>,
}
use tracing_subscriber::{EnvFilter, FmtSubscriber};

use crate::cli::{
    notes_edit_requires_inference, BackupCommand, Cli, Commands, ConfigCommand, DoctorFormat,
    JobsCommand, SourcesCommand,
};
use crate::dispatch::print_backup_summary;
fn inference_provider_config(config: &RuntimeConfig) -> InferenceProviderConfig {
    InferenceProviderConfig {
        embedding_provider: config.inference.embedding_provider.clone(),
        embedding_url: config.inference.embedding_url.clone(),
        embedding_model: config.inference.embedding_model.clone(),
        extraction_provider: config.inference.extraction_provider.clone(),
        extraction_url: config.inference.extraction_url.clone(),
        extraction_model: config.inference.extraction_model.clone(),
        timeout_secs: config.inference.timeout_secs,
        tei_max_batch: config.inference.tei_max_batch,
        tei_prompt_name_query: config.inference.tei_prompt_name_query.clone(),
        tei_prompt_name_passage: config.inference.tei_prompt_name_passage.clone(),
        strict_entity_json: config.inference.strict_entity_json,
        max_entities: config.inference.max_entities,
        max_relationships: config.inference.max_relationships,
        ollama_timeout_secs: config.inference.ollama_timeout_secs,
        ollama_options: config.inference.ollama_options.clone(),
    }
}

fn processing_config(
    config: &RuntimeConfig,
    concurrency: Option<usize>,
    retry_attempts: Option<usize>,
    no_cache: bool,
) -> Result<ProcessingConfig> {
    let concurrency = concurrency.unwrap_or(config.inference.processing_concurrency);
    let retry_attempts = retry_attempts.unwrap_or(config.inference.retry_attempts);
    if concurrency == 0 {
        anyhow::bail!("--concurrency must be at least 1");
    }
    if retry_attempts == 0 {
        anyhow::bail!("--retry-attempts must be at least 1");
    }
    Ok(ProcessingConfig {
        concurrency,
        request_timeout: Duration::from_secs(config.inference.timeout_secs),
        retry_attempts,
        initial_backoff: Duration::from_millis(config.inference.retry_initial_backoff_ms),
        max_backoff: Duration::from_millis(config.inference.retry_max_backoff_ms),
        use_cache: config.inference.cache_enabled && !no_cache,
    })
}

/// Ollama extraction has its own typed request timeout because structured
/// generation is materially slower than a TEI embedding request. The outer
/// resilient wrapper must never shorten that provider contract.
fn extraction_processing_config(
    config: &RuntimeConfig,
    concurrency: Option<usize>,
    retry_attempts: Option<usize>,
    no_cache: bool,
) -> Result<ProcessingConfig> {
    let mut processing = processing_config(config, concurrency, retry_attempts, no_cache)?;
    if config
        .inference
        .extraction_provider
        .eq_ignore_ascii_case("ollama")
    {
        processing.request_timeout = Duration::from_secs(config.inference.ollama_timeout_secs);
    }
    Ok(processing)
}

/// Record an interrupt and return whether graceful cancellation was already
/// requested. A second interrupt is therefore an explicit force-exit signal.
fn request_cancellation(requested: &AtomicBool) -> bool {
    requested.swap(true, Ordering::AcqRel)
}

fn install_cancellation_handler() -> Arc<AtomicBool> {
    let requested = Arc::new(AtomicBool::new(false));
    let listener = requested.clone();
    tokio::spawn(async move {
        while tokio::signal::ctrl_c().await.is_ok() {
            if request_cancellation(&listener) {
                eprintln!("Second Ctrl-C received; exiting immediately.");
                std::process::exit(130);
            }
            eprintln!(
                "Cancellation requested; finishing the current item and saving its checkpoint. Press Ctrl-C again to exit immediately."
            );
        }
    });
    requested
}

pub(crate) fn configured_search_agent(
    repo: Repository,
    embedder: SharedEmbedder,
    search: &SearchConfig,
) -> SearchAgent {
    let strategy = match search.fusion_strategy.trim().to_ascii_lowercase().as_str() {
        "rrf" => FusionStrategy::ReciprocalRank,
        // `RuntimeConfig::validate` rejects every other value before this
        // function is reached. Keep this fallback defensive for embedders that
        // construct SearchConfig directly.
        "weighted" => FusionStrategy::Weighted,
        _ => FusionStrategy::ReciprocalRank,
    };
    SearchAgent::new(repo, embedder)
        .with_fusion_config(
            FusionConfig {
                strategy,
                rrf_k: search.rrf_k,
                vector_weight: search.vector_weight,
                fulltext_weight: search.fulltext_weight,
                candidate_pool_multiplier: search.candidate_pool_multiplier,
                candidate_pool_min: search.candidate_pool_min,
                candidate_pool_max: search.candidate_pool_max,
            },
            search.note_weight,
            search.message_weight,
            search.conversation_summary_weight,
        )
        .with_graph_config(GraphRetrievalConfig {
            enabled: search.graph_enabled,
            max_seed_entities: search.graph_max_seed_entities,
            max_seed_notes: search.graph_max_seed_notes,
            max_hops: search.graph_max_hops,
            per_node_fanout: search.graph_per_node_fanout,
            allowed_edge_types: search.graph_allowed_edge_types.clone(),
            allow_outbound: search.graph_allow_outbound,
            allow_inbound: search.graph_allow_inbound,
            min_confidence: search.graph_min_confidence,
            per_hop_decay: search.graph_per_hop_decay,
            candidate_cap: search.graph_candidate_cap,
            seed_score: search.graph_seed_score,
        })
}

pub(crate) fn augment_options(
    max_chunks: usize,
    max_total_tokens: usize,
    max_chunk_tokens: usize,
    config: &AugmentConfig,
) -> AugmentOptions {
    AugmentOptions {
        max_chunks,
        max_total_tokens,
        max_chunk_tokens,
        novelty_weight: config.novelty_weight,
        min_relevance: config.min_relevance,
        near_duplicate_threshold: config.near_duplicate_threshold,
        ..Default::default()
    }
}

/// Zero budgets normally return an empty context without contacting the
/// embedder. Explain mode deliberately samples a bounded candidate set so it
/// can report identified exclusion decisions; ordinary output must retain the
/// established no-inference fast path.
pub(crate) fn should_sample_augment_candidates(explain: bool, options: &AugmentOptions) -> bool {
    explain
        || (options.max_chunks != 0
            && options.max_total_tokens != 0
            && options.max_chunk_tokens != 0)
}

pub(crate) fn augment_needs_tei(
    explain: bool,
    limit: Option<usize>,
    max_tokens: Option<usize>,
    max_chunk_tokens: Option<usize>,
    config: &AugmentConfig,
) -> bool {
    should_sample_augment_candidates(
        explain,
        &augment_options(
            limit.unwrap_or(config.default_limit),
            max_tokens.unwrap_or(config.max_tokens),
            max_chunk_tokens.unwrap_or(config.max_chunk_tokens),
            config,
        ),
    )
}

/// Provider health checks required by a command after database bootstrap.
/// Keeping this mapping centralized makes lazy resource behavior testable
/// without constructing providers or contacting an inference service.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ProviderRequirements {
    embedder: bool,
    extractor: bool,
}

fn provider_requirements(
    command: &Commands,
    explain: bool,
    config: &RuntimeConfig,
    skip_extraction: bool,
) -> ProviderRequirements {
    let notes_edit_reprocesses = matches!(
        command,
        Commands::Notes { command } if notes_edit_requires_inference(command)
    );
    let augment_needs_embeddings = match command {
        Commands::Augment {
            limit,
            max_tokens,
            max_chunk_tokens,
            ..
        } => augment_needs_tei(
            explain,
            *limit,
            *max_tokens,
            *max_chunk_tokens,
            &config.augment,
        ),
        _ => false,
    };
    let embedder = augment_needs_embeddings
        || notes_edit_reprocesses
        || matches!(
            command,
            Commands::Add { .. }
                | Commands::Import { .. }
                | Commands::Sources {
                    command: SourcesCommand::Reimport { .. },
                }
                | Commands::ImportChats { .. }
                | Commands::MigrateChats { .. }
                | Commands::Search { .. }
                | Commands::EvalAugment { .. }
                | Commands::Reindex { .. }
                | Commands::Interactive
        );
    let extractor = (notes_edit_reprocesses
        || matches!(
            command,
            Commands::Add { .. }
                | Commands::Import { .. }
                | Commands::Sources {
                    command: SourcesCommand::Reimport { .. },
                }
                | Commands::ImportChats { .. }
                | Commands::Interactive
                | Commands::ExtractEntities { .. }
                | Commands::MigrateChats {
                    with_notes: true,
                    ..
                }
        ))
        && (!skip_extraction || matches!(command, Commands::ExtractEntities { .. }));

    ProviderRequirements {
        embedder,
        extractor,
    }
}

pub(crate) fn zero_budget_augment_context(
    query: String,
    scope: SearchScope,
    entity_filter: Option<String>,
) -> graphrag_agents::AugmentContext {
    let diagnostics = AugmentDiagnostics {
        token_count_mode: TokenCountMode::Estimated,
        header_tokens: 0,
        dropped_duplicates: 0,
        dropped_near_duplicates: 0,
        dropped_for_relevance: 0,
        dropped_for_budget: 0,
        dropped_for_entity_filter: 0,
        graph_candidates_considered: 0,
        graph_candidates_selected: 0,
        graph_candidates_dropped: 0,
    };
    graphrag_agents::AugmentContext {
        query,
        scope,
        entity_filter,
        chunks: Vec::new(),
        exclusions: Vec::new(),
        total_tokens: 0,
        diagnostics,
        dropped_duplicates: 0,
        dropped_for_budget: 0,
        dropped_for_entity_filter: 0,
    }
}

pub(crate) fn packing_diagnostics_text(diagnostics: &AugmentDiagnostics) -> String {
    let token_count_mode = match diagnostics.token_count_mode {
        TokenCountMode::Exact => "exact",
        TokenCountMode::Estimated => "estimated",
    };
    format!(
        "token_mode={token_count_mode}; header_tokens={}; dropped_duplicates={}; dropped_near_duplicates={}; dropped_for_relevance={}; dropped_for_budget={}; dropped_for_entity_filter={}; graph_considered={}; graph_selected={}; graph_dropped={}",
        diagnostics.header_tokens,
        diagnostics.dropped_duplicates,
        diagnostics.dropped_near_duplicates,
        diagnostics.dropped_for_relevance,
        diagnostics.dropped_for_budget,
        diagnostics.dropped_for_entity_filter,
        diagnostics.graph_candidates_considered,
        diagnostics.graph_candidates_selected,
        diagnostics.graph_candidates_dropped,
    )
}

pub(crate) fn librarian_runtime_config(
    config: &RuntimeConfig,
    cli_skip_extraction: bool,
) -> LibrarianRuntimeConfig {
    LibrarianRuntimeConfig {
        min_chunk_size: config.librarian.min_chunk_size,
        target_chunk_size: config.librarian.target_chunk_size,
        max_chunk_size: config.librarian.max_chunk_size,
        chunk_overlap: config.librarian.chunk_overlap,
        skip_entity_extraction: cli_skip_extraction || config.librarian.skip_entity_extraction,
        extract_log_each: config.librarian.extract_log_each,
        extract_max_chars: config.librarian.extract_max_chars,
        extract_progress_every: config.librarian.extract_progress_every,
        extract_progress_every_secs: config.librarian.extract_progress_every_secs,
        import_progress_every: config.librarian.import_progress_every,
        import_progress_every_secs: config.librarian.import_progress_every_secs,
    }
}

fn print_doctor_report(report: &doctor::DoctorReport, format: DoctorFormat) -> Result<()> {
    match format {
        DoctorFormat::Human => print!("{}", report.render_human()),
        DoctorFormat::Json => println!("{}", serde_json::to_string_pretty(report)?),
    }
    Ok(())
}

/// Run archive-only operations before resolving runtime configuration. These
/// commands never open the configured database or contact a provider, so an
/// invalid config must not prevent an operator from inspecting or dry-running
/// recovery data.
async fn run_archive_only_command(cli: &Cli) -> Result<bool> {
    if let Commands::Backup {
        command: BackupCommand::Verify { path, format },
    } = &cli.command
    {
        print_backup_summary(&backup::verify_backup(path)?, *format)?;
        return Ok(true);
    }
    if let Commands::ImportData {
        path,
        dry_run: true,
        format,
    } = &cli.command
    {
        if cli.memory {
            anyhow::bail!("import-data requires a fresh persistent --db-path, not --memory");
        }
        let target = cli.db_path.as_deref().context(
            "import-data requires an explicit fresh --db-path; it never imports over the configured database",
        )?;
        print_backup_summary(&backup::import_jsonl(path, target, true).await?, *format)?;
        return Ok(true);
    }
    if let Commands::Backup {
        command:
            BackupCommand::Restore {
                path,
                dry_run: true,
                format,
            },
    } = &cli.command
    {
        if cli.memory {
            anyhow::bail!("backup restore requires a fresh persistent --db-path, not --memory");
        }
        let target = cli.db_path.as_deref().context(
            "backup restore requires an explicit fresh --db-path; it never restores over the configured database",
        )?;
        print_backup_summary(&backup::restore_backup(path, target, true).await?, *format)?;
        return Ok(true);
    }
    Ok(false)
}

/// Convert expected command failures into the documented automation contract.
/// Typed errors take precedence; message fallbacks cover validation failures
/// produced by Clap-adjacent handlers that intentionally use `anyhow::bail!`.
pub(crate) fn exit_code_for(error: &anyhow::Error) -> output::ExitCode {
    for cause in error.chain() {
        if cause
            .downcast_ref::<commands::notes::NotesEditValidationError>()
            .is_some()
        {
            return output::ExitCode::Validation;
        }
        if cause
            .downcast_ref::<graphrag_config::ConfigError>()
            .is_some()
        {
            // Configuration is supplied at invocation time through --config,
            // environment, or the resolved local config path. A read, parse,
            // or validation failure is therefore a recoverable user-input
            // error, not an internal command failure.
            return output::ExitCode::Validation;
        }
        if let Some(error) = cause.downcast_ref::<graphrag_db::DbError>() {
            return match error {
                graphrag_db::DbError::NotFound(_, _) => output::ExitCode::NotFound,
                graphrag_db::DbError::EmbeddingCompatibility { .. }
                | graphrag_db::DbError::LegacyEmbeddingMetadata { .. } => {
                    output::ExitCode::Compatibility
                }
                _ => output::ExitCode::Internal,
            };
        }
        if let Some(error) = cause.downcast_ref::<graphrag_agents::AgentError>() {
            return match error {
                graphrag_agents::AgentError::NotFound(_) => output::ExitCode::NotFound,
                graphrag_agents::AgentError::Database(graphrag_db::DbError::NotFound(_, _)) => {
                    output::ExitCode::NotFound
                }
                graphrag_agents::AgentError::Database(
                    graphrag_db::DbError::EmbeddingCompatibility { .. }
                    | graphrag_db::DbError::LegacyEmbeddingMetadata { .. },
                ) => output::ExitCode::Compatibility,
                graphrag_agents::AgentError::DurablePartialFailure { .. } => {
                    output::ExitCode::PartialFailure
                }
                _ => output::ExitCode::Internal,
            };
        }
    }

    let message = error.to_string().to_lowercase();
    if message.contains("not found") {
        output::ExitCode::NotFound
    } else if message.contains("compatibility") || message.contains("legacy vector") {
        output::ExitCode::Compatibility
    } else if [
        "requires",
        "required",
        "refusing",
        "cannot",
        "invalid",
        "must ",
        "without --yes",
        "empty",
    ]
    .iter()
    .any(|needle| message.contains(needle))
    {
        output::ExitCode::Validation
    } else {
        output::ExitCode::Internal
    }
}

pub(crate) async fn run() -> Result<()> {
    // Load environment variables from .env if present.
    dotenvy::dotenv().ok();

    let cli = Cli::parse();
    if run_archive_only_command(&cli).await? {
        return Ok(());
    }
    let doctor_format = match &cli.command {
        Commands::Doctor { format } => Some(*format),
        _ => None,
    };
    let overrides = CliOverrides {
        database_path: cli.db_path.clone(),
    };
    let config = match RuntimeConfig::load(cli.config.as_deref(), &overrides) {
        Ok(config) => config,
        Err(error) if doctor_format.is_some() => {
            let report = doctor::DoctorReport::configuration_error(error);
            print_doctor_report(&report, doctor_format.expect("doctor format is present"))?;
            std::process::exit(report.exit_code);
        }
        Err(error) => return Err(error).context("failed to resolve runtime configuration"),
    };

    if let Commands::Doctor { format } = &cli.command {
        let report = doctor::run(&config, &inference_provider_config(&config), cli.memory).await;
        print_doctor_report(&report, *format)?;
        std::process::exit(report.exit_code);
    }

    if let Commands::Config { command } = &cli.command {
        match command {
            ConfigCommand::Show => print!("{}", config.redacted_toml()?),
            ConfigCommand::Validate => println!("Configuration is valid."),
        }
        return Ok(());
    }

    // Verification and restore intentionally run before normal database
    // startup. Verification must not open a database at all, and restore must
    // validate an archive before creating its staged fresh target.
    if let Commands::Backup {
        command: BackupCommand::Verify { path, format },
    } = &cli.command
    {
        print_backup_summary(&backup::verify_backup(path)?, *format)?;
        return Ok(());
    }
    if let Commands::ImportData {
        path,
        dry_run,
        format,
    } = &cli.command
    {
        if cli.memory {
            anyhow::bail!("import-data requires a fresh persistent --db-path, not --memory");
        }
        let target = cli.db_path.as_deref().context(
            "import-data requires an explicit fresh --db-path; it never imports over the configured database",
        )?;
        print_backup_summary(
            &backup::import_jsonl(path, target, *dry_run).await?,
            *format,
        )?;
        return Ok(());
    }
    if let Commands::Backup {
        command:
            BackupCommand::Restore {
                path,
                dry_run,
                format,
            },
    } = &cli.command
    {
        if cli.memory {
            anyhow::bail!("backup restore requires a fresh persistent --db-path, not --memory");
        }
        let target = cli.db_path.as_deref().context(
            "backup restore requires an explicit fresh --db-path; it never restores over the configured database",
        )?;
        print_backup_summary(
            &backup::restore_backup(path, target, *dry_run).await?,
            *format,
        )?;
        return Ok(());
    }

    let cli_skip_extraction = matches!(
        &cli.command,
        Commands::ImportChats {
            skip_extraction: true,
            ..
        } | Commands::MigrateChats {
            skip_extraction: true,
            ..
        }
    );
    let librarian_config = librarian_runtime_config(&config, cli_skip_extraction);
    let skip_extraction = librarian_config.skip_entity_extraction;
    let inference_config = inference_provider_config(&config);

    if let Commands::EmbeddingDim { text } = &cli.command {
        let tei = InferenceProviders::from_config(&inference_config).embedder;
        let tei_ok = tei.health().await.unwrap_or(false);
        if !tei_ok {
            eprintln!("Error: embeddings service is not reachable.");
            eprintln!("  TEI (embeddings): {}", tei.capabilities().endpoint);
            anyhow::bail!("Embeddings service unavailable");
        }

        let probe = text
            .clone()
            .unwrap_or_else(|| "dimension probe".to_string());
        let embedding = tei.embed(&probe, false).await?;
        println!("Embedding dimension: {}", embedding.len());
        return Ok(());
    }

    // Setup logging: default to WARN, allow explicit DEBUG via --verbose,
    // and support custom filters through RUST_LOG.
    let log_filter = if cli.verbose {
        EnvFilter::new("debug")
    } else {
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new(config.logging.level.clone()))
    };
    let subscriber = FmtSubscriber::builder()
        .with_env_filter(log_filter)
        .with_target(false)
        .with_writer(std::io::stderr)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Initialize database
    if let Commands::ResetDb { db_path } = &cli.command {
        let path = db_path
            .clone()
            .unwrap_or_else(|| config.database.path.clone());

        if path.exists() {
            std::fs::remove_dir_all(&path)
                .with_context(|| format!("Failed to remove db at {}", path.display()))?;
            println!("✓ Removed database at {}", path.display());
        } else {
            println!(
                "Database not found at {}, nothing to remove",
                path.display()
            );
        }
        return Ok(());
    }

    let db = if cli.memory {
        info!("Using in-memory database");
        init_memory().await?
    } else {
        let db_path = config.database.path.clone();

        // Ensure directory exists
        if let Some(parent) = db_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        info!("Using database at: {}", db_path.display());
        init_persistent(&db_path).await?
    };

    if matches!(cli.command, Commands::SchemaVersion) {
        println!(
            "Schema version: {} (latest supported: {})",
            migrations::current_version(&db).await?,
            migrations::latest_version()
        );
        return Ok(());
    }

    let repo = Repository::new(db);
    // Resolve local notes-edit validation before provider health checks so an
    // offline service cannot mask a deterministic validation/not-found error.
    let prepared_notes_edit = match &cli.command {
        Commands::Notes { command } => commands::notes::prepare_edit(&repo, command).await?,
        _ => None,
    };
    let providers = InferenceProviders::from_config(&inference_config);
    let processing = processing_config(&config, cli.concurrency, cli.retry_attempts, cli.no_cache)?;
    let extraction_processing =
        extraction_processing_config(&config, cli.concurrency, cli.retry_attempts, cli.no_cache)?;
    // The wrappers share their semaphores and counters across every clone in
    // this invocation, so imports, extraction, and search cannot overload a
    // local provider merely by arriving through different commands.
    let tei: SharedEmbedder = Arc::new(ResilientEmbedder::new(
        providers.embedder,
        Some(repo.clone()),
        processing.clone(),
    ));
    let tgi: SharedEntityExtractor = Arc::new(ResilientEntityExtractor::new(
        providers.extractor,
        Some(repo.clone()),
        extraction_processing,
    ));

    // Check inference services only when needed.
    let requirements = provider_requirements(&cli.command, cli.explain, &config, skip_extraction);

    if requirements.embedder {
        let tei_ok = tei.health().await.unwrap_or(false);
        if !tei_ok {
            eprintln!("Error: embeddings service is not reachable.");
            eprintln!("  TEI (embeddings): {}", tei.capabilities().endpoint);
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Embeddings service unavailable");
        }
    }

    if requirements.extractor {
        let tgi_ok = tgi.health().await.unwrap_or(false);
        if !tgi_ok {
            eprintln!("Error: extraction service is not reachable.");
            eprintln!("  TGI (extraction): {}", tgi.capabilities().endpoint);
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Extraction service unavailable");
        }
    }

    // A persistent RocksDB database is exclusively owned by this process, so
    // a second `graphrag jobs cancel` process cannot reliably attach while work
    // is active. Ctrl-C is the in-process control plane: it requests a safe
    // stop and the Librarian observes it between atomic item mutations.
    let cancellation_requested = matches!(
        &cli.command,
        Commands::ExtractEntities { .. }
            | Commands::Reindex { .. }
            | Commands::Jobs {
                command: JobsCommand::Resume { .. }
            }
    )
    .then(install_cancellation_handler);

    crate::dispatch::execute(
        AppContext {
            repo,
            config,
            tei,
            tgi,
            librarian_config,
            cancellation_requested,
            prepared_notes_edit,
        },
        cli.command,
        cli.explain,
    )
    .await
}

#[cfg(test)]
use crate::dispatch::*;
#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;
    use crate::cli::{GardenCommand, GraphModeArg, SearchScopeArg};
    use crate::commands::notes::NotesCommand;
    use crate::dispatch::{AugmentMachineChunk, SearchMachineResult};
    use graphrag_agents::{GraphEvidence, GraphPathStep, ProcessingRunResult, SearchHitType};
    use graphrag_core::record_id_to_string;
    use graphrag_db::{repository::RelatedNotes, ProcessingJobType};
    use std::path::PathBuf;

    #[cfg(unix)]
    #[test]
    fn rejects_non_utf8_import_paths_without_identity_collapse() {
        use std::os::unix::ffi::OsStringExt;

        let first = PathBuf::from(std::ffi::OsString::from_vec(vec![b'a', 0x80]));
        let second = PathBuf::from(std::ffi::OsString::from_vec(vec![b'b', 0x81]));
        assert!(import_path_utf8(&first).is_err());
        assert!(import_path_utf8(&second).is_err());
    }

    #[test]
    fn source_delete_defaults_to_a_non_mutating_preview() {
        assert!(delete_is_dry_run(false, false));
        assert!(delete_is_dry_run(true, false));
        assert!(!delete_is_dry_run(false, true));
    }

    #[test]
    fn second_cancellation_request_is_a_force_exit_signal() {
        let requested = AtomicBool::new(false);
        assert!(!request_cancellation(&requested));
        assert!(requested.load(Ordering::Acquire));
        assert!(request_cancellation(&requested));
    }

    #[test]
    fn cancelled_entity_extraction_is_not_reported_as_success() {
        let cancelled = ProcessingRunResult {
            job_id: "processing_job:resume-me".into(),
            completed: 3,
            failed: 0,
            cancelled: true,
        };
        assert!(report_entity_extraction_result(&cancelled)
            .unwrap_err()
            .to_string()
            .contains("Entity extraction cancelled"));

        let completed = ProcessingRunResult {
            cancelled: false,
            ..cancelled
        };
        assert!(report_entity_extraction_result(&completed).is_ok());
    }

    #[tokio::test]
    async fn resume_health_check_uses_only_the_job_required_provider() {
        use graphrag_agents::{DeterministicEmbedder, FixtureEntityExtractor};

        let repo = Repository::new(init_memory().await.unwrap());
        let embedding_job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::Embedding,
                None,
                1,
                Some("test".into()),
                vec!["note:embedding".into()],
            )
            .await
            .unwrap();
        let entity_job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::EntityExtraction,
                None,
                1,
                Some("test".into()),
                vec!["note:entity".into()],
            )
            .await
            .unwrap();
        let healthy_embedder: SharedEmbedder = Arc::new(DeterministicEmbedder::default());
        let unhealthy_embedder: SharedEmbedder =
            Arc::new(DeterministicEmbedder::default().unhealthy());
        let healthy_extractor: SharedEntityExtractor = Arc::new(FixtureEntityExtractor::default());
        let unhealthy_extractor: SharedEntityExtractor =
            Arc::new(FixtureEntityExtractor::default().unhealthy());

        assert!(ensure_resume_provider_health(
            &embedding_job,
            &healthy_embedder,
            &unhealthy_extractor,
        )
        .await
        .is_ok());
        assert!(ensure_resume_provider_health(
            &embedding_job,
            &unhealthy_embedder,
            &healthy_extractor,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Embeddings service unavailable"));
        assert!(ensure_resume_provider_health(
            &entity_job,
            &unhealthy_embedder,
            &healthy_extractor,
        )
        .await
        .is_ok());
        assert!(ensure_resume_provider_health(
            &entity_job,
            &healthy_embedder,
            &unhealthy_extractor,
        )
        .await
        .unwrap_err()
        .to_string()
        .contains("Extraction service unavailable"));
    }

    #[tokio::test]
    async fn resume_entity_failure_returns_non_success_after_persisting_diagnostic() {
        use graphrag_agents::{DeterministicEmbedder, FixtureEntityExtractor};

        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(graphrag_core::Note::new("entity extraction that will fail"))
            .await
            .unwrap();
        let job = repo
            .create_processing_job_with_scope(
                ProcessingJobType::EntityExtraction,
                None,
                1,
                Some("note_ids:force=false".into()),
                vec![record_id_to_string(note.id.as_ref().unwrap())],
            )
            .await
            .unwrap();
        let job_id = record_id_to_string(job.id.as_ref().unwrap());
        repo.cancel_processing_job(job.id.as_ref().unwrap())
            .await
            .unwrap();

        let error = cmd_jobs(
            repo.clone(),
            Arc::new(DeterministicEmbedder::default()),
            Arc::new(FixtureEntityExtractor::default().fail_next_requests(1, "timeout")),
            LibrarianRuntimeConfig::default(),
            Arc::new(AtomicBool::new(false)),
            JobsCommand::Resume { id: job_id.clone() },
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("timeout"));

        let job = repo.get_processing_job(&job_id).await.unwrap().unwrap();
        assert_eq!(job.status, "failed");
        assert_eq!(job.failed_count, 1);
        assert!(job
            .last_error
            .as_deref()
            .is_some_and(|diagnostic| diagnostic.contains("timeout")));
    }

    #[test]
    fn librarian_runtime_config_forwards_legacy_derived_chunking_bounds() {
        let mut config = RuntimeConfig::default();
        config.librarian = graphrag_config::LibrarianConfig {
            min_chunk_size: 40,
            target_chunk_size: 80,
            max_chunk_size: 80,
            chunk_overlap: 79,
            ..graphrag_config::LibrarianConfig::default()
        };

        let runtime = librarian_runtime_config(&config, false);
        assert_eq!(runtime.min_chunk_size, 40);
        assert_eq!(runtime.target_chunk_size, 80);
        assert_eq!(runtime.max_chunk_size, 80);
        assert_eq!(runtime.chunk_overlap, 79);
    }

    #[test]
    fn edge_dry_run_reports_actual_existence() {
        assert_eq!(
            edge_dry_run_message("related_to:example", true),
            "Dry run: related_to:example exists and would be deleted; no changes made."
        );
        assert_eq!(
            edge_dry_run_message("related_to:example", false),
            "Dry run: related_to:example is absent; no changes made."
        );
    }

    #[test]
    fn augment_commands_forward_runtime_tuning_to_packing_options() {
        let config = AugmentConfig {
            novelty_weight: 0.4,
            min_relevance: 0.2,
            near_duplicate_threshold: 0.7,
            ..Default::default()
        };
        let options = augment_options(3, 90, 30, &config);
        assert_eq!(options.max_chunks, 3);
        assert_eq!(options.max_total_tokens, 90);
        assert_eq!(options.max_chunk_tokens, 30);
        assert_eq!(options.novelty_weight, 0.4);
        assert_eq!(options.min_relevance, 0.2);
        assert_eq!(options.near_duplicate_threshold, 0.7);
    }

    #[test]
    fn zero_budget_augmentation_samples_candidates_only_for_explain() {
        let zero_total = augment_options(1, 0, 32, &AugmentConfig::default());
        assert!(!should_sample_augment_candidates(false, &zero_total));
        assert!(should_sample_augment_candidates(true, &zero_total));

        let zero_chunk = augment_options(1, 32, 0, &AugmentConfig::default());
        assert!(!should_sample_augment_candidates(false, &zero_chunk));

        assert!(!augment_needs_tei(
            false,
            Some(1),
            Some(0),
            Some(32),
            &AugmentConfig::default(),
        ));
        assert!(augment_needs_tei(
            true,
            Some(1),
            Some(0),
            Some(32),
            &AugmentConfig::default(),
        ));

        let context = zero_budget_augment_context("query".into(), SearchScope::Notes, None);
        assert!(context.chunks.is_empty());
        assert!(context.exclusions.is_empty());
        assert_eq!(context.total_tokens, 0);
    }

    #[test]
    fn human_packing_diagnostics_include_all_budget_and_selection_decisions() {
        let diagnostics = AugmentDiagnostics {
            token_count_mode: TokenCountMode::Estimated,
            header_tokens: 12,
            dropped_duplicates: 1,
            dropped_near_duplicates: 2,
            dropped_for_relevance: 3,
            dropped_for_budget: 4,
            dropped_for_entity_filter: 5,
            graph_candidates_considered: 6,
            graph_candidates_selected: 2,
            graph_candidates_dropped: 4,
        };
        assert_eq!(
            packing_diagnostics_text(&diagnostics),
            "token_mode=estimated; header_tokens=12; dropped_duplicates=1; dropped_near_duplicates=2; dropped_for_relevance=3; dropped_for_budget=4; dropped_for_entity_filter=5; graph_considered=6; graph_selected=2; graph_dropped=4"
        );
    }

    #[test]
    fn garden_apply_requires_the_explicit_confirmation_flag() {
        let cli = Cli::try_parse_from(["graphrag", "garden", "apply", "--yes"]).unwrap();
        assert!(matches!(
            cli.command,
            Commands::Garden {
                command: GardenCommand::Apply { yes: true }
            }
        ));
        assert!(Cli::try_parse_from(["graphrag", "garden", "apply"]).is_err());
    }

    #[test]
    fn search_and_augment_accept_explicit_graph_modes() {
        let search = Cli::try_parse_from(["graphrag", "search", "atlas", "--graph=off"]).unwrap();
        assert!(matches!(
            search.command,
            Commands::Search {
                graph: GraphModeArg::Off,
                ..
            }
        ));

        let augment =
            Cli::try_parse_from(["graphrag", "augment", "atlas", "--graph", "on"]).unwrap();
        assert!(matches!(
            augment.command,
            Commands::Augment {
                graph: GraphModeArg::On,
                ..
            }
        ));
    }

    #[test]
    fn explain_is_global_and_search_surfaces_use_the_shared_output_format() {
        let search = Cli::try_parse_from([
            "graphrag",
            "search",
            "atlas",
            "--explain",
            "--format",
            "json",
        ])
        .unwrap();
        assert!(search.explain);
        assert!(matches!(
            search.command,
            Commands::Search {
                format: output::OutputFormat::Json,
                ..
            }
        ));

        let augment = Cli::try_parse_from(["graphrag", "--explain", "augment", "atlas"]).unwrap();
        assert!(augment.explain);
    }

    #[test]
    fn notes_commands_expose_safe_edit_delete_and_machine_output_flags() {
        let edit = Cli::try_parse_from([
            "graphrag", "notes", "edit", "note:one", "--title", "Revised", "--detach", "--format",
            "json",
        ])
        .unwrap();
        assert!(matches!(
            edit.command,
            Commands::Notes {
                command: NotesCommand::Edit {
                    detach: true,
                    format: output::OutputFormat::Json,
                    ..
                }
            }
        ));

        let delete = Cli::try_parse_from([
            "graphrag",
            "notes",
            "delete",
            "note:one",
            "--dry-run",
            "--format",
            "jsonl",
        ])
        .unwrap();
        assert!(matches!(
            delete.command,
            Commands::Notes {
                command: NotesCommand::Delete {
                    dry_run: true,
                    yes: false,
                    format: output::OutputFormat::Jsonl,
                    ..
                }
            }
        ));
    }

    #[test]
    fn metadata_only_notes_edit_skips_provider_preflight() {
        let metadata_only = NotesCommand::Edit {
            id: "note:one".into(),
            title: Some("Retitled".into()),
            content_file: None,
            stdin: false,
            tags: None,
            detach: false,
            format: output::OutputFormat::Human,
        };
        assert!(!notes_edit_requires_inference(&metadata_only));

        let content_edit = NotesCommand::Edit {
            id: "note:one".into(),
            title: None,
            content_file: Some(PathBuf::from("replacement.md")),
            stdin: false,
            tags: None,
            detach: false,
            format: output::OutputFormat::Human,
        };
        assert!(notes_edit_requires_inference(&content_edit));

        let detached_edit = NotesCommand::Edit {
            id: "note:one".into(),
            title: None,
            content_file: None,
            stdin: false,
            tags: None,
            detach: true,
            format: output::OutputFormat::Human,
        };
        assert!(notes_edit_requires_inference(&detached_edit));
    }

    #[test]
    fn provider_requirements_keep_read_only_and_metadata_commands_offline() {
        let config = RuntimeConfig::default();
        let stats = provider_requirements(&Commands::Stats, false, &config, false);
        assert_eq!(
            stats,
            ProviderRequirements {
                embedder: false,
                extractor: false,
            }
        );

        let metadata_edit = Commands::Notes {
            command: NotesCommand::Edit {
                id: "note:one".into(),
                title: Some("renamed".into()),
                content_file: None,
                stdin: false,
                tags: None,
                detach: false,
                format: output::OutputFormat::Human,
            },
        };
        assert_eq!(
            provider_requirements(&metadata_edit, false, &config, false),
            ProviderRequirements {
                embedder: false,
                extractor: false,
            }
        );

        let search = Commands::Search {
            query: "query".into(),
            limit: None,
            scope: SearchScopeArg::Notes,
            since_days: None,
            source_uri: None,
            context: false,
            graph: GraphModeArg::Auto,
            format: output::OutputFormat::Json,
        };
        assert_eq!(
            provider_requirements(&search, false, &config, false),
            ProviderRequirements {
                embedder: true,
                extractor: false,
            }
        );

        let extraction = Commands::ExtractEntities {
            limit: 1,
            all: false,
            note_ids: vec![],
            force: false,
        };
        assert_eq!(
            provider_requirements(&extraction, false, &config, true),
            ProviderRequirements {
                embedder: false,
                extractor: true,
            }
        );
    }

    #[test]
    fn output_envelope_has_stable_required_machine_fields() {
        let envelope =
            output::OutputEnvelope::success("notes.show", serde_json::json!({"id": "note:one"}));
        let json = serde_json::to_value(envelope).unwrap();
        for key in [
            "schema_version",
            "command",
            "success",
            "data",
            "warnings",
            "errors",
        ] {
            assert!(json.get(key).is_some(), "missing output key {key}");
        }
    }

    #[test]
    fn non_explain_machine_adapters_omit_retrieval_evidence() {
        let search = SearchMachineResult::from_scoped(
            &graphrag_agents::search::ScopedSearchResult {
                hit_type: SearchHitType::Note,
                id: "note:one".into(),
                title: Some("One".into()),
                content: "ordinary result".into(),
                created_at: None,
                source_uri: None,
                score: 0.75,
                fusion: Default::default(),
                score_kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                effective_weight: 1.0,
                conversation_uuid: None,
                message_index: None,
                role: None,
                graph: None,
            },
            None,
        );
        let search = serde_json::to_value(search).unwrap();
        assert_eq!(search["id"], "note:one");
        for evidence_field in [
            "final_score",
            "fused",
            "vector",
            "full_text",
            "graph",
            "inclusion",
            "effective_weight",
        ] {
            assert!(search.get(evidence_field).is_none());
        }

        let search_with_context = SearchMachineResult::from_scoped(
            &graphrag_agents::search::ScopedSearchResult {
                hit_type: SearchHitType::Note,
                id: "note:context".into(),
                title: Some("Context".into()),
                content: "context result".into(),
                created_at: None,
                source_uri: None,
                score: 0.75,
                fusion: Default::default(),
                score_kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                effective_weight: 1.0,
                conversation_uuid: None,
                message_index: None,
                role: None,
                graph: None,
            },
            Some(RelatedNotes::default()),
        );
        let search_with_context = serde_json::to_value(search_with_context).unwrap();
        assert!(search_with_context.get("related").is_some());

        let augment = serde_json::to_value(AugmentMachineChunk {
            citation: 1,
            hit_type: "note",
            id: "note:one".into(),
            title: Some("One".into()),
            snippet: "ordinary result".into(),
            created_at: None,
            source_uri: None,
            score: 0.75,
            conversation_uuid: None,
            message_index: None,
            role: None,
            approx_tokens: 2,
            rendered_tokens: 2,
            truncated: false,
            selected_span_start: Some(0),
            selected_span_end: Some(15),
        })
        .unwrap();
        assert_eq!(augment["id"], "note:one");
        assert!(augment.get("final_score").is_none());
        assert!(augment.get("inclusion").is_none());
    }

    #[test]
    fn documented_exit_codes_classify_typed_and_validation_errors() {
        let not_found = anyhow::Error::new(graphrag_db::DbError::NotFound(
            "note".into(),
            "note:missing".into(),
        ));
        assert_eq!(exit_code_for(&not_found), output::ExitCode::NotFound);

        let compatibility =
            anyhow::Error::new(graphrag_db::DbError::LegacyEmbeddingMetadata { vector_records: 1 });
        assert_eq!(
            exit_code_for(&compatibility),
            output::ExitCode::Compatibility
        );

        let validation = anyhow::anyhow!("refusing to delete without --yes");
        assert_eq!(exit_code_for(&validation), output::ExitCode::Validation);

        let unreadable_file = anyhow::Error::new(
            commands::notes::NotesEditValidationError::UnreadableContentFile {
                path: PathBuf::from("missing.md"),
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
            },
        );
        assert_eq!(
            exit_code_for(&unreadable_file),
            output::ExitCode::Validation
        );

        let unreadable_config = anyhow::Error::new(graphrag_config::ConfigError::ReadFile {
            path: PathBuf::from("missing-config.toml"),
            source: std::io::Error::new(std::io::ErrorKind::NotFound, "missing"),
        });
        assert_eq!(
            exit_code_for(&unreadable_config),
            output::ExitCode::Validation
        );

        let malformed_config_file = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(malformed_config_file.path(), "database = [").unwrap();
        let malformed_config = anyhow::Error::new(
            graphrag_config::RuntimeConfig::from_file(malformed_config_file.path())
                .expect_err("fixture must not parse"),
        );
        assert_eq!(
            exit_code_for(&malformed_config),
            output::ExitCode::Validation
        );

        let partial = anyhow::Error::new(graphrag_agents::AgentError::DurablePartialFailure {
            job_id: "processing_job:one".into(),
            completed: 3,
            failed: 1,
            message: "provider rejected one item".into(),
        });
        assert_eq!(exit_code_for(&partial), output::ExitCode::PartialFailure);

        // Ordinary errors can happen to contain this word, but are not a
        // persisted partial durable outcome and must retain their normal
        // failure classification.
        let ordinary_failure = anyhow::Error::new(graphrag_agents::AgentError::Processing(
            "provider failed before a job existed".into(),
        ));
        assert_eq!(exit_code_for(&ordinary_failure), output::ExitCode::Internal);
    }

    #[test]
    fn graph_path_rendering_retains_edge_and_endpoint_record_ids() {
        let path = vec![
            GraphPathStep {
                edge_id: "note_edge:one".into(),
                edge_type: "supports".into(),
                direction: "outbound".into(),
                confidence: 0.9,
                from_id: "note:seed".into(),
                to_id: "note:middle".into(),
            },
            GraphPathStep {
                edge_id: "note_edge:two".into(),
                edge_type: "contradicts".into(),
                direction: "inbound".into(),
                confidence: 0.75,
                from_id: "note:target".into(),
                to_id: "note:middle".into(),
            },
        ];

        assert_eq!(
            render_graph_path(&path, " -> "),
            "outbound:supports edge=note_edge:one endpoints=note:seed→note:middle confidence=0.90 -> inbound:contradicts edge=note_edge:two endpoints=note:target→note:middle confidence=0.75"
        );
    }

    #[test]
    fn graph_citation_rendering_retains_source_uri_and_reconstructable_path() {
        let graph = GraphEvidence {
            query_entities: vec!["Atlas".into()],
            seed_note_id: "note:seed".into(),
            path: vec![GraphPathStep {
                edge_id: "note_edge:one".into(),
                edge_type: "supports".into(),
                direction: "outbound".into(),
                confidence: 0.9,
                from_id: "note:seed".into(),
                to_id: "note:target".into(),
            }],
            hops: 1,
            decay: 0.8,
            score: 0.72,
            source_uri: Some("file:///notes/atlas.md".into()),
            provenance_ids: vec!["message:provenance".into()],
        };

        assert_eq!(
            render_graph_citation(&graph, " | "),
            "graph_seed=note:seed, graph_hops=1, graph_decay=0.80, graph_source_uri=file:///notes/atlas.md, graph_path=outbound:supports edge=note_edge:one endpoints=note:seed→note:target confidence=0.90, graph_provenance=message:provenance"
        );
        assert_eq!(
            render_search_graph_evidence(&graph),
            "entities=[Atlas] graph_seed=note:seed, graph_hops=1, graph_decay=0.80, graph_source_uri=file:///notes/atlas.md, graph_path=outbound:supports edge=note_edge:one endpoints=note:seed→note:target confidence=0.90, graph_provenance=message:provenance"
        );
        assert_eq!(
            render_augment_graph_evidence(&graph),
            ", graph_seed=note:seed, graph_hops=1, graph_decay=0.80, graph_source_uri=file:///notes/atlas.md, graph_path=outbound:supports edge=note_edge:one endpoints=note:seed→note:target confidence=0.90, graph_provenance=message:provenance"
        );
    }

    #[tokio::test]
    async fn graph_mode_context_enrichment_keeps_primary_hits_after_one_lookup_failure() {
        let hits = vec![
            graphrag_agents::search::ScopedSearchResult {
                hit_type: SearchHitType::Note,
                id: "note:available".into(),
                title: Some("available primary hit".into()),
                content: "available".into(),
                created_at: None,
                source_uri: None,
                score: 1.0,
                fusion: Default::default(),
                score_kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                effective_weight: 1.0,
                conversation_uuid: None,
                message_index: None,
                role: None,
                graph: None,
            },
            graphrag_agents::search::ScopedSearchResult {
                hit_type: SearchHitType::Note,
                id: "note:unavailable".into(),
                title: Some("unavailable primary hit".into()),
                content: "unavailable".into(),
                created_at: None,
                source_uri: None,
                score: 0.9,
                fusion: Default::default(),
                score_kind: graphrag_agents::ScoreKind::ReciprocalRankFusion,
                effective_weight: 1.0,
                conversation_uuid: None,
                message_index: None,
                role: None,
                graph: None,
            },
        ];
        let context = best_effort_related_notes(&hits, |id| async move {
            if id == "note:unavailable" {
                Err("injected related-note lookup failure")
            } else {
                Ok(RelatedNotes::default())
            }
        })
        .await
        .unwrap();

        // Both graph-capable command forms flow through this same enrichment
        // path, which must retain primary hits and successful context when a
        // separate related-note lookup fails.
        for graph in ["auto", "on"] {
            let cli =
                Cli::try_parse_from(["graphrag", "search", "atlas", "--context", "--graph", graph])
                    .unwrap();
            assert!(matches!(
                cli.command,
                Commands::Search {
                    context: true,
                    graph: GraphModeArg::Auto | GraphModeArg::On,
                    ..
                }
            ));
        }
        assert_eq!(hits.len(), 2, "primary graph search hits are unchanged");
        assert_eq!(context.len(), 1, "only the failed enrichment is omitted");
        assert!(context.contains_key("note:available"));
    }

    #[test]
    fn inference_cli_overrides_reject_zero() {
        let config = RuntimeConfig::default();
        assert!(processing_config(&config, Some(0), None, false).is_err());
        assert!(processing_config(&config, None, Some(0), false).is_err());
    }

    #[test]
    fn ollama_extraction_wrapper_keeps_the_provider_timeout() {
        let mut config = RuntimeConfig::default();
        config.inference.timeout_secs = 30;
        config.inference.ollama_timeout_secs = 120;
        config.inference.extraction_provider = "ollama".into();
        assert_eq!(
            extraction_processing_config(&config, None, None, false)
                .unwrap()
                .request_timeout,
            Duration::from_secs(120)
        );
        config.inference.extraction_provider = "tgi".into();
        assert_eq!(
            extraction_processing_config(&config, None, None, false)
                .unwrap()
                .request_timeout,
            Duration::from_secs(30)
        );
    }

    #[test]
    fn portable_and_reindex_json_summaries_have_stable_machine_fields() {
        let backup = backup::BackupSummary {
            path: PathBuf::from("/tmp/archive"),
            schema_version: 9,
            records: 4,
            record_counts: std::collections::BTreeMap::from([("note".into(), 4)]),
            includes_embeddings: false,
            dry_run: true,
        };
        let backup_json = serde_json::to_value(&backup).unwrap();
        for field in [
            "path",
            "schema_version",
            "records",
            "record_counts",
            "includes_embeddings",
            "dry_run",
        ] {
            assert!(
                backup_json.get(field).is_some(),
                "missing backup field {field}"
            );
        }
        let reindex = ReindexOutput {
            dry_run: true,
            scope: "notes".into(),
            item_count: 4,
            estimated_input_characters: 123,
            provider: "fixture".into(),
            model: "model".into(),
            dimension: 1024,
            job_id: None,
            completed: None,
            cancelled: false,
        };
        let reindex_json = serde_json::to_value(reindex).unwrap();
        for field in [
            "dry_run",
            "scope",
            "item_count",
            "estimated_input_characters",
            "provider",
            "model",
            "dimension",
            "job_id",
            "completed",
            "cancelled",
        ] {
            assert!(
                reindex_json.get(field).is_some(),
                "missing reindex field {field}"
            );
        }
    }

    #[test]
    fn reindex_cli_requires_an_explicit_scope_unless_resuming() {
        let cli = Cli::try_parse_from([
            "graphrag",
            "reindex",
            "--all",
            "--dry-run",
            "--format",
            "json",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Commands::Reindex {
                all: true,
                dry_run: true,
                ..
            }
        ));
        let resume =
            Cli::try_parse_from(["graphrag", "reindex", "--resume", "processing_job:one"]).unwrap();
        assert!(matches!(
            resume.command,
            Commands::Reindex {
                resume: Some(_),
                ..
            }
        ));
    }
}
