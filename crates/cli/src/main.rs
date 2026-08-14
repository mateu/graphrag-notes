//! GraphRAG Notes CLI
//!
//! A command-line interface for the GraphRAG Notes system.

mod doctor;
mod eval;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use eval::{
    build_baseline_comparison, evaluate_ranked_results_with_tokens, load_baseline, load_eval_cases,
    parse_regression_thresholds, AugmentationDiagnosticsReport, EvalCaseReport, EvalMetadata,
    EvalOutputFormat, EvalRunReport, EvalScope, RankedResult, EVAL_SCHEMA_VERSION,
};
use graphrag_agents::{
    AugmentDiagnostics, AugmentOptions, ChatImportMode, ChatIngestOptions, GardenerAgent,
    InferenceProviderConfig, InferenceProviders, LibrarianAgent, LibrarianRuntimeConfig,
    SearchAgent, SearchHitType, SearchScope, SharedEmbedder, SharedEntityExtractor, TokenCountMode,
};
use graphrag_config::{AugmentConfig, CliOverrides, RuntimeConfig, SearchConfig};
use graphrag_core::{record_id_to_string, ChatExport, ProposedEdgeStatus, Source};
use graphrag_db::{
    fusion::{FusionConfig, FusionStrategy},
    init_memory, init_persistent, migrations, parse_record_id, Repository, SourceDeleteSummary,
};
use serde::Serialize;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;
use tracing_subscriber::{EnvFilter, FmtSubscriber};

/// GraphRAG Notes - An evolving knowledge graph for your notes
#[derive(Parser)]
#[command(name = "graphrag")]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Configuration file. Defaults to GRAPHRAG_CONFIG, then
    /// ~/.config/graphrag/config.toml when that file exists.
    #[arg(long, global = true, value_name = "PATH")]
    config: Option<PathBuf>,

    /// Database path (overrides the resolved configuration)
    #[arg(short, long)]
    db_path: Option<PathBuf>,

    /// Use in-memory database (for testing)
    #[arg(long)]
    memory: bool,

    /// Verbose output (DEBUG logs)
    #[arg(short, long)]
    verbose: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Diagnose configuration, database compatibility, and local providers without changing data
    Doctor {
        /// Output format for the diagnostic report
        #[arg(long, value_enum, default_value_t = DoctorFormat::Human)]
        format: DoctorFormat,
    },

    /// Inspect or validate the resolved runtime configuration
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },

    /// Add a new note
    Add {
        /// Note content (reads from stdin if not provided)
        content: Option<String>,

        /// Note title
        #[arg(short, long)]
        title: Option<String>,

        /// Tags (comma-separated)
        #[arg(short = 'T', long)]
        tags: Option<String>,
    },

    /// Import from a file
    Import {
        /// Path to file
        path: PathBuf,

        /// Rebuild the source even when its normalized content hash is unchanged
        #[arg(long)]
        force: bool,
    },

    /// Inspect, remove, or reimport file-backed sources
    Sources {
        #[command(subcommand)]
        command: SourcesCommand,
    },

    /// Import chat conversations from Claude Desktop or other chat exports
    ImportChats {
        /// Path to JSON file containing chat export
        path: PathBuf,

        /// Import mode: qa (default), message, or hybrid
        #[arg(long, value_enum, default_value_t = ImportModeArg::Qa)]
        mode: ImportModeArg,

        /// Skip entity extraction (faster for testing)
        #[arg(long)]
        skip_extraction: bool,
    },

    /// Migrate chat exports into conversation/message tables (and optionally regenerate notes)
    MigrateChats {
        /// Path to JSON file containing chat export
        path: PathBuf,

        /// Preview counts without writing to the database
        #[arg(long)]
        dry_run: bool,

        /// Also regenerate derived notes (idempotent: skips conversations already linked to notes)
        #[arg(long)]
        with_notes: bool,

        /// Note import mode when using --with-notes
        #[arg(long, value_enum, default_value_t = ImportModeArg::Hybrid)]
        mode: ImportModeArg,

        /// Skip entity extraction when using --with-notes
        #[arg(long)]
        skip_extraction: bool,
    },

    /// Search notes, messages, or conversation summaries
    Search {
        /// Search query
        query: String,

        /// Maximum results
        #[arg(short, long)]
        limit: Option<usize>,

        /// Search scope: notes, messages, or all
        #[arg(long, value_enum, default_value_t = SearchScopeArg::Notes)]
        scope: SearchScopeArg,

        /// Restrict results to records updated/created in the last N days
        #[arg(long)]
        since_days: Option<u32>,

        /// Restrict results to a specific source URI
        #[arg(long)]
        source_uri: Option<String>,

        /// Include graph context
        #[arg(short, long)]
        context: bool,
    },

    /// Build prompt-ready augmentation context with citations
    Augment {
        /// Retrieval query
        query: String,

        /// Maximum context chunks to include
        #[arg(short, long)]
        limit: Option<usize>,

        /// Search scope: notes, messages, or all
        #[arg(long, value_enum, default_value_t = SearchScopeArg::All)]
        scope: SearchScopeArg,

        /// Restrict results to records updated/created in the last N days
        #[arg(long)]
        since_days: Option<u32>,

        /// Restrict results to a specific source URI
        #[arg(long)]
        source_uri: Option<String>,

        /// Filter note chunks to notes linked to matching entities
        #[arg(long)]
        entity: Option<String>,

        /// Approximate max tokens across all chunks
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Approximate max tokens per chunk
        #[arg(long)]
        max_chunk_tokens: Option<usize>,
    },

    /// Evaluate augmentation retrieval quality from a JSON/JSONL test set
    EvalAugment {
        /// Path to eval cases (.json array or .jsonl object-per-line)
        path: PathBuf,

        /// Default max chunks when case limit is not set
        #[arg(short, long)]
        limit: Option<usize>,

        /// Default scope when case scope is not set
        #[arg(long, value_enum, default_value_t = SearchScopeArg::All)]
        scope: SearchScopeArg,

        /// Default recency filter when case since_days is not set
        #[arg(long)]
        since_days: Option<u32>,

        /// Default source filter when case source_uri is not set
        #[arg(long)]
        source_uri: Option<String>,

        /// Default global token budget when case max_tokens is not set
        #[arg(long)]
        max_tokens: Option<usize>,

        /// Default per-chunk token budget when case max_chunk_tokens is not set
        #[arg(long)]
        max_chunk_tokens: Option<usize>,

        /// Exit with status 1 when any expected case misses
        #[arg(long)]
        fail_on_miss: bool,

        /// Output format for the evaluation report
        #[arg(long, value_enum, default_value_t = EvalOutputFormat::Human)]
        format: EvalOutputFormat,

        /// Compare aggregate metrics against a prior JSON report
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Allowed metric decrease, such as `recall_at_k=0.02` (repeatable)
        #[arg(
            long = "max-regression",
            value_name = "METRIC=MAX_DROP",
            requires = "baseline"
        )]
        max_regression: Vec<String>,
    },

    /// List recent notes
    List {
        /// Maximum results
        #[arg(short, long, default_value = "20")]
        limit: usize,
    },

    /// Scan and review persistent Gardener edge proposals
    Garden {
        #[command(subcommand)]
        command: GardenCommand,
    },

    /// Show database statistics
    Stats,

    /// Show the database's current and this binary's latest schema version
    SchemaVersion,

    /// Interactive mode
    Interactive,

    /// Show the embedding dimension from the active embeddings provider
    EmbeddingDim {
        /// Optional text to embed (defaults to "dimension probe")
        text: Option<String>,
    },

    /// Extract entities for notes that are missing entity links
    ExtractEntities {
        /// Maximum notes to process (or page size when using --all)
        #[arg(short, long, default_value = "100")]
        limit: usize,

        /// Process all notes (not just those missing mentions)
        #[arg(long)]
        all: bool,

        /// Process only specific note id(s); repeat this flag per note
        #[arg(long = "note-id")]
        note_ids: Vec<String>,

        /// Clear existing mentions before re-extracting (use with --all or --note-id)
        #[arg(long)]
        force: bool,
    },

    /// Show entities linked to a note
    ShowEntities {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Show a note by ID
    ShowNote {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// List note-to-note edges
    ListEdges {
        /// Maximum edges per edge type
        #[arg(short, long, default_value = "10")]
        limit: usize,
    },

    /// Show note-to-note edges for a specific note
    ShowNoteEdges {
        /// Note ID (e.g., note:xxxxxxxx)
        note_id: String,
    },

    /// Safely inspect or remove accepted note-to-note edges
    Edges {
        #[command(subcommand)]
        command: EdgesCommand,
    },

    /// Delete the local database (fresh start)
    ResetDb {
        /// Database path (defaults to ~/.graphrag/data-v3)
        #[arg(short, long)]
        db_path: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum ConfigCommand {
    /// Print the fully resolved configuration (with secrets redacted)
    Show,
    /// Validate configuration and exit without opening the database
    Validate,
}

#[derive(Subcommand)]
enum SourcesCommand {
    /// List source lifecycle state
    List {
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Show one source by record id or normalized URI
    Show {
        id_or_uri: String,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Preview or permanently delete one source and its generated records
    Delete {
        id_or_uri: String,
        /// Print exact mutation counts without changing the database
        #[arg(long)]
        dry_run: bool,
        /// Confirm permanent deletion (required unless --dry-run is set)
        #[arg(long)]
        yes: bool,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
    /// Read the original local file and run the normal staged import flow
    Reimport {
        id_or_uri: String,
        #[arg(long)]
        force: bool,
        #[arg(long, value_enum, default_value_t = SourceOutputFormat::Human)]
        format: SourceOutputFormat,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum SourceOutputFormat {
    Human,
    Json,
}

#[derive(Subcommand)]
enum GardenCommand {
    /// Generate reviewable proposals; accepted edge tables are never mutated
    Scan {
        /// Preview candidate count without persisting proposals
        #[arg(long)]
        dry_run: bool,
    },
    /// Scan and apply the explicitly configured auto-apply policy
    Apply {
        /// Confirm mutation of accepted edges under the configured policy
        #[arg(long, required = true)]
        yes: bool,
    },
    /// Inspect or act on persisted proposals
    Proposals {
        #[command(subcommand)]
        command: ProposalCommand,
    },
}

#[derive(Subcommand)]
enum ProposalCommand {
    List {
        #[arg(long, value_enum)]
        status: Option<ProposalStatusArg>,
        #[arg(short, long, default_value_t = 50)]
        limit: usize,
    },
    Show {
        id: String,
    },
    Accept {
        /// Proposal ID. Omit only with --all.
        id: Option<String>,
        /// Accept every pending Gardener proposal meeting --min-confidence.
        #[arg(long)]
        all: bool,
        /// Required with --all; values are in [0, 1].
        #[arg(long)]
        min_confidence: Option<f32>,
        /// Confirm a mutating acceptance.
        #[arg(long)]
        yes: bool,
        #[arg(long)]
        reason: Option<String>,
    },
    Reject {
        id: String,
        #[arg(long)]
        reason: Option<String>,
        /// Confirm a mutating rejection.
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Subcommand)]
enum EdgesCommand {
    Delete {
        id: String,
        /// Show whether the edge exists without deleting it.
        #[arg(long)]
        dry_run: bool,
        /// Confirm deletion.
        #[arg(long)]
        yes: bool,
    },
    /// Alias for `edges delete`; preserves proposal audit state.
    Undo {
        id: String,
        #[arg(long)]
        dry_run: bool,
        #[arg(long)]
        yes: bool,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ProposalStatusArg {
    Pending,
    Accepting,
    Accepted,
    Rejected,
    Superseded,
}

impl From<ProposalStatusArg> for ProposedEdgeStatus {
    fn from(value: ProposalStatusArg) -> Self {
        match value {
            ProposalStatusArg::Pending => Self::Pending,
            ProposalStatusArg::Accepting => Self::Accepting,
            ProposalStatusArg::Accepted => Self::Accepted,
            ProposalStatusArg::Rejected => Self::Rejected,
            ProposalStatusArg::Superseded => Self::Superseded,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum ImportModeArg {
    Qa,
    Message,
    Hybrid,
}

#[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Eq)]
enum SearchScopeArg {
    Notes,
    Messages,
    All,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DoctorFormat {
    Human,
    Json,
}

fn to_import_mode(mode: ImportModeArg) -> ChatImportMode {
    match mode {
        ImportModeArg::Qa => ChatImportMode::Qa,
        ImportModeArg::Message => ChatImportMode::Message,
        ImportModeArg::Hybrid => ChatImportMode::Hybrid,
    }
}

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

fn configured_search_agent(
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
    SearchAgent::new(repo, embedder).with_fusion_config(
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
}

fn augment_options(
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

fn packing_diagnostics_text(diagnostics: &AugmentDiagnostics) -> String {
    let token_count_mode = match diagnostics.token_count_mode {
        TokenCountMode::Exact => "exact",
        TokenCountMode::Estimated => "estimated",
    };
    format!(
        "token_mode={token_count_mode}; header_tokens={}; dropped_duplicates={}; dropped_near_duplicates={}; dropped_for_relevance={}; dropped_for_budget={}; dropped_for_entity_filter={}",
        diagnostics.header_tokens,
        diagnostics.dropped_duplicates,
        diagnostics.dropped_near_duplicates,
        diagnostics.dropped_for_relevance,
        diagnostics.dropped_for_budget,
        diagnostics.dropped_for_entity_filter,
    )
}

fn librarian_runtime_config(
    config: &RuntimeConfig,
    cli_skip_extraction: bool,
) -> LibrarianRuntimeConfig {
    LibrarianRuntimeConfig {
        min_chunk_size: config.librarian.min_chunk_size,
        max_chunk_size: config.librarian.max_chunk_size,
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

#[tokio::main]
async fn main() -> Result<()> {
    // Load environment variables from .env if present.
    dotenvy::dotenv().ok();

    let cli = Cli::parse();
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
    let providers = InferenceProviders::from_config(&inference_config);
    let tei = providers.embedder;
    let tgi = providers.extractor;

    // Check inference services only when needed
    let needs_tei = matches!(
        cli.command,
        Commands::Add { .. }
            | Commands::Import { .. }
            | Commands::Sources {
                command: SourcesCommand::Reimport { .. },
            }
            | Commands::ImportChats { .. }
            | Commands::MigrateChats { .. }
            | Commands::Search { .. }
            | Commands::Augment { .. }
            | Commands::EvalAugment { .. }
            | Commands::Interactive
    );
    let needs_tgi = matches!(
        &cli.command,
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
    ) && (!skip_extraction
        || matches!(&cli.command, Commands::ExtractEntities { .. }));

    if needs_tei {
        let tei_ok = tei.health().await.unwrap_or(false);
        if !tei_ok {
            eprintln!("Error: embeddings service is not reachable.");
            eprintln!("  TEI (embeddings): {}", tei.capabilities().endpoint);
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Embeddings service unavailable");
        }
    }

    if needs_tgi {
        let tgi_ok = tgi.health().await.unwrap_or(false);
        if !tgi_ok {
            eprintln!("Error: extraction service is not reachable.");
            eprintln!("  TGI (extraction): {}", tgi.capabilities().endpoint);
            eprintln!("Start it with: docker compose up -d");
            anyhow::bail!("Extraction service unavailable");
        }
    }

    // Execute command
    match cli.command {
        Commands::Add {
            content,
            title,
            tags,
        } => {
            cmd_add(repo, tei, tgi, librarian_config, content, title, tags).await?;
        }
        Commands::Import { path, force } => {
            cmd_import(repo, tei, tgi, librarian_config, path, force).await?;
        }
        Commands::Sources { command } => {
            cmd_sources(repo, tei, tgi, librarian_config, command).await?;
        }
        Commands::ImportChats { path, mode, .. } => {
            cmd_import_chats(repo, tei, tgi, librarian_config, path, mode).await?;
        }
        Commands::MigrateChats {
            path,
            dry_run,
            with_notes,
            mode,
            ..
        } => {
            cmd_migrate_chats(
                repo,
                tei,
                tgi,
                librarian_config,
                path,
                dry_run,
                with_notes,
                mode,
            )
            .await?;
        }
        Commands::Search {
            query,
            limit,
            scope,
            since_days,
            source_uri,
            context,
        } => {
            cmd_search(
                repo,
                tei,
                query,
                limit.unwrap_or(config.search.default_limit),
                scope,
                since_days,
                source_uri,
                context,
                config.search.clone(),
            )
            .await?;
        }
        Commands::EvalAugment {
            path,
            limit,
            scope,
            since_days,
            source_uri,
            max_tokens,
            max_chunk_tokens,
            fail_on_miss,
            format,
            baseline,
            max_regression,
        } => {
            cmd_eval_augment(
                repo,
                tei,
                path,
                limit.unwrap_or(config.augment.default_limit),
                scope,
                since_days,
                source_uri,
                max_tokens.unwrap_or(config.augment.max_tokens),
                max_chunk_tokens.unwrap_or(config.augment.max_chunk_tokens),
                config.search.clone(),
                config.augment.clone(),
                fail_on_miss,
                format,
                baseline,
                max_regression,
            )
            .await?;
        }
        Commands::Augment {
            query,
            limit,
            scope,
            since_days,
            source_uri,
            entity,
            max_tokens,
            max_chunk_tokens,
        } => {
            cmd_augment(
                repo,
                tei,
                query,
                limit.unwrap_or(config.augment.default_limit),
                scope,
                since_days,
                source_uri,
                entity,
                max_tokens.unwrap_or(config.augment.max_tokens),
                max_chunk_tokens.unwrap_or(config.augment.max_chunk_tokens),
                config.search.clone(),
                config.augment.clone(),
            )
            .await?;
        }
        Commands::List { limit } => {
            cmd_list(repo, limit).await?;
        }
        Commands::Garden { command } => {
            cmd_garden(
                repo,
                command,
                config.gardener.similarity_threshold,
                config.gardener.auto_apply_threshold,
                config.gardener.auto_apply,
                config.gardener.max_suggestions,
            )
            .await?;
        }
        Commands::Stats => {
            cmd_stats(repo).await?;
        }
        Commands::SchemaVersion => {
            // Handled immediately after database initialization.
        }
        Commands::Interactive => {
            cmd_interactive(
                repo,
                tei,
                tgi,
                librarian_config,
                config.search.default_limit,
                config.search.clone(),
                config.gardener.similarity_threshold,
                config.gardener.auto_apply_threshold,
                config.gardener.auto_apply,
                config.gardener.max_suggestions,
            )
            .await?;
        }
        Commands::ExtractEntities {
            limit,
            all,
            note_ids,
            force,
        } => {
            cmd_extract_entities(
                repo,
                tei,
                tgi,
                librarian_config,
                limit,
                all,
                note_ids,
                force,
            )
            .await?;
        }
        Commands::ShowEntities { note_id } => {
            cmd_show_entities(repo, note_id).await?;
        }
        Commands::ShowNote { note_id } => {
            cmd_show_note(repo, note_id).await?;
        }
        Commands::ListEdges { limit } => {
            cmd_list_edges(repo, limit).await?;
        }
        Commands::ShowNoteEdges { note_id } => {
            cmd_show_note_edges(repo, note_id).await?;
        }
        Commands::Edges { command } => {
            cmd_edges(repo, command).await?;
        }
        Commands::EmbeddingDim { .. } => {
            // Handled before database init.
        }
        Commands::Config { .. } => unreachable!("configuration commands return before startup"),
        Commands::ResetDb { .. } => {
            // Handled before database init.
        }
        Commands::Doctor { .. } => unreachable!("doctor returns before database initialization"),
    }

    Ok(())
}

async fn cmd_add(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    content: Option<String>,
    title: Option<String>,
    tags: Option<String>,
) -> Result<()> {
    let content = match content {
        Some(c) => c,
        None => {
            // Read from stdin
            eprintln!("Enter note content (Ctrl+D to finish):");
            let stdin = io::stdin();
            let lines: Vec<String> = stdin.lock().lines().filter_map(|l| l.ok()).collect();
            lines.join("\n")
        }
    };

    if content.trim().is_empty() {
        anyhow::bail!("Note content cannot be empty");
    }

    let tags = tags
        .map(|t| t.split(',').map(|s| s.trim().to_string()).collect())
        .unwrap_or_default();

    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let note = librarian.ingest_text(content, title, tags).await?;

    println!(
        "✓ Created note: {}",
        note.id
            .as_ref()
            .map(record_id_to_string)
            .unwrap_or_else(|| "(no id)".to_string())
    );

    Ok(())
}

async fn cmd_import(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    path: PathBuf,
    force: bool,
) -> Result<()> {
    let source_path = import_path_utf8(&path)?;
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let summary = librarian
        .ingest_markdown_with_options(source_path, content, force)
        .await?;

    print_import_summary(&summary, SourceOutputFormat::Human)?;

    Ok(())
}

/// File-source identities are UTF-8 URI strings. Reject non-UTF-8 paths at
/// the CLI boundary rather than collapsing distinct native paths to a sentinel
/// such as `unknown`.
fn import_path_utf8(path: &std::path::Path) -> Result<&str> {
    path.to_str().ok_or_else(|| {
        anyhow::anyhow!(
            "cannot import a path that is not valid UTF-8: {}; rename the file or use a UTF-8 path",
            path.display()
        )
    })
}

#[derive(Serialize)]
struct ImportSummaryOutput<'a> {
    source_id: &'a str,
    source_uri: &'a str,
    generation: u64,
    action: &'a str,
    created: u64,
    unchanged: u64,
    updated: u64,
    deleted: u64,
    failed: u64,
    cleanup: &'a SourceDeleteSummary,
}

fn print_import_summary(
    result: &graphrag_agents::MarkdownImportResult,
    format: SourceOutputFormat,
) -> Result<()> {
    let action = match result.action {
        graphrag_db::SourceImportAction::Created => "created",
        graphrag_db::SourceImportAction::Updated => "updated",
        graphrag_db::SourceImportAction::Unchanged => "unchanged",
    };
    let output = ImportSummaryOutput {
        source_id: &result.source_id,
        source_uri: &result.source_uri,
        generation: result.generation,
        action,
        created: result.created,
        unchanged: result.unchanged,
        updated: result.updated,
        deleted: result.deleted,
        failed: result.failed,
        cleanup: &result.cleanup,
    };
    match format {
        SourceOutputFormat::Human => println!(
            "Source {} generation {}: {} (created {}, unchanged {}, updated {}, deleted {}, failed {}; cleanup notes {}, mentions {}, note edges {}, proposals {}, conversation provenance {}, message provenance {})",
            output.source_uri,
            output.generation,
            output.action,
            output.created,
            output.unchanged,
            output.updated,
            output.deleted,
            output.failed,
            output.cleanup.notes,
            output.cleanup.mentions,
            output.cleanup.note_edges,
            output.cleanup.proposals,
            output.cleanup.note_conversation_provenance,
            output.cleanup.note_message_provenance,
        ),
        SourceOutputFormat::Json => println!("{}", serde_json::to_string(&output)?),
    }
    Ok(())
}

async fn cmd_sources(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    command: SourcesCommand,
) -> Result<()> {
    match command {
        SourcesCommand::List { format } => {
            let sources = repo.list_sources().await?;
            match format {
                SourceOutputFormat::Json => println!("{}", serde_json::to_string(&sources)?),
                SourceOutputFormat::Human => {
                    for source in sources {
                        println!(
                            "{}\t{}\t{}\tgeneration={} successful={} status={:?}",
                            source
                                .id
                                .as_ref()
                                .map(record_id_to_string)
                                .unwrap_or_default(),
                            source.normalized_uri.or(source.uri).unwrap_or_default(),
                            format!("{:?}", source.source_type).to_lowercase(),
                            source.generation,
                            source.successful_generation,
                            source.status,
                        );
                    }
                }
            }
        }
        SourcesCommand::Show { id_or_uri, format } => {
            let source = repo
                .get_source(&id_or_uri)
                .await?
                .ok_or_else(|| anyhow::anyhow!("source not found: {id_or_uri}"))?;
            match format {
                SourceOutputFormat::Json => println!("{}", serde_json::to_string(&source)?),
                SourceOutputFormat::Human => println!(
                    "{}\nuri: {}\ngeneration: {} (last successful {})\nstatus: {:?}\nlast error: {}",
                    source.id.as_ref().map(record_id_to_string).unwrap_or_default(),
                    source.normalized_uri.or(source.uri).unwrap_or_default(),
                    source.generation,
                    source.successful_generation,
                    source.status,
                    source.last_error.unwrap_or_else(|| "none".to_string()),
                ),
            }
        }
        SourcesCommand::Delete {
            id_or_uri,
            dry_run,
            yes,
            format,
        } => {
            let source = repo
                .get_source(&id_or_uri)
                .await?
                .ok_or_else(|| anyhow::anyhow!("source not found: {id_or_uri}"))?;
            let summary = repo.preview_source_delete(&source).await?;
            // Safe by default: omitting both flags prints the same exact
            // preview as --dry-run. Only --yes authorizes mutation.
            let dry_run = delete_is_dry_run(dry_run, yes);
            if !dry_run {
                repo.delete_source(&source).await?;
            }
            print_delete_summary(&source, &summary, dry_run, format)?;
        }
        SourcesCommand::Reimport {
            id_or_uri,
            force,
            format,
        } => {
            let librarian =
                LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
            let summary = librarian
                .reimport_markdown_source(&id_or_uri, force)
                .await?;
            print_import_summary(&summary, format)?;
        }
    }
    Ok(())
}

fn delete_is_dry_run(dry_run: bool, yes: bool) -> bool {
    dry_run || !yes
}

fn print_delete_summary(
    source: &Source,
    summary: &SourceDeleteSummary,
    dry_run: bool,
    format: SourceOutputFormat,
) -> Result<()> {
    #[derive(Serialize)]
    struct Output<'a> {
        source_id: String,
        dry_run: bool,
        summary: &'a SourceDeleteSummary,
    }
    let output = Output {
        source_id: source
            .id
            .as_ref()
            .map(record_id_to_string)
            .unwrap_or_default(),
        dry_run,
        summary,
    };
    match format {
        SourceOutputFormat::Json => println!("{}", serde_json::to_string(&output)?),
        SourceOutputFormat::Human => println!(
            "{} source {}: notes={}, mentions={}, note_edges={}, proposals={}, conversation_provenance={}, message_provenance={}",
            if dry_run { "Would delete" } else { "Deleted" },
            output.source_id,
            summary.notes,
            summary.mentions,
            summary.note_edges,
            summary.proposals,
            summary.note_conversation_provenance,
            summary.note_message_provenance,
        ),
    }
    Ok(())
}

async fn cmd_import_chats(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    path: PathBuf,
    mode: ImportModeArg,
) -> Result<()> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    // Parse the chat export
    let export = ChatExport::from_json(&content)
        .with_context(|| format!("Failed to parse chat export from: {}", path.display()))?;

    println!(
        "Found {} conversations with {} total messages",
        export.conversation_count(),
        export.total_messages()
    );
    let conversations_with_messages = export
        .conversations
        .iter()
        .filter(|conv| !conv.messages.is_empty())
        .count();
    let summary_only = export
        .conversations
        .iter()
        .filter(|conv| conv.messages.is_empty() && !conv.summary.is_empty())
        .count();
    println!(
        "  • With messages: {}, without messages: {}, summary-only: {}",
        conversations_with_messages,
        export.conversation_count() - conversations_with_messages,
        summary_only
    );

    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let mode = to_import_mode(mode);
    let result = librarian
        .ingest_chat_export(export, Some(path.display().to_string()), mode)
        .await?;

    println!("\n✓ Import complete:");
    println!(
        "  • Conversations imported: {}",
        result.conversations_imported
    );
    println!("  • Conversations failed: {}", result.conversations_failed);
    println!("  • Conversations total: {}", result.conversations_total);
    println!(
        "  • Conversations with messages: {}",
        result.conversations_with_messages
    );
    println!(
        "  • Conversations without messages: {}",
        result.conversations_without_messages
    );
    println!(
        "  • Summary-only conversations: {}",
        result.conversations_summary_only
    );
    println!("  • Messages seen: {}", result.messages_total);
    println!("  • Notes created: {}", result.notes_created);
    println!("    - From Q&A: {}", result.notes_from_qa);
    println!("    - From messages: {}", result.notes_from_messages);
    println!("    - From summaries: {}", result.notes_from_summaries);
    println!(
        "    - From fallback chunking: {}",
        result.notes_from_fallback
    );
    println!("  • Q&A pairs created: {}", result.qa_pairs_created);
    println!(
        "    - Dropped (short question): {}",
        result.qa_pairs_dropped_short_question
    );
    println!(
        "    - Dropped (short answer): {}",
        result.qa_pairs_dropped_short_answer
    );
    println!(
        "    - Assistant without pending question: {}",
        result.assistant_without_human
    );
    println!(
        "    - Conversations with trailing unpaired human turn: {}",
        result.trailing_unpaired_human
    );
    println!(
        "  • Conversation records upserted: {}",
        result.conversation_records_upserted
    );
    println!(
        "  • Message records upserted: {}",
        result.message_records_upserted
    );
    println!(
        "  • Note→Conversation links created: {}",
        result.note_conversation_links_created
    );
    println!(
        "  • Note→Message links created: {}",
        result.note_message_links_created
    );

    if result.conversations_failed > 0 {
        for error in &result.errors {
            println!("    - {}", error);
        }
    }

    Ok(())
}

async fn cmd_migrate_chats(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    path: PathBuf,
    dry_run: bool,
    with_notes: bool,
    mode: ImportModeArg,
) -> Result<()> {
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;
    let export = ChatExport::from_json(&content)
        .with_context(|| format!("Failed to parse chat export from: {}", path.display()))?;
    let mode = to_import_mode(mode);

    let preview = LibrarianAgent::preview_chat_export(&export, mode, with_notes);
    println!("Migration preview:");
    println!("  • Conversations total: {}", preview.conversations_total);
    println!(
        "  • With messages: {}, without messages: {}, summary-only: {}",
        preview.conversations_with_messages,
        preview.conversations_without_messages,
        preview.summary_only_conversations
    );
    println!("  • Messages total: {}", preview.messages_total);
    println!("  • Q&A pairs: {}", preview.qa_pairs);
    println!(
        "  • Q&A dropped (short question/answer): {}/{}",
        preview.qa_pairs_dropped_short_question, preview.qa_pairs_dropped_short_answer
    );
    if with_notes {
        println!(
            "  • Estimated notes to generate: {}",
            preview.estimated_notes
        );
        println!("    - From Q&A: {}", preview.notes_from_qa);
        println!("    - From messages: {}", preview.notes_from_messages);
        println!("    - From summaries: {}", preview.notes_from_summaries);
        println!(
            "    - From fallback chunking: {}",
            preview.notes_from_fallback
        );
    } else {
        println!("  • Mode: records-only backfill (no new notes)");
    }

    if dry_run {
        println!("\nDry run complete. No writes performed.");
        return Ok(());
    }

    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let result = if with_notes {
        librarian
            .ingest_chat_export_with_options(
                export,
                Some(path.display().to_string()),
                mode,
                ChatIngestOptions {
                    persist_notes: true,
                    skip_notes_if_linked: true,
                },
            )
            .await?
    } else {
        librarian
            .backfill_chat_export_records(export, Some(path.display().to_string()))
            .await?
    };

    println!("\n✓ Migration complete:");
    println!(
        "  • Conversations imported: {}",
        result.conversations_imported
    );
    println!("  • Conversations failed: {}", result.conversations_failed);
    println!(
        "  • Conversation records upserted: {}",
        result.conversation_records_upserted
    );
    println!(
        "  • Message records upserted: {}",
        result.message_records_upserted
    );
    println!("  • Notes created: {}", result.notes_created);
    println!(
        "  • Note→Conversation links created: {}",
        result.note_conversation_links_created
    );
    println!(
        "  • Note→Message links created: {}",
        result.note_message_links_created
    );

    if result.conversations_failed > 0 {
        for error in &result.errors {
            println!("    - {}", error);
        }
    }

    Ok(())
}

async fn cmd_extract_entities(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    limit: usize,
    all: bool,
    note_ids: Vec<String>,
    force: bool,
) -> Result<()> {
    if all && !note_ids.is_empty() {
        anyhow::bail!("--all and --note-id cannot be used together");
    }
    if force && !all && note_ids.is_empty() {
        anyhow::bail!("--force requires --all or at least one --note-id");
    }
    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let processed = if !note_ids.is_empty() {
        librarian
            .extract_entities_for_note_ids(&note_ids, force)
            .await?
    } else if all {
        let processed = librarian
            .extract_entities_for_all_notes(limit, force)
            .await?;
        processed
    } else {
        let processed = librarian.extract_entities_for_notes(limit).await?;
        processed
    };
    println!("✓ Extracted entities for {} notes", processed);
    Ok(())
}

async fn cmd_show_entities(repo: Repository, note_id: String) -> Result<()> {
    let key = note_id.strip_prefix("note:").unwrap_or(&note_id);
    let note = repo
        .get_note(key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Note not found: {}", note_id))?;

    let title = note.title.as_deref().unwrap_or("(untitled)");
    println!("Note: {} ({})", title, note_id);

    let entities = repo.get_entities_for_note(key).await?;
    if entities.is_empty() {
        println!("No entities linked to this note.");
        return Ok(());
    }

    println!("Entities ({}):", entities.len());
    for entity in entities {
        let entity_type = serde_json::to_string(&entity.entity_type)
            .unwrap_or_else(|_| "\"other\"".to_string())
            .trim_matches('"')
            .to_string();
        println!("  • {} [{}]", entity.name, entity_type);
    }

    Ok(())
}

async fn cmd_show_note(repo: Repository, note_id: String) -> Result<()> {
    let key = note_id.strip_prefix("note:").unwrap_or(&note_id);
    let note = repo
        .get_note(key)
        .await?
        .ok_or_else(|| anyhow::anyhow!("Note not found: {}", note_id))?;

    println!(
        "Note: {} ({})",
        note.title.as_deref().unwrap_or("(untitled)"),
        note_id
    );
    println!("Type: {:?}", note.note_type);
    println!(
        "Tags: {}",
        if note.tags.is_empty() {
            "(none)".into()
        } else {
            note.tags.join(", ")
        }
    );
    println!();
    println!("{}", note.content);
    Ok(())
}

async fn cmd_list_edges(repo: Repository, limit: usize) -> Result<()> {
    let edges = repo.list_note_edges(limit).await?;
    if edges.is_empty() {
        println!("No note edges found.");
        return Ok(());
    }

    println!("Note edges (up to {} per type):", limit);
    for edge in edges {
        let reason = edge.reason.as_deref().unwrap_or("");
        let confidence = edge
            .confidence
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "-".into());
        println!(
            "  • {}: {} -> {} (confidence: {}){}",
            edge.edge_type,
            record_id_to_string(&edge.in_id),
            record_id_to_string(&edge.out_id),
            confidence,
            if reason.is_empty() {
                "".into()
            } else {
                format!(" reason: {}", reason)
            }
        );
    }
    Ok(())
}

async fn cmd_show_note_edges(repo: Repository, note_id: String) -> Result<()> {
    let edges = repo.get_note_edges(&note_id).await?;
    if edges.is_empty() {
        println!("No edges found for {}", note_id);
        return Ok(());
    }

    println!("Edges for {}:", note_id);
    for edge in edges {
        let reason = edge.reason.as_deref().unwrap_or("");
        let confidence = edge
            .confidence
            .map(|c| format!("{:.2}", c))
            .unwrap_or_else(|| "-".into());
        println!(
            "  • {}: {} -> {} (confidence: {}){}",
            edge.edge_type,
            record_id_to_string(&edge.in_id),
            record_id_to_string(&edge.out_id),
            confidence,
            if reason.is_empty() {
                "".into()
            } else {
                format!(" reason: {}", reason)
            }
        );
    }
    Ok(())
}

async fn cmd_search(
    repo: Repository,
    tei: SharedEmbedder,
    query: String,
    limit: usize,
    scope: SearchScopeArg,
    since_days: Option<u32>,
    source_uri: Option<String>,
    context: bool,
    search_config: SearchConfig,
) -> Result<()> {
    let search = configured_search_agent(repo, tei, &search_config);
    let scope = match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    };

    if context && scope == SearchScope::Notes {
        let results = search
            .search_with_context_filtered(&query, limit, since_days, source_uri)
            .await?;

        if results.is_empty() {
            println!("No results found.");
            return Ok(());
        }

        println!("Found {} results:\n", results.len());

        for (i, result) in results.iter().enumerate() {
            let r = &result.result;
            println!("{}. {}", i + 1, r.title.as_deref().unwrap_or("(untitled)"));
            println!("   ID: {}", record_id_to_string(&r.id));
            println!("   Type: {}", r.note_type);

            // Truncate content for display
            let preview: String = r.content.chars().take(200).collect();
            println!(
                "   {}{}",
                preview,
                if r.content.len() > 200 { "..." } else { "" }
            );

            if let Some(ref related) = result.related {
                let total =
                    related.supporting.len() + related.contradicting.len() + related.related.len();
                if total > 0 {
                    println!("   → {} related notes", total);
                }
            }

            println!();
        }
    } else {
        if context && scope != SearchScope::Notes {
            println!("Context is only available for notes scope; continuing without context.\n");
        }

        let results = search
            .search_with_scope(&query, limit, scope, since_days, source_uri)
            .await?;

        if results.is_empty() {
            println!("No results found.");
            return Ok(());
        }

        println!("Found {} results:\n", results.len());

        for (i, r) in results.iter().enumerate() {
            let kind = match r.hit_type {
                SearchHitType::Note => "note",
                SearchHitType::Message => "message",
                SearchHitType::ConversationSummary => "conversation-summary",
            };
            println!(
                "{}. [{}] {}",
                i + 1,
                kind,
                r.title.as_deref().unwrap_or("(untitled)")
            );
            println!("   ID: {}", r.id);
            println!("   Score: {:.3}", r.score);
            if let Some(ref conversation_uuid) = r.conversation_uuid {
                println!("   Conversation UUID: {}", conversation_uuid);
            }
            if let Some(message_index) = r.message_index {
                println!("   Message #: {}", message_index + 1);
            }
            if let Some(ref role) = r.role {
                println!("   Role: {}", role);
            }
            if let Some(created_at) = r.created_at {
                println!("   Created/Updated: {}", created_at.to_rfc3339());
            }

            let preview: String = r.content.chars().take(200).collect();
            println!(
                "   {}{}",
                preview,
                if r.content.len() > 200 { "..." } else { "" }
            );
            println!();
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_augment(
    repo: Repository,
    tei: SharedEmbedder,
    query: String,
    limit: usize,
    scope: SearchScopeArg,
    since_days: Option<u32>,
    source_uri: Option<String>,
    entity: Option<String>,
    max_tokens: usize,
    max_chunk_tokens: usize,
    search_config: SearchConfig,
    augment_config: AugmentConfig,
) -> Result<()> {
    if entity.is_some() && scope != SearchScopeArg::Notes {
        anyhow::bail!("--entity currently requires --scope notes");
    }

    let scope = match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    };

    let search = configured_search_agent(repo, tei, &search_config);
    let ctx = search
        .build_augmented_context(
            &query,
            scope,
            since_days,
            source_uri,
            entity.clone(),
            augment_options(limit, max_tokens, max_chunk_tokens, &augment_config),
        )
        .await?;

    println!("Augmentation context:");
    println!("  • Query: {}", ctx.query);
    println!("  • Scope: {:?}", ctx.scope);
    if let Some(filter) = ctx.entity_filter.as_deref() {
        println!("  • Entity filter: {}", filter);
    }
    println!("  • Chunks selected: {}", ctx.chunks.len());
    println!("  • Rendered tokens used: {}", ctx.total_tokens);
    println!(
        "  • Packing diagnostics: {}",
        packing_diagnostics_text(&ctx.diagnostics)
    );

    if ctx.chunks.is_empty() {
        println!("No augmentation context found.");
        return Ok(());
    }

    println!("\nPrompt-ready context block:\n");
    println!("{}", ctx.render_prompt_block());

    println!("\nCitations:");
    for chunk in &ctx.chunks {
        let hit_kind = match chunk.hit_type {
            SearchHitType::Note => "note",
            SearchHitType::Message => "message",
            SearchHitType::ConversationSummary => "conversation-summary",
        };
        let mut provenance = format!("id={}", chunk.id);
        if let Some(conversation_uuid) = chunk.conversation_uuid.as_ref() {
            provenance.push_str(&format!(", conversation_uuid={}", conversation_uuid));
        }
        if let Some(message_index) = chunk.message_index {
            provenance.push_str(&format!(", message_index={}", message_index + 1));
        }
        if let Some(role) = chunk.role.as_ref() {
            provenance.push_str(&format!(", role={}", role));
        }
        if let Some(created_at) = chunk.created_at {
            provenance.push_str(&format!(", created_at={}", created_at.to_rfc3339()));
        }
        println!(
            "  [C{}] {} | score={:.3} | tokens={} | {}",
            chunk.citation, hit_kind, chunk.score, chunk.approx_tokens, provenance
        );
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn cmd_eval_augment(
    repo: Repository,
    tei: SharedEmbedder,
    path: PathBuf,
    default_limit: usize,
    default_scope: SearchScopeArg,
    default_since_days: Option<u32>,
    default_source_uri: Option<String>,
    default_max_tokens: usize,
    default_max_chunk_tokens: usize,
    search_config: SearchConfig,
    augment_config: AugmentConfig,
    fail_on_miss: bool,
    format: EvalOutputFormat,
    baseline_path: Option<PathBuf>,
    max_regression: Vec<String>,
) -> Result<()> {
    let cases = load_eval_cases(&path)?;
    if cases.is_empty() {
        anyhow::bail!("No eval cases found in {}", path.display());
    }
    let thresholds = parse_regression_thresholds(&max_regression)?;
    if !thresholds.is_empty() && baseline_path.is_none() {
        anyhow::bail!("--max-regression requires --baseline");
    }

    let capabilities = tei.capabilities();
    let provider = capabilities.provider;
    let model = capabilities.model;
    let search = configured_search_agent(repo, tei, &search_config);
    let mut reports = Vec::with_capacity(cases.len());

    if matches!(format, EvalOutputFormat::Human) {
        println!("Running {} eval cases from {}", cases.len(), path.display());
        println!();
    }

    for (idx, case) in cases.iter().enumerate() {
        let scope = case
            .scope
            .map(eval_scope_to_search_scope)
            .unwrap_or_else(|| search_scope_arg_to_scope(default_scope));
        let limit = case.limit.unwrap_or(default_limit);
        let k = case.resolved_k(default_limit);
        let since_days = case.since_days.or(default_since_days);
        let source_uri = case
            .source_uri
            .clone()
            .or_else(|| default_source_uri.clone());
        let max_tokens = case.max_tokens.unwrap_or(default_max_tokens);
        let max_chunk_tokens = case.max_chunk_tokens.unwrap_or(default_max_chunk_tokens);

        let started = Instant::now();
        let ctx = search
            .build_augmented_context(
                &case.query,
                scope,
                since_days,
                source_uri,
                case.entity.clone(),
                augment_options(limit, max_tokens, max_chunk_tokens, &augment_config),
            )
            .await?;
        let latency_ms = started.elapsed().as_millis().try_into().unwrap_or(u64::MAX);
        let ranked: Vec<RankedResult> = ctx
            .chunks
            .iter()
            .map(|chunk| RankedResult {
                id: chunk.id.clone(),
                text: format!(
                    "{}\n{}",
                    chunk.title.as_deref().unwrap_or("(untitled)"),
                    chunk.snippet
                ),
                source_uri: chunk.source_uri.clone(),
                conversation_uuid: chunk.conversation_uuid.clone(),
                approx_tokens: chunk.approx_tokens,
            })
            .collect();
        let metrics =
            evaluate_ranked_results_with_tokens(case, &ranked, k, latency_ms, ctx.total_tokens);

        if matches!(format, EvalOutputFormat::Human) {
            let status = match metrics.checks_passed {
                Some(true) => "PASS",
                Some(false) => "MISS",
                None => "UNSCORED",
            };
            println!(
                "{}. {} [{}] k={} chunks={} tokens={} latency={}ms | packing: {}",
                idx + 1,
                case.display_name(),
                status,
                metrics.k,
                metrics.chunks,
                metrics.tokens,
                metrics.latency_ms,
                packing_diagnostics_text(&ctx.diagnostics),
            );
        }
        reports.push(EvalCaseReport {
            name: case.display_name().to_string(),
            query: case.query.clone(),
            metrics,
            augmentation: Some(AugmentationDiagnosticsReport {
                token_count_mode: match ctx.diagnostics.token_count_mode {
                    TokenCountMode::Exact => "exact",
                    TokenCountMode::Estimated => "estimated",
                }
                .to_string(),
                header_tokens: ctx.diagnostics.header_tokens,
                dropped_duplicates: ctx.diagnostics.dropped_duplicates,
                dropped_near_duplicates: ctx.diagnostics.dropped_near_duplicates,
                dropped_for_relevance: ctx.diagnostics.dropped_for_relevance,
                dropped_for_budget: ctx.diagnostics.dropped_for_budget,
                dropped_for_entity_filter: ctx.diagnostics.dropped_for_entity_filter,
            }),
        });
    }

    let metadata = EvalMetadata {
        schema_version: EVAL_SCHEMA_VERSION,
        provider,
        model,
    };
    let mut report = EvalRunReport::from_cases(metadata, reports);
    if let Some(baseline_path) = baseline_path {
        let baseline = load_baseline(&baseline_path)?;
        report.baseline = Some(build_baseline_comparison(&report, &baseline, &thresholds)?);
    }
    let regressions = report
        .baseline
        .as_ref()
        .map(|comparison| comparison.regressions.as_slice())
        .unwrap_or_default();

    match format {
        EvalOutputFormat::Human => {
            println!();
            print!("{}", report.human_report());
        }
        EvalOutputFormat::Json => println!("{}", serde_json::to_string_pretty(&report)?),
    }

    if fail_on_miss && report.summary.cases_missed > 0 {
        anyhow::bail!(
            "Eval failed: {} case(s) missed expectations",
            report.summary.cases_missed
        );
    }
    if !regressions.is_empty() {
        anyhow::bail!(
            "Eval failed: {} configured regression threshold(s) exceeded",
            regressions.len()
        );
    }

    Ok(())
}

fn eval_scope_to_search_scope(scope: EvalScope) -> SearchScope {
    match scope {
        EvalScope::Notes => SearchScope::Notes,
        EvalScope::Messages => SearchScope::Messages,
        EvalScope::All => SearchScope::All,
    }
}

fn search_scope_arg_to_scope(scope: SearchScopeArg) -> SearchScope {
    match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    }
}

async fn cmd_list(repo: Repository, limit: usize) -> Result<()> {
    let notes = repo.list_notes(limit).await?;

    if notes.is_empty() {
        println!("No notes yet. Add one with: graphrag add \"your note\"");
        return Ok(());
    }

    println!("Recent notes ({}):\n", notes.len());

    for note in notes {
        let title = note.title.as_deref().unwrap_or("(untitled)");
        let id = record_id_to_string(&note.id);
        let preview: String = note.content.chars().take(80).collect();

        println!("• {} [{}]", title, id);
        println!(
            "  {}{}",
            preview,
            if note.content.len() > 80 { "..." } else { "" }
        );
        println!();
    }

    Ok(())
}

async fn cmd_garden(
    repo: Repository,
    command: GardenCommand,
    similarity_threshold: f32,
    auto_apply_threshold: f32,
    auto_apply: bool,
    max_suggestions: usize,
) -> Result<()> {
    let gardener = GardenerAgent::new(repo.clone())
        .with_threshold(similarity_threshold)
        .with_auto_apply_policy(auto_apply, auto_apply_threshold)
        .with_max_suggestions(max_suggestions);

    match command {
        GardenCommand::Scan { dry_run } => {
            let report = gardener.scan(dry_run).await?;
            if report.dry_run {
                println!("Dry run: no proposals or accepted edges were changed.");
            } else {
                println!(
                    "Scan persisted {} proposal(s); accepted edges were not changed.",
                    report.proposals.len()
                );
            }
            println!("  • Orphans found: {}", report.orphans_found);
            println!(
                "  • Similarity suggestions: {}",
                report.suggestions_generated
            );
        }
        GardenCommand::Apply { yes } => {
            if !yes {
                anyhow::bail!("refusing to apply Gardener proposals without --yes");
            }
            if !auto_apply {
                anyhow::bail!("Gardener auto-apply is disabled; set [gardener].auto_apply = true before using `garden apply`");
            }
            let report = gardener.run_maintenance().await?;
            println!(
                "Maintenance applied {} policy-approved related_to proposal(s).",
                report.connections_applied
            );
            println!("  • Orphans found: {}", report.orphans_found);
            println!("  • Proposals generated: {}", report.suggestions_generated);
            println!("  • Orphans remaining: {}", report.orphans_remaining);
        }
        GardenCommand::Proposals { command } => {
            cmd_proposals(repo, command).await?;
        }
    }

    Ok(())
}

async fn cmd_proposals(repo: Repository, command: ProposalCommand) -> Result<()> {
    match command {
        ProposalCommand::List { status, limit } => {
            let proposals = repo
                .list_edge_proposals(status.map(Into::into), limit)
                .await?;
            if proposals.is_empty() {
                println!("No matching edge proposals.");
                return Ok(());
            }
            for proposal in proposals {
                println!(
                    "{}  {}  {} → {}  {:.0}%  {}",
                    proposal
                        .id
                        .as_ref()
                        .map(record_id_to_string)
                        .unwrap_or_default(),
                    proposal.status,
                    record_id_to_string(&proposal.from_id),
                    record_id_to_string(&proposal.to_id),
                    proposal.confidence * 100.0,
                    proposal.reason,
                );
            }
        }
        ProposalCommand::Show { id } => {
            let id = parse_record_id(&id, Some("proposed_edge"))?;
            let proposal = repo.get_edge_proposal(&id).await?.ok_or_else(|| {
                anyhow::anyhow!("proposal {} was not found", record_id_to_string(&id))
            })?;
            println!("ID: {}", record_id_to_string(&id));
            println!("Status: {}", proposal.status);
            println!(
                "Edge: {} {} → {}",
                proposal.edge_type,
                record_id_to_string(&proposal.from_id),
                record_id_to_string(&proposal.to_id)
            );
            println!("Confidence: {:.1}%", proposal.confidence * 100.0);
            println!("Reason: {}", proposal.reason);
            println!(
                "Generator: {}{}",
                proposal.generator,
                proposal
                    .generator_version
                    .as_deref()
                    .map(|version| format!(" ({version})"))
                    .unwrap_or_default()
            );
            if let Some(edge_id) = proposal.resulting_edge_id.as_ref() {
                println!("Accepted edge: {}", record_id_to_string(edge_id));
            }
            if let Some(reason) = proposal.action_reason.as_deref() {
                println!("Action reason: {reason}");
            }
            if let Some(reason) = proposal.supersession_reason.as_deref() {
                println!("Supersession reason: {reason}");
            }
        }
        ProposalCommand::Accept {
            id,
            all,
            min_confidence,
            yes,
            reason,
        } => {
            if !yes {
                anyhow::bail!("refusing to accept proposals without --yes");
            }
            if all {
                if id.is_some() {
                    anyhow::bail!("use either a proposal id or --all, not both");
                }
                let threshold = min_confidence
                    .ok_or_else(|| anyhow::anyhow!("--all requires --min-confidence"))?;
                if !(0.0..=1.0).contains(&threshold) {
                    anyhow::bail!("--min-confidence must be between 0 and 1");
                }
                let count = repo
                    .accept_gardener_proposals_above_with_audit(
                        threshold,
                        Some("cli batch acceptance".into()),
                        reason.unwrap_or_else(|| "explicit CLI batch acceptance".into()),
                        true,
                    )
                    .await?;
                println!("Accepted {count} policy-approved related_to proposal(s).");
            } else {
                if min_confidence.is_some() {
                    anyhow::bail!("--min-confidence is only valid with --all");
                }
                let id = id.ok_or_else(|| anyhow::anyhow!("provide a proposal id or --all"))?;
                let id = parse_record_id(&id, Some("proposed_edge"))?;
                let proposal = repo
                    .accept_edge_proposal(&id, Some("cli".into()), reason, true)
                    .await?;
                println!(
                    "Proposal {} is {}.",
                    record_id_to_string(&id),
                    proposal.status
                );
            }
        }
        ProposalCommand::Reject { id, reason, yes } => {
            if !yes {
                anyhow::bail!("refusing to reject a proposal without --yes");
            }
            let id = parse_record_id(&id, Some("proposed_edge"))?;
            let proposal = repo
                .reject_edge_proposal(&id, Some("cli".into()), reason)
                .await?;
            println!(
                "Proposal {} is {}.",
                record_id_to_string(&id),
                proposal.status
            );
        }
    }
    Ok(())
}

async fn cmd_edges(repo: Repository, command: EdgesCommand) -> Result<()> {
    let (id, dry_run, yes) = match command {
        EdgesCommand::Delete { id, dry_run, yes } | EdgesCommand::Undo { id, dry_run, yes } => {
            (id, dry_run, yes)
        }
    };
    let id = parse_record_id(&id, None)?;
    if !matches!(
        id.table.as_str(),
        "supports" | "contradicts" | "derived_from" | "related_to"
    ) {
        anyhow::bail!("{} is not a note-edge id", record_id_to_string(&id));
    }
    if dry_run {
        println!(
            "{}",
            edge_dry_run_message(&record_id_to_string(&id), repo.note_edge_exists(&id).await?)
        );
        return Ok(());
    }
    if !yes {
        anyhow::bail!("refusing to delete an edge without --yes (or use --dry-run)");
    }
    if repo
        .undo_edge(&id, Some("edge deleted through CLI".into()))
        .await?
    {
        println!(
            "Deleted {} and preserved its proposal audit trail.",
            record_id_to_string(&id)
        );
    } else {
        println!(
            "{} was already absent; no changes made.",
            record_id_to_string(&id)
        );
    }
    Ok(())
}

fn edge_dry_run_message(id: &str, exists: bool) -> String {
    if exists {
        format!("Dry run: {id} exists and would be deleted; no changes made.")
    } else {
        format!("Dry run: {id} is absent; no changes made.")
    }
}

async fn cmd_stats(repo: Repository) -> Result<()> {
    let stats = repo.get_stats().await?;

    println!("Database Statistics:");
    println!("  • Notes: {}", stats.note_count);
    println!("  • Entities: {}", stats.entity_count);
    println!("  • Sources: {}", stats.source_count);
    println!("  • Conversations: {}", stats.conversation_count);
    println!("  • Messages: {}", stats.message_count);
    println!("  • Mentions: {}", stats.mention_count);
    println!(
        "  • Note→Conversation links: {}",
        stats.note_conversation_link_count
    );
    println!("  • Note→Message links: {}", stats.note_message_link_count);
    println!("  • Edges: {}", stats.edge_count);

    Ok(())
}

async fn cmd_interactive(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    default_search_limit: usize,
    search_config: SearchConfig,
    similarity_threshold: f32,
    auto_apply_threshold: f32,
    auto_apply: bool,
    max_suggestions: usize,
) -> Result<()> {
    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi.clone())
        .with_runtime_config(librarian_config);
    let search = configured_search_agent(repo.clone(), tei.clone(), &search_config);
    let gardener = GardenerAgent::new(repo.clone())
        .with_threshold(similarity_threshold)
        .with_auto_apply_policy(auto_apply, auto_apply_threshold)
        .with_max_suggestions(max_suggestions);

    println!("GraphRAG Notes - Interactive Mode");
    println!("Commands: add, search, list, garden, stats, help, quit");
    println!();

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    loop {
        print!("graphrag> ");
        stdout.flush()?;

        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            break; // EOF
        }

        let parts: Vec<&str> = line.trim().splitn(2, ' ').collect();
        let cmd = parts.first().map(|s| *s).unwrap_or("");
        let arg = parts.get(1).map(|s| *s).unwrap_or("");

        match cmd {
            "" => continue,

            "add" | "a" => {
                if arg.is_empty() {
                    println!("Usage: add <content>");
                    continue;
                }
                match librarian.ingest_text(arg, None, vec![]).await {
                    Ok(note) => println!(
                        "✓ Added: {}",
                        note.id
                            .as_ref()
                            .map(record_id_to_string)
                            .unwrap_or_else(|| "(no id)".to_string())
                    ),
                    Err(e) => println!("Error: {}", e),
                }
            }

            "search" | "s" => {
                if arg.is_empty() {
                    println!("Usage: search <query>");
                    continue;
                }
                match search.search(arg, default_search_limit).await {
                    Ok(results) => {
                        if results.is_empty() {
                            println!("No results.");
                        } else {
                            for r in results {
                                let preview: String = r.content.chars().take(100).collect();
                                println!(
                                    "• {} - {}{}",
                                    r.title.as_deref().unwrap_or("(untitled)"),
                                    preview,
                                    if r.content.len() > 100 { "..." } else { "" }
                                );
                            }
                        }
                    }
                    Err(e) => println!("Error: {}", e),
                }
            }

            "list" | "l" => match repo.list_notes(10).await {
                Ok(notes) => {
                    if notes.is_empty() {
                        println!("No notes yet.");
                    } else {
                        for note in notes {
                            let preview: String = note.content.chars().take(60).collect();
                            println!(
                                "• {} - {}{}",
                                note.title.as_deref().unwrap_or("(untitled)"),
                                preview,
                                if note.content.len() > 60 { "..." } else { "" }
                            );
                        }
                    }
                }
                Err(e) => println!("Error: {}", e),
            },

            "garden" | "g" => match gardener.run_maintenance().await {
                Ok(report) => {
                    println!(
                        "Maintenance: {} orphans, {} suggestions, {} applied",
                        report.orphans_found,
                        report.suggestions_generated,
                        report.connections_applied,
                    );
                }
                Err(e) => println!("Error: {}", e),
            },

            "stats" => match repo.get_stats().await {
                Ok(s) => println!(
                    "Notes: {}, Entities: {}, Edges: {}",
                    s.note_count, s.entity_count, s.edge_count
                ),
                Err(e) => println!("Error: {}", e),
            },

            "help" | "h" | "?" => {
                println!("Commands:");
                println!("  add <content>    - Add a new note");
                println!("  search <query>   - Search notes");
                println!("  list             - List recent notes");
                println!("  garden           - Run maintenance");
                println!("  stats            - Show statistics");
                println!("  quit             - Exit");
            }

            "quit" | "q" | "exit" => {
                println!("Goodbye!");
                break;
            }

            _ => {
                println!(
                    "Unknown command: {}. Type 'help' for available commands.",
                    cmd
                );
            }
        }

        println!();
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn human_packing_diagnostics_include_all_budget_and_selection_decisions() {
        let diagnostics = AugmentDiagnostics {
            token_count_mode: TokenCountMode::Estimated,
            header_tokens: 12,
            dropped_duplicates: 1,
            dropped_near_duplicates: 2,
            dropped_for_relevance: 3,
            dropped_for_budget: 4,
            dropped_for_entity_filter: 5,
        };
        assert_eq!(
            packing_diagnostics_text(&diagnostics),
            "token_mode=estimated; header_tokens=12; dropped_duplicates=1; dropped_near_duplicates=2; dropped_for_relevance=3; dropped_for_budget=4; dropped_for_entity_filter=5"
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
}
