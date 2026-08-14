//! GraphRAG Notes CLI
//!
//! A command-line interface for the GraphRAG Notes system.

mod doctor;
mod eval;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use eval::{
    build_baseline_comparison, evaluate_ranked_results, load_baseline, load_eval_cases,
    parse_regression_thresholds, EvalCaseReport, EvalMetadata, EvalOutputFormat, EvalRunReport,
    EvalScope, RankedResult, EVAL_SCHEMA_VERSION,
};
use graphrag_agents::{
    AugmentOptions, ChatImportMode, ChatIngestOptions, GardenerAgent, InferenceProviderConfig,
    InferenceProviders, LibrarianAgent, LibrarianRuntimeConfig, SearchAgent, SearchHitType,
    SearchScope, SharedEmbedder, SharedEntityExtractor,
};
use graphrag_config::{CliOverrides, RuntimeConfig, SearchConfig};
use graphrag_core::{record_id_to_string, ChatExport, Source};
use graphrag_db::{
    fusion::{FusionConfig, FusionStrategy},
    init_memory, init_persistent, migrations, Repository, SourceDeleteSummary,
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

    /// Run the gardener (maintenance)
    Garden {
        /// Only show suggestions, don't apply
        #[arg(long)]
        dry_run: bool,
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
            )
            .await?;
        }
        Commands::List { limit } => {
            cmd_list(repo, limit).await?;
        }
        Commands::Garden { dry_run } => {
            cmd_garden(
                repo,
                dry_run,
                config.gardener.similarity_threshold,
                config.gardener.auto_apply_threshold,
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
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read file: {}", path.display()))?;

    let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
    let summary = librarian
        .ingest_markdown_with_options(path.to_str().unwrap_or("unknown"), content, force)
        .await?;

    print_import_summary(&summary, SourceOutputFormat::Human)?;

    Ok(())
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
            "Source {} generation {}: {} (created {}, unchanged {}, updated {}, deleted {}, failed {})",
            output.source_uri,
            output.generation,
            output.action,
            output.created,
            output.unchanged,
            output.updated,
            output.deleted,
            output.failed,
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
                            source.id.as_ref().map(record_id_to_string).unwrap_or_default(),
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
            if !dry_run && !yes {
                anyhow::bail!("refusing to delete without --yes; use --dry-run to inspect the cascade");
            }
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
            let librarian = LibrarianAgent::new(repo, tei, tgi).with_runtime_config(librarian_config);
            let summary = librarian.reimport_markdown_source(&id_or_uri, force).await?;
            print_import_summary(&summary, format)?;
        }
    }
    Ok(())
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
        source_id: source.id.as_ref().map(record_id_to_string).unwrap_or_default(),
        dry_run,
        summary,
    };
    match format {
        SourceOutputFormat::Json => println!("{}", serde_json::to_string(&output)?),
        SourceOutputFormat::Human => println!(
            "{} source {}: notes={}, mentions={}, note_edges={}, conversation_provenance={}, message_provenance={}",
            if dry_run { "Would delete" } else { "Deleted" },
            output.source_id,
            summary.notes,
            summary.mentions,
            summary.note_edges,
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
            AugmentOptions {
                max_chunks: limit,
                max_total_tokens: max_tokens,
                max_chunk_tokens,
            },
        )
        .await?;

    if ctx.chunks.is_empty() {
        println!("No augmentation context found.");
        return Ok(());
    }

    println!("Augmentation context:");
    println!("  • Query: {}", ctx.query);
    println!("  • Scope: {:?}", ctx.scope);
    if let Some(filter) = ctx.entity_filter.as_deref() {
        println!("  • Entity filter: {}", filter);
    }
    println!("  • Chunks selected: {}", ctx.chunks.len());
    println!("  • Approx tokens used: {}", ctx.total_tokens);
    println!(
        "  • Dropped (duplicates/budget/entity-filter): {}/{}/{}",
        ctx.dropped_duplicates, ctx.dropped_for_budget, ctx.dropped_for_entity_filter
    );

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
                AugmentOptions {
                    max_chunks: limit,
                    max_total_tokens: max_tokens,
                    max_chunk_tokens,
                },
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
        let metrics = evaluate_ranked_results(case, &ranked, k, latency_ms);

        if matches!(format, EvalOutputFormat::Human) {
            let status = match metrics.checks_passed {
                Some(true) => "PASS",
                Some(false) => "MISS",
                None => "UNSCORED",
            };
            println!(
                "{}. {} [{}] k={} chunks={} tokens={} latency={}ms",
                idx + 1,
                case.display_name(),
                status,
                metrics.k,
                metrics.chunks,
                metrics.tokens,
                metrics.latency_ms,
            );
        }
        reports.push(EvalCaseReport {
            name: case.display_name().to_string(),
            query: case.query.clone(),
            metrics,
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
    dry_run: bool,
    similarity_threshold: f32,
    auto_apply_threshold: f32,
    max_suggestions: usize,
) -> Result<()> {
    let gardener = GardenerAgent::new(repo)
        .with_threshold(similarity_threshold)
        .with_auto_apply_threshold(auto_apply_threshold)
        .with_max_suggestions(max_suggestions);

    if dry_run {
        println!("Finding orphan notes...\n");

        let orphans = gardener.find_orphans().await?;

        if orphans.is_empty() {
            println!("No orphan notes found. Your knowledge graph is well connected!");
            return Ok(());
        }

        println!("Found {} orphan notes:", orphans.len());
        for orphan in &orphans {
            println!("  • {}", orphan.title.as_deref().unwrap_or("(untitled)"));
        }

        println!("\nGenerating suggestions...\n");

        let suggestions = gardener.suggest_connections().await?;

        if suggestions.is_empty() {
            println!("No connection suggestions found.");
        } else {
            println!("Suggested connections:");
            for s in &suggestions {
                println!(
                    "  {} → {} ({:.0}%: {})",
                    s.from_note.title.as_deref().unwrap_or("(untitled)"),
                    s.to_note.title.as_deref().unwrap_or("(untitled)"),
                    s.similarity * 100.0,
                    s.reason,
                );
            }
        }
    } else {
        println!("Running maintenance...\n");

        let report = gardener.run_maintenance().await?;

        println!("Maintenance complete:");
        println!("  • Orphans found: {}", report.orphans_found);
        println!(
            "  • Suggestions generated: {}",
            report.suggestions_generated
        );
        println!("  • Connections applied: {}", report.connections_applied);
        println!("  • Orphans remaining: {}", report.orphans_remaining);
    }

    Ok(())
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
    max_suggestions: usize,
) -> Result<()> {
    let librarian = LibrarianAgent::new(repo.clone(), tei.clone(), tgi.clone())
        .with_runtime_config(librarian_config);
    let search = configured_search_agent(repo.clone(), tei.clone(), &search_config);
    let gardener = GardenerAgent::new(repo.clone())
        .with_threshold(similarity_threshold)
        .with_auto_apply_threshold(auto_apply_threshold)
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
