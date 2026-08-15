//! Command handlers and command-specific renderers.
//!
//! Handlers keep CLI orchestration thin and delegate domain work to core, db,
//! and agent crates. Bootstrap remains in the app module and parser types in cli.

use crate::app::{
    augment_options, configured_search_agent, packing_diagnostics_text,
    should_sample_augment_candidates, zero_budget_augment_context,
};
use crate::cli::{
    to_import_mode, BackupCommand, BackupOutputFormat, Commands, EdgesCommand, GardenCommand,
    GraphModeArg, ImportModeArg, JobOutputFormat, JobsCommand, PortableDataFormat, ProposalCommand,
    SearchScopeArg, SourceOutputFormat, SourcesCommand,
};
use crate::eval::{
    build_baseline_comparison, evaluate_ranked_results_with_tokens, load_baseline, load_eval_cases,
    parse_regression_thresholds, AugmentationDiagnosticsReport, EvalCaseReport, EvalMetadata,
    EvalOutputFormat, EvalRunReport, EvalScope, RankedResult, EVAL_SCHEMA_VERSION,
};
use crate::interactive::cmd_interactive;
use crate::{backup, commands, explain, output};
use anyhow::{Context, Result};
use graphrag_agents::{
    AugmentDiagnostics, ChatIngestOptions, GardenerAgent, GraphEvidence, GraphPathStep,
    LibrarianAgent, LibrarianRuntimeConfig, ProcessingRunResult, ReindexAgent, ReindexScope,
    SearchHitType, SearchScope, SharedEmbedder, SharedEntityExtractor, TokenCountMode,
};
use graphrag_config::{AugmentConfig, SearchConfig};
use graphrag_core::{record_id_to_string, ChatExport, Source};
use graphrag_db::{
    compatibility::EmbeddingIdentity, parse_record_id, repository::RelatedNotes, ProcessingJob,
    ProcessingJobType, Repository, SourceDeleteSummary,
};
use serde::Serialize;
use std::collections::HashMap;
use std::io::{self, BufRead};
use std::path::PathBuf;
use std::sync::{atomic::AtomicBool, Arc};
use std::time::Instant;
use tracing::warn;

/// Dispatch a parsed command after bootstrap has constructed only the
/// resources required by that command.
pub(crate) async fn execute(
    context: crate::app::AppContext,
    command: crate::cli::Commands,
    explain: bool,
) -> Result<()> {
    let crate::app::AppContext {
        repo,
        config,
        tei,
        tgi,
        librarian_config,
        cancellation_requested,
        prepared_notes_edit,
    } = context;

    match command {
        Commands::Export {
            path,
            format: PortableDataFormat::Jsonl,
            output,
        } => {
            print_backup_summary(&backup::export_jsonl(&repo, &path).await?, output)?;
        }
        Commands::ImportData { .. } => {
            unreachable!("import-data returns before database startup")
        }
        Commands::Backup {
            command:
                BackupCommand::Create {
                    path,
                    include_embeddings,
                    format,
                },
        } => {
            print_backup_summary(
                &backup::create_backup(&repo, &path, include_embeddings).await?,
                format,
            )?;
        }
        Commands::Backup { .. } => {
            unreachable!("verify and restore return before database startup")
        }
        Commands::Add {
            content,
            title,
            tags,
        } => {
            cmd_add(repo, tei, tgi, librarian_config, content, title, tags).await?;
        }
        Commands::Notes { command } => {
            let librarian =
                LibrarianAgent::new(repo.clone(), tei, tgi).with_runtime_config(librarian_config);
            commands::notes::run(repo, librarian, command, prepared_notes_edit).await?;
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
            graph,
            format,
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
                graph,
                explain,
                format,
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
            graph,
            format,
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
                graph,
                config.search.clone(),
                config.augment.clone(),
                explain,
                format,
            )
            .await?;
        }
        Commands::List { limit } => {
            eprintln!("Deprecated: use `graphrag notes list` for stable --format output.");
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
        Commands::Jobs { command } => {
            cmd_jobs(
                repo,
                tei,
                tgi,
                librarian_config,
                cancellation_requested
                    .clone()
                    .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
                command,
            )
            .await?;
        }
        Commands::Reindex {
            notes,
            messages,
            summaries,
            all,
            dry_run,
            resume,
            format,
        } => {
            cmd_reindex(
                repo,
                tei,
                cancellation_requested
                    .clone()
                    .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
                notes,
                messages,
                summaries,
                all,
                dry_run,
                resume,
                format,
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
                cancellation_requested
                    .clone()
                    .unwrap_or_else(|| Arc::new(AtomicBool::new(false))),
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
            eprintln!(
                "Deprecated: use `graphrag notes show {note_id}` for stable --format output."
            );
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

pub(crate) fn print_backup_summary(
    summary: &backup::BackupSummary,
    format: BackupOutputFormat,
) -> Result<()> {
    match format {
        BackupOutputFormat::Human => {
            let operation = if summary.dry_run {
                "Dry-run restore"
            } else {
                "Backup"
            };
            println!("{operation} verified: {}", summary.path.display());
            println!("Schema version: {}", summary.schema_version);
            println!("Records: {}", summary.records);
            println!("Embeddings included: {}", summary.includes_embeddings);
            for (table, count) in &summary.record_counts {
                println!("  {table}: {count}");
            }
        }
        BackupOutputFormat::Json => println!("{}", serde_json::to_string(summary)?),
    }
    Ok(())
}

#[derive(Serialize)]
struct JobOutput {
    id: String,
    job_type: String,
    source_generation: Option<String>,
    scope: Option<String>,
    item_count: usize,
    status: String,
    total_count: i64,
    completed_count: i64,
    failed_count: i64,
    checkpoint: Option<String>,
    last_error: Option<String>,
    created_at: String,
    updated_at: String,
    finished_at: Option<String>,
}

impl From<&graphrag_db::ProcessingJob> for JobOutput {
    fn from(job: &graphrag_db::ProcessingJob) -> Self {
        Self {
            id: job.id.as_ref().map(record_id_to_string).unwrap_or_default(),
            job_type: job.job_type.clone(),
            source_generation: job.source_generation.clone(),
            scope: job.scope.clone(),
            item_count: job.item_ids.len(),
            status: job.status.clone(),
            total_count: job.total_count,
            completed_count: job.completed_count,
            failed_count: job.failed_count,
            checkpoint: job.checkpoint.clone(),
            last_error: job.last_error.clone(),
            created_at: job.created_at.to_rfc3339(),
            updated_at: job.updated_at.to_rfc3339(),
            finished_at: job.finished_at.map(|value| value.to_rfc3339()),
        }
    }
}

pub(crate) fn print_job(job: &graphrag_db::ProcessingJob, format: JobOutputFormat) -> Result<()> {
    match format {
        JobOutputFormat::Json => println!("{}", serde_json::to_string(&JobOutput::from(job))?),
        JobOutputFormat::Human => println!(
            "{} {} status={} scope={} items={} completed={}/{} failed={} checkpoint={}{}",
            job.id.as_ref().map(record_id_to_string).unwrap_or_default(),
            job.job_type,
            job.status,
            job.scope.as_deref().unwrap_or("-"),
            job.item_ids.len(),
            job.completed_count,
            job.total_count,
            job.failed_count,
            job.checkpoint.as_deref().unwrap_or("-"),
            job.last_error
                .as_deref()
                .map(|error| format!(" error={error}"))
                .unwrap_or_default(),
        ),
    }
    Ok(())
}

pub(crate) async fn cmd_jobs(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    cancellation_requested: Arc<AtomicBool>,
    command: JobsCommand,
) -> Result<()> {
    match command {
        JobsCommand::List { limit, format } => {
            let jobs = repo.list_processing_jobs(limit).await?;
            if format == JobOutputFormat::Json {
                let rows = jobs.iter().map(JobOutput::from).collect::<Vec<_>>();
                println!("{}", serde_json::to_string(&rows)?);
            } else if jobs.is_empty() {
                println!("No processing jobs.");
            } else {
                for job in &jobs {
                    print_job(job, format)?;
                }
            }
        }
        JobsCommand::Show { id, format } => {
            let job = repo
                .get_processing_job(&id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Processing job not found: {id}"))?;
            print_job(&job, format)?;
        }
        JobsCommand::Cancel { id } => {
            let record = parse_record_id(&id, Some("processing_job"))?;
            let job = repo.cancel_processing_job(&record).await?;
            print_job(&job, JobOutputFormat::Human)?;
        }
        JobsCommand::Resume { id } => {
            let job = repo
                .get_processing_job(&id)
                .await?
                .ok_or_else(|| anyhow::anyhow!("Processing job not found: {id}"))?;
            ensure_resume_provider_health(&job, &tei, &tgi).await?;
            let librarian = LibrarianAgent::new(repo, tei, tgi)
                .with_runtime_config(librarian_config)
                .with_cancellation_flag(cancellation_requested);
            let result = librarian.resume_processing_job(&id).await?;
            println!(
                "job={} completed={} failed={} cancelled={}",
                result.job_id, result.completed, result.failed, result.cancelled
            );
        }
    }
    Ok(())
}

#[derive(Serialize)]
pub(crate) struct ReindexOutput {
    pub(crate) dry_run: bool,
    pub(crate) scope: String,
    pub(crate) item_count: usize,
    pub(crate) estimated_input_characters: u64,
    pub(crate) provider: String,
    pub(crate) model: String,
    pub(crate) dimension: usize,
    pub(crate) job_id: Option<String>,
    pub(crate) completed: Option<u64>,
    pub(crate) cancelled: bool,
}

pub(crate) fn print_reindex_output(output: &ReindexOutput, format: JobOutputFormat) -> Result<()> {
    match format {
        JobOutputFormat::Json => println!("{}", serde_json::to_string(output)?),
        JobOutputFormat::Human => {
            if output.dry_run {
                println!(
                    "Reindex dry run: scope={} items={} estimated_input_characters={}",
                    output.scope, output.item_count, output.estimated_input_characters
                );
            } else {
                println!(
                    "Reindex job {}: completed={}/{}{}",
                    output.job_id.as_deref().unwrap_or("-"),
                    output.completed.unwrap_or(0),
                    output.item_count,
                    if output.cancelled { " (cancelled)" } else { "" }
                );
            }
            println!(
                "Embedding target: {}/{} ({} dimensions)",
                output.provider, output.model, output.dimension
            );
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_reindex(
    repo: Repository,
    tei: SharedEmbedder,
    cancellation_requested: Arc<AtomicBool>,
    notes: bool,
    messages: bool,
    summaries: bool,
    all: bool,
    dry_run: bool,
    resume: Option<String>,
    format: JobOutputFormat,
) -> Result<()> {
    if resume.is_some() && (notes || messages || summaries || all || dry_run) {
        anyhow::bail!("--resume cannot be combined with scope selectors or --dry-run");
    }
    let capabilities = tei.capabilities();
    // A real probe is deliberately performed before creating/resuming a job:
    // provider configuration alone is not a proof of the indexed dimension.
    let probe = tei.embed("graphrag reindex dimension probe", false).await?;
    if probe.len() != graphrag_db::schema::EMBEDDING_DIMENSION {
        anyhow::bail!(
            "active embedding provider {}/{} returned {} dimensions; this database schema indexes {}. Choose a compatible model before reindexing",
            capabilities.provider,
            capabilities.model,
            probe.len(),
            graphrag_db::schema::EMBEDDING_DIMENSION,
        );
    }
    let identity = EmbeddingIdentity::new(
        capabilities.provider.clone(),
        capabilities.model.clone(),
        probe.len(),
    );
    let agent = ReindexAgent::new(repo.clone(), tei).with_cancellation_flag(cancellation_requested);
    if let Some(job_id) = resume {
        let result = agent.resume(&job_id, identity).await?;
        let job = repo
            .get_processing_job(&result.job_id)
            .await?
            .ok_or_else(|| anyhow::anyhow!("reindex job disappeared: {}", result.job_id))?;
        print_reindex_output(
            &ReindexOutput {
                dry_run: false,
                scope: job.scope.unwrap_or_else(|| "reindex".into()),
                item_count: job.item_ids.len(),
                estimated_input_characters: 0,
                provider: capabilities.provider,
                model: capabilities.model,
                dimension: probe.len(),
                job_id: Some(result.job_id),
                completed: Some(result.completed),
                cancelled: result.cancelled,
            },
            format,
        )?;
        return Ok(());
    }
    let scope = if all {
        ReindexScope::all()
    } else {
        ReindexScope {
            notes,
            messages,
            summaries,
        }
    };
    let preview = agent.preview(scope).await?;
    if dry_run {
        print_reindex_output(
            &ReindexOutput {
                dry_run: true,
                scope: preview.scope.label(),
                item_count: preview.item_ids.len(),
                estimated_input_characters: preview.estimated_input_characters,
                provider: capabilities.provider,
                model: capabilities.model,
                dimension: probe.len(),
                job_id: None,
                completed: None,
                cancelled: false,
            },
            format,
        )?;
        return Ok(());
    }
    let item_count = preview.item_ids.len();
    let estimated_input_characters = preview.estimated_input_characters;
    let scope = preview.scope.label();
    let result = agent.start(preview, identity).await?;
    print_reindex_output(
        &ReindexOutput {
            dry_run: false,
            scope,
            item_count,
            estimated_input_characters,
            provider: capabilities.provider,
            model: capabilities.model,
            dimension: probe.len(),
            job_id: Some(result.job_id),
            completed: Some(result.completed),
            cancelled: result.cancelled,
        },
        format,
    )
}

/// Validate only the provider required by the persisted job before changing
/// its state to running. Embedding jobs do not need extraction service health,
/// and entity jobs do not need embeddings service health.
pub(crate) async fn ensure_resume_provider_health(
    job: &ProcessingJob,
    tei: &SharedEmbedder,
    tgi: &SharedEntityExtractor,
) -> Result<()> {
    match job.job_type_enum() {
        Some(ProcessingJobType::Embedding) => {
            if !tei.health().await.unwrap_or(false) {
                eprintln!("Error: embeddings service is not reachable.");
                eprintln!("  TEI (embeddings): {}", tei.capabilities().endpoint);
                anyhow::bail!("Embeddings service unavailable")
            }
        }
        Some(ProcessingJobType::EntityExtraction) => {
            if !tgi.health().await.unwrap_or(false) {
                eprintln!("Error: extraction service is not reachable.");
                eprintln!("  TGI (extraction): {}", tgi.capabilities().endpoint);
                anyhow::bail!("Extraction service unavailable")
            }
        }
        Some(ProcessingJobType::Reindex) => anyhow::bail!(
            "reindex jobs must be resumed with `graphrag reindex --resume {}`",
            job.id.as_ref().map(record_id_to_string).unwrap_or_default()
        ),
        None => anyhow::bail!("Unsupported processing job type: {}", job.job_type),
    }
    Ok(())
}

pub(crate) async fn cmd_add(
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
            let lines: Vec<String> = stdin.lock().lines().map_while(Result::ok).collect();
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

pub(crate) async fn cmd_import(
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
pub(crate) fn import_path_utf8(path: &std::path::Path) -> Result<&str> {
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

pub(crate) fn print_import_summary(
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

pub(crate) async fn cmd_sources(
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

pub(crate) fn delete_is_dry_run(dry_run: bool, yes: bool) -> bool {
    dry_run || !yes
}

pub(crate) fn print_delete_summary(
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

pub(crate) async fn cmd_import_chats(
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
#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_migrate_chats(
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

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_extract_entities(
    repo: Repository,
    tei: SharedEmbedder,
    tgi: SharedEntityExtractor,
    librarian_config: LibrarianRuntimeConfig,
    cancellation_requested: Arc<AtomicBool>,
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
    let librarian = LibrarianAgent::new(repo, tei, tgi)
        .with_runtime_config(librarian_config)
        .with_cancellation_flag(cancellation_requested);
    let result = if !note_ids.is_empty() {
        librarian
            .extract_entities_for_note_ids_result(&note_ids, force)
            .await?
    } else if all {
        librarian
            .extract_entities_for_all_notes_result(limit, force)
            .await?
    } else {
        librarian.extract_entities_for_notes_result(limit).await?
    };
    report_entity_extraction_result(&result)
}

/// Render entity extraction only as a success after its durable run reached a
/// terminal non-cancelled state. A Ctrl-C is resumable work, not a successful
/// zero-item invocation.
pub(crate) fn report_entity_extraction_result(result: &ProcessingRunResult) -> Result<()> {
    if result.cancelled {
        if result.job_id.is_empty() {
            eprintln!(
                "Entity extraction cancelled before durable work was created; no job needs resuming."
            );
            anyhow::bail!("Entity extraction cancelled")
        }
        eprintln!(
            "Entity extraction cancelled: job={} completed={} failed={}. Resume with `graphrag jobs resume {}`.",
            result.job_id, result.completed, result.failed, result.job_id
        );
        anyhow::bail!("Entity extraction cancelled")
    }
    println!("✓ Extracted entities for {} notes", result.completed);
    Ok(())
}

pub(crate) async fn cmd_show_entities(repo: Repository, note_id: String) -> Result<()> {
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

pub(crate) async fn cmd_show_note(repo: Repository, note_id: String) -> Result<()> {
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

pub(crate) async fn cmd_list_edges(repo: Repository, limit: usize) -> Result<()> {
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

pub(crate) async fn cmd_show_note_edges(repo: Repository, note_id: String) -> Result<()> {
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

#[derive(Serialize)]
pub(crate) struct SearchMachineResult {
    pub(crate) id: String,
    pub(crate) hit_type: &'static str,
    pub(crate) title: Option<String>,
    pub(crate) content: String,
    pub(crate) created_at: Option<String>,
    pub(crate) source_uri: Option<String>,
    pub(crate) score: f32,
    pub(crate) conversation_uuid: Option<String>,
    pub(crate) message_index: Option<i64>,
    pub(crate) role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) related: Option<RelatedNotes>,
}

#[derive(Serialize)]
struct SearchMachineOutput {
    results: Vec<SearchMachineResult>,
}

impl SearchMachineResult {
    pub(crate) fn from_context(result: &graphrag_agents::search::EnrichedSearchResult) -> Self {
        let note = &result.result;
        Self {
            id: record_id_to_string(&note.id),
            hit_type: "note",
            title: note.title.clone(),
            content: note.content.clone(),
            created_at: Some(note.created_at.to_rfc3339()),
            source_uri: note.source_uri.clone(),
            score: result.final_score(),
            conversation_uuid: None,
            message_index: None,
            role: None,
            related: result.related.clone(),
        }
    }

    pub(crate) fn from_scoped(
        result: &graphrag_agents::search::ScopedSearchResult,
        related: Option<RelatedNotes>,
    ) -> Self {
        Self {
            id: result.id.clone(),
            hit_type: search_hit_type_name(result.hit_type),
            title: result.title.clone(),
            content: result.content.clone(),
            created_at: result.created_at.map(|value| value.to_rfc3339()),
            source_uri: result.source_uri.clone(),
            score: result.score,
            conversation_uuid: result.conversation_uuid.clone(),
            message_index: result.message_index,
            role: result.role.clone(),
            related,
        }
    }
}

#[derive(Serialize)]
pub(crate) struct AugmentMachineChunk {
    pub(crate) citation: usize,
    pub(crate) hit_type: &'static str,
    pub(crate) id: String,
    pub(crate) title: Option<String>,
    pub(crate) snippet: String,
    pub(crate) created_at: Option<String>,
    pub(crate) source_uri: Option<String>,
    pub(crate) score: f32,
    pub(crate) conversation_uuid: Option<String>,
    pub(crate) message_index: Option<i64>,
    pub(crate) role: Option<String>,
    pub(crate) approx_tokens: usize,
    pub(crate) rendered_tokens: usize,
    pub(crate) truncated: bool,
    pub(crate) selected_span_start: Option<usize>,
    pub(crate) selected_span_end: Option<usize>,
}

#[derive(Serialize)]
pub(crate) struct AugmentMachineOutput {
    query: String,
    scope: String,
    entity_filter: Option<String>,
    prompt: String,
    chunks: Vec<AugmentMachineChunk>,
    total_tokens: usize,
    diagnostics: AugmentDiagnostics,
}

pub(crate) fn search_hit_type_name(hit_type: SearchHitType) -> &'static str {
    match hit_type {
        SearchHitType::Note => "note",
        SearchHitType::Message => "message",
        SearchHitType::ConversationSummary => "conversation-summary",
    }
}

pub(crate) fn augment_machine_output(
    ctx: &graphrag_agents::AugmentContext,
) -> AugmentMachineOutput {
    AugmentMachineOutput {
        query: ctx.query.clone(),
        scope: format!("{:?}", ctx.scope),
        entity_filter: ctx.entity_filter.clone(),
        prompt: ctx.render_prompt_block(),
        chunks: ctx
            .chunks
            .iter()
            .map(|chunk| AugmentMachineChunk {
                citation: chunk.citation,
                hit_type: search_hit_type_name(chunk.hit_type),
                id: chunk.id.clone(),
                title: chunk.title.clone(),
                snippet: chunk.snippet.clone(),
                created_at: chunk.created_at.map(|value| value.to_rfc3339()),
                source_uri: chunk.source_uri.clone(),
                score: chunk.score,
                conversation_uuid: chunk.conversation_uuid.clone(),
                message_index: chunk.message_index,
                role: chunk.role.clone(),
                approx_tokens: chunk.approx_tokens,
                rendered_tokens: chunk.rendered_tokens,
                truncated: chunk.truncated,
                selected_span_start: chunk.selected_span_start,
                selected_span_end: chunk.selected_span_end,
            })
            .collect(),
        total_tokens: ctx.total_tokens,
        diagnostics: ctx.diagnostics.clone(),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_search(
    repo: Repository,
    tei: SharedEmbedder,
    query: String,
    limit: usize,
    scope: SearchScopeArg,
    since_days: Option<u32>,
    source_uri: Option<String>,
    context: bool,
    graph: GraphModeArg,
    explain: bool,
    format: output::OutputFormat,
    search_config: SearchConfig,
) -> Result<()> {
    let search = configured_search_agent(repo.clone(), tei, &search_config);
    let embedding_identity = search.embedding_identity();
    let filters = serde_json::json!({
        "scope": format!("{scope:?}"),
        "since_days": since_days,
        "source_uri": source_uri,
        "context": context,
        "graph": format!("{graph:?}"),
    });
    let scope = match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    };

    if context && scope == SearchScope::Notes && graph == GraphModeArg::Off {
        let results = search
            .search_with_context_filtered(&query, limit, since_days, source_uri.clone())
            .await?;
        let related_by_note = results
            .iter()
            .filter_map(|result| {
                result
                    .related
                    .clone()
                    .map(|related| (record_id_to_string(&result.result.id), related))
            })
            .collect::<HashMap<_, _>>();

        if format != output::OutputFormat::Human {
            if !explain {
                let output = results
                    .iter()
                    .map(SearchMachineResult::from_context)
                    .collect::<Vec<_>>();
                return match format {
                    output::OutputFormat::Json => output::print(
                        format,
                        "search",
                        SearchMachineOutput { results: output },
                        |_| Ok(()),
                    ),
                    output::OutputFormat::Jsonl => output::print_jsonl("search", output),
                    output::OutputFormat::Human => unreachable!("handled above"),
                };
            }
            let explanations = results
                .iter()
                .map(|result| {
                    result
                        .explanation()
                        .with_embedding_identity(&embedding_identity.0, &embedding_identity.1)
                })
                .collect::<Vec<_>>();
            return match format {
                output::OutputFormat::Json => output::print(
                    format,
                    "search",
                    explain::search_json(
                        &explanations,
                        &graphrag_agents::GraphRetrievalSummary::default(),
                        filters,
                        &related_by_note,
                    ),
                    |_| Ok(()),
                ),
                output::OutputFormat::Jsonl => output::print_jsonl_with_pipeline(
                    "search",
                    explanations.iter(),
                    explain::search_pipeline(
                        &graphrag_agents::GraphRetrievalSummary::default(),
                        filters,
                        &related_by_note,
                    ),
                ),
                output::OutputFormat::Human => unreachable!("handled above"),
            };
        }

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

            if explain {
                println!("   Explain: {}", explain::human(&result.explanation()));
            }

            println!();
        }
    } else {
        if context && scope != SearchScope::Notes {
            eprintln!("Context is only available for notes scope; continuing without context.");
        }

        let results = search
            .search_with_scope_graph(
                &query,
                limit,
                scope,
                since_days,
                source_uri.clone(),
                graph.into(),
            )
            .await?;
        // `--context` is part of the command result, not merely human
        // decoration. Resolve it before either machine renderer returns so
        // JSON and JSONL retain the same related-note data as terminal output.
        let related_by_note = if context && scope == SearchScope::Notes {
            let context_repo = repo.clone();
            best_effort_related_notes(&results.hits, move |id| {
                let repo = context_repo.clone();
                async move {
                    let note_id = parse_record_id(&id, Some("note"))?;
                    Ok::<_, anyhow::Error>(repo.get_related_notes(&note_id).await?)
                }
            })
            .await?
        } else {
            HashMap::new()
        };

        if format != output::OutputFormat::Human {
            if !explain {
                let output = results
                    .hits
                    .iter()
                    .map(|result| {
                        SearchMachineResult::from_scoped(
                            result,
                            related_by_note.get(&result.id).cloned(),
                        )
                    })
                    .collect::<Vec<_>>();
                return match format {
                    output::OutputFormat::Json => output::print(
                        format,
                        "search",
                        SearchMachineOutput { results: output },
                        |_| Ok(()),
                    ),
                    output::OutputFormat::Jsonl => output::print_jsonl("search", output),
                    output::OutputFormat::Human => unreachable!("handled above"),
                };
            }
            let explanations = results
                .hits
                .iter()
                .map(|result| {
                    result
                        .explanation()
                        .with_embedding_identity(&embedding_identity.0, &embedding_identity.1)
                })
                .collect::<Vec<_>>();
            return match format {
                output::OutputFormat::Json => output::print(
                    format,
                    "search",
                    explain::search_json(
                        &explanations,
                        &results.summary,
                        filters,
                        &related_by_note,
                    ),
                    |_| Ok(()),
                ),
                output::OutputFormat::Jsonl => output::print_jsonl_with_pipeline(
                    "search",
                    explanations.iter(),
                    explain::search_pipeline(&results.summary, filters, &related_by_note),
                ),
                output::OutputFormat::Human => unreachable!("handled above"),
            };
        }

        if results.hits.is_empty() {
            println!("No results found.");
            return Ok(());
        }

        println!("Found {} results:\n", results.hits.len());
        println!(
            "Graph: entities={} considered={} selected={} dropped={}\n",
            results.summary.entities_matched,
            results.summary.candidates_considered,
            results.summary.candidates_selected,
            results.summary.candidates_dropped,
        );

        // `--context` predates graph retrieval and remains an independent
        // accepted-edge summary. Keep it for the default `--graph=auto`
        // path as well as explicit modes; graph evidence is additive, not a
        // replacement for get_related_notes output.
        for (i, r) in results.hits.iter().enumerate() {
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
            if let Some(graph) = r.graph.as_ref() {
                println!("   Graph path: {}", render_search_graph_evidence(graph),);
            }
            if let Some(related) = related_by_note.get(&r.id) {
                let total =
                    related.supporting.len() + related.contradicting.len() + related.related.len();
                if total > 0 {
                    println!("   → {} related notes", total);
                }
            }
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
            if explain {
                println!("   Explain: {}", explain::human(&r.explanation()));
            }
            println!();
        }
    }

    Ok(())
}

pub(crate) async fn best_effort_related_notes<F, Fut, E>(
    hits: &[graphrag_agents::search::ScopedSearchResult],
    mut lookup: F,
) -> Result<HashMap<String, RelatedNotes>>
where
    F: FnMut(String) -> Fut,
    Fut: std::future::Future<Output = std::result::Result<RelatedNotes, E>>,
    E: std::fmt::Display,
{
    let mut related = HashMap::new();
    for hit in hits
        .iter()
        .filter(|hit| hit.hit_type == SearchHitType::Note)
    {
        // A malformed primary search ID remains a command error. Only the
        // additive lookup below is intentionally non-fatal.
        parse_record_id(&hit.id, Some("note"))?;
        retain_related_note_lookup(&mut related, hit.id.clone(), lookup(hit.id.clone()).await);
    }
    Ok(related)
}

pub(crate) fn retain_related_note_lookup<E: std::fmt::Display>(
    related: &mut HashMap<String, RelatedNotes>,
    id: String,
    result: std::result::Result<RelatedNotes, E>,
) {
    match result {
        Ok(notes) => {
            related.insert(id, notes);
        }
        Err(error) => {
            // Context is additive. Preserve already-retrieved primary
            // results and successful note summaries if one enrichment query
            // fails, while keeping the failure observable.
            warn!(note_id = %id, error = %error, "unable to load related-note context");
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_augment(
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
    graph: GraphModeArg,
    search_config: SearchConfig,
    augment_config: AugmentConfig,
    explain: bool,
    format: output::OutputFormat,
) -> Result<()> {
    if entity.is_some() && scope != SearchScopeArg::Notes {
        anyhow::bail!("--entity currently requires --scope notes");
    }

    let filters = serde_json::json!({
        "scope": format!("{scope:?}"),
        "since_days": since_days,
        "source_uri": source_uri,
        "entity": entity,
        "graph": format!("{graph:?}"),
    });
    let scope = match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    };

    let options = augment_options(limit, max_tokens, max_chunk_tokens, &augment_config);
    let (ctx, embedding_identity) = if should_sample_augment_candidates(explain, &options) {
        let search = configured_search_agent(repo, tei, &search_config);
        let embedding_identity = search.embedding_identity();
        let ctx = if explain {
            search
                .build_augmented_context_with_graph_explain(
                    &query,
                    scope,
                    since_days,
                    source_uri.clone(),
                    entity.clone(),
                    options.clone(),
                    graph.into(),
                )
                .await?
        } else {
            search
                .build_augmented_context_with_graph(
                    &query,
                    scope,
                    since_days,
                    source_uri.clone(),
                    entity.clone(),
                    options.clone(),
                    graph.into(),
                )
                .await?
        };
        (ctx, Some(embedding_identity))
    } else {
        (
            zero_budget_augment_context(query.clone(), scope, entity.clone()),
            None,
        )
    };

    if format != output::OutputFormat::Human {
        if !explain {
            let output = augment_machine_output(&ctx);
            return match format {
                output::OutputFormat::Json => output::print(format, "augment", output, |_| Ok(())),
                output::OutputFormat::Jsonl => output::print_jsonl("augment", output.chunks),
                output::OutputFormat::Human => unreachable!("handled above"),
            };
        }
        let embedding_identity = embedding_identity
            .as_ref()
            .expect("explain mode always samples augmentation candidates");
        let mut explanations = ctx
            .chunks
            .iter()
            .enumerate()
            .map(|(index, chunk)| {
                chunk
                    .explanation(index + 1)
                    .with_embedding_identity(&embedding_identity.0, &embedding_identity.1)
            })
            .collect::<Vec<_>>();
        explanations.extend(ctx.exclusions.iter().cloned().map(|explanation| {
            explanation.with_embedding_identity(&embedding_identity.0, &embedding_identity.1)
        }));
        return match format {
            output::OutputFormat::Json => output::print(
                format,
                "augment",
                explain::augmentation_json(
                    &explanations,
                    &ctx.diagnostics,
                    ctx.total_tokens,
                    filters,
                    &options,
                ),
                |_| Ok(()),
            ),
            output::OutputFormat::Jsonl => output::print_jsonl_with_pipeline(
                "augment",
                explanations.iter(),
                explain::augmentation_pipeline(
                    &ctx.diagnostics,
                    ctx.total_tokens,
                    filters,
                    &options,
                ),
            ),
            output::OutputFormat::Human => unreachable!("handled above"),
        };
    }

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

    if explain && !ctx.exclusions.is_empty() {
        println!("\nExcluded candidates:");
        for exclusion in &ctx.exclusions {
            println!("  • {}", explain::human(exclusion));
        }
    }

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
        if let Some(graph) = chunk.graph.as_ref() {
            provenance.push_str(&render_augment_graph_evidence(graph));
        }
        println!(
            "  [C{}] {} | score={:.3} | tokens={} | {}",
            chunk.citation, hit_kind, chunk.score, chunk.approx_tokens, provenance
        );
        if explain {
            println!(
                "      Explain: {}",
                explain::human(&chunk.explanation(chunk.citation))
            );
        }
    }

    Ok(())
}

/// Keep terminal graph evidence reconstructable even when multiple edges share
/// a type or a path has more than one hop. `GraphPathStep` is also serialized
/// in structured agent output; this compact rendering retains the same record
/// identities for the human CLI surfaces.
pub(crate) fn render_graph_path(path: &[GraphPathStep], separator: &str) -> String {
    path.iter()
        .map(|step| {
            format!(
                "{}:{} edge={} endpoints={}→{} confidence={:.2}",
                step.direction,
                step.edge_type,
                step.edge_id,
                step.from_id,
                step.to_id,
                step.confidence
            )
        })
        .collect::<Vec<_>>()
        .join(separator)
}

pub(crate) fn render_graph_citation(graph: &GraphEvidence, path_separator: &str) -> String {
    format!(
        "graph_seed={}, graph_hops={}, graph_decay={:.2}, graph_source_uri={}, graph_path={}, graph_provenance={}",
        graph.seed_note_id,
        graph.hops,
        graph.decay,
        graph.source_uri.as_deref().unwrap_or_default(),
        render_graph_path(&graph.path, path_separator),
        graph.provenance_ids.join(" | "),
    )
}

pub(crate) fn render_search_graph_evidence(graph: &GraphEvidence) -> String {
    format!(
        "entities=[{}] {}",
        graph.query_entities.join(", "),
        render_graph_citation(graph, " -> "),
    )
}

pub(crate) fn render_augment_graph_evidence(graph: &GraphEvidence) -> String {
    format!(", {}", render_graph_citation(graph, " | "))
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn cmd_eval_augment(
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

pub(crate) fn eval_scope_to_search_scope(scope: EvalScope) -> SearchScope {
    match scope {
        EvalScope::Notes => SearchScope::Notes,
        EvalScope::Messages => SearchScope::Messages,
        EvalScope::All => SearchScope::All,
    }
}

pub(crate) fn search_scope_arg_to_scope(scope: SearchScopeArg) -> SearchScope {
    match scope {
        SearchScopeArg::Notes => SearchScope::Notes,
        SearchScopeArg::Messages => SearchScope::Messages,
        SearchScopeArg::All => SearchScope::All,
    }
}

pub(crate) async fn cmd_list(repo: Repository, limit: usize) -> Result<()> {
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

pub(crate) async fn cmd_garden(
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

pub(crate) async fn cmd_proposals(repo: Repository, command: ProposalCommand) -> Result<()> {
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

pub(crate) async fn cmd_edges(repo: Repository, command: EdgesCommand) -> Result<()> {
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

pub(crate) fn edge_dry_run_message(id: &str, exists: bool) -> String {
    if exists {
        format!("Dry run: {id} exists and would be deleted; no changes made.")
    } else {
        format!("Dry run: {id} is absent; no changes made.")
    }
}

pub(crate) async fn cmd_stats(repo: Repository) -> Result<()> {
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
