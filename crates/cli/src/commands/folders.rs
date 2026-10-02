//! Explicit folder registration, sync reports, and confirmed missing-source pruning.
use crate::cli::FolderCommand;
use crate::output::{self, OutputEnvelope, OutputFormat};
use anyhow::{Context, Result};
use graphrag_agents::{
    ingestion::folders::{self, FileSyncResult, FileSyncStatus, FolderSpec, SyncReport},
    LibrarianAgent,
};
use graphrag_config::{FolderConfig, RuntimeConfig};
use graphrag_db::{Repository, SourceDeleteSummary};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc};

#[derive(Debug)]
pub(crate) struct SyncPartialFailure;
impl std::fmt::Display for SyncPartialFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("folder operation had failed files; see its per-file report")
    }
}
impl std::error::Error for SyncPartialFailure {}

#[derive(Debug)]
pub(crate) struct FolderValidationError(String);
impl std::fmt::Display for FolderValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for FolderValidationError {}

async fn validated_resume_plan(
    repo: &Repository,
    job_id: &str,
) -> Result<(SyncReport, graphrag_db::ProcessingJob)> {
    let job = repo.get_processing_job(job_id).await?.ok_or_else(|| {
        graphrag_agents::AgentError::NotFound(format!("folder sync job {job_id}"))
    })?;
    if job.job_type_enum() != Some(graphrag_db::ProcessingJobType::FolderSync)
        || !matches!(job.status.as_str(), "running" | "failed" | "cancelled")
    {
        return Err(FolderValidationError("sync --resume requires a running, failed, or cancelled folder sync job; completed jobs need a fresh named-folder sync".into()).into());
    }
    Ok(folders::resume_plan(repo, job_id).await?)
}

pub(crate) fn run_config_command(
    config: &RuntimeConfig,
    explicit: Option<&Path>,
    command: &FolderCommand,
) -> Result<bool> {
    match command {
        FolderCommand::Add {
            name,
            path,
            no_recursive,
            include,
            exclude,
            format,
        } => {
            let config_path = graphrag_config::selected_config_path(explicit).context(
                "folder registration requires --config or an available platform config path",
            )?;
            if !config_path.exists() {
                anyhow::bail!("folder registration requires an existing config; run graphrag init --write first using the same --config path");
            }
            let mut folder = FolderConfig {
                path: path.clone(),
                recursive: !no_recursive,
                ..Default::default()
            };
            if !include.is_empty() {
                folder.include = include.clone();
            }
            folder.exclude.extend(exclude.iter().cloned());
            let registration = graphrag_config::register_folder(&config_path, name, folder)?;
            let folder = registration.folder;
            let config_backup = registration.config_backup;
            output::print(
                *format,
                "folders.add",
                serde_json::json!({"name":name,"config_path":config_path,"config_backup":config_backup,"folder":folder}),
                |writer| {
                    writeln!(writer, "Registered {name}: {}", folder.path.display())?;
                    writeln!(
                        writer,
                        "Previous config retained: {}",
                        config_backup.display()
                    )?;
                    writeln!(
                        writer,
                        "Preview: {} sync {} --dry-run",
                        base_command(config, Some(&config_path)),
                        quote(name)
                    )
                },
            )?;
            Ok(true)
        }
        FolderCommand::List { format } => {
            output::print(*format, "folders.list", &config.folders, |writer| {
                if config.folders.is_empty() {
                    writeln!(
                        writer,
                        "No folders registered. Use graphrag folders add NAME PATH."
                    )?;
                }
                for (name, folder) in &config.folders {
                    writeln!(
                        writer,
                        "{name}: {} (recursive={})",
                        folder.path.display(),
                        folder.recursive
                    )?;
                }
                Ok(())
            })?;
            Ok(true)
        }
        FolderCommand::Prune { .. } => Ok(false),
    }
}

fn folder_spec(name: &str, folder: &FolderConfig) -> FolderSpec {
    FolderSpec {
        name: name.into(),
        path: folder.path.clone(),
        recursive: folder.recursive,
        include: folder.include.clone(),
        exclude: folder.exclude.clone(),
    }
}
fn selected_folders(
    config: &RuntimeConfig,
    name: Option<&str>,
    all: bool,
) -> Result<Vec<FolderSpec>> {
    if all {
        if config.folders.is_empty() {
            anyhow::bail!("sync --all requires at least one registered folder");
        }
        Ok(config
            .folders
            .iter()
            .map(|(name, folder)| folder_spec(name, folder))
            .collect())
    } else {
        let name = name.context("sync requires NAME, --all, or --resume JOB")?;
        let folder = config
            .folders
            .get(name)
            .with_context(|| format!("folder not found: {name}; use graphrag folders list"))?;
        Ok(vec![folder_spec(name, folder)])
    }
}

fn quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\\''"))
}
fn base_command(config: &RuntimeConfig, config_path: Option<&Path>) -> String {
    let mut command = "graphrag".to_string();
    if let Some(path) = config_path {
        let path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(path)
        };
        command.push_str(&format!(" --config {}", quote(&path.to_string_lossy())));
    }
    let path = if config.database.path.is_absolute() {
        config.database.path.clone()
    } else {
        std::env::current_dir()
            .unwrap_or_default()
            .join(&config.database.path)
    };
    command.push_str(&format!(" --db-path {}", quote(&path.to_string_lossy())));
    command
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn sync(
    repo: &Repository,
    librarian: &LibrarianAgent,
    config: &RuntimeConfig,
    config_path: Option<&Path>,
    name: Option<String>,
    all: bool,
    dry_run: bool,
    resume: Option<String>,
    format: OutputFormat,
    cancellation: Arc<AtomicBool>,
) -> Result<()> {
    let (plan, resumed) = if let Some(job_id) = &resume {
        let (plan, job) = validated_resume_plan(repo, job_id).await?;
        (plan, Some(job))
    } else {
        (
            folders::discover(repo, selected_folders(config, name.as_deref(), all)?).await?,
            None,
        )
    };
    let mut report = if dry_run {
        plan
    } else {
        folders::execute(repo, librarian, plan, resumed, cancellation).await?
    };
    let base = base_command(config, config_path);
    for file in &mut report.files {
        if file.retry_command.is_some() {
            file.retry_command = Some(match &report.job_id {
                Some(id) if !dry_run && file.source_uri.is_some() => {
                    format!("{base} sync --resume {}", quote(id))
                }
                _ => format!("{base} sync {}", quote(&file.folder)),
            });
        }
    }
    print_report("sync", format, &report)?;
    if report.failed() > 0 || report.cancelled {
        return Err(SyncPartialFailure.into());
    }
    Ok(())
}

fn print_report(command: &str, format: OutputFormat, report: &SyncReport) -> Result<()> {
    let mut envelope = OutputEnvelope::success(command, report);
    envelope.success = report.failed() == 0 && !report.cancelled;
    envelope.warnings = report.warnings.clone();
    envelope.errors = report
        .files
        .iter()
        .filter(|file| file.is_failure())
        .filter_map(|file| file.error.clone())
        .collect();
    output::print_envelope(format, &envelope, |writer| {
        writeln!(
            writer,
            "{}{}",
            if report.dry_run {
                "Folder sync preview"
            } else {
                "Folder sync"
            },
            if report.cancelled { " (cancelled)" } else { "" }
        )?;
        if let Some(job) = &report.job_id {
            writeln!(writer, "Job: {job}")?;
        }
        for file in &report.files {
            writeln!(writer, "{:?}: {} [{}]", file.status, file.path, file.folder)?;
            if let Some(error) = &file.error {
                writeln!(writer, "  {error}")?;
            }
            if let Some(retry) = &file.retry_command {
                writeln!(writer, "  Retry: {retry}")?;
            }
        }
        for warning in &report.warnings {
            writeln!(writer, "Warning: {warning}")?;
        }
        let missing = report
            .files
            .iter()
            .filter(|file| file.status == FileSyncStatus::Missing)
            .count();
        if missing > 0 {
            writeln!(writer, "{missing} missing sources retained. Preview removal separately with graphrag folders prune NAME.")?;
        }
        Ok(())
    })
}

#[derive(Serialize)]
struct PruneFile {
    file: FileSyncResult,
    generated_records: SourceDeleteSummary,
}

fn verify_prune_candidate(folders: &[FolderSpec], candidate: &PruneFile) -> Result<()> {
    let folder = folders
        .iter()
        .find(|folder| folder.name == candidate.file.folder)
        .context("missing source has no pinned folder definition")?;
    if !folders::source_is_missing(folder, Path::new(&candidate.file.path))
        .context("cannot verify missing source before prune; restore access and preview again")?
    {
        anyhow::bail!("source path returned or became a symlink during prune; preview again before confirming");
    }
    Ok(())
}

async fn prune_candidate(
    repo: &Repository,
    folders: &[FolderSpec],
    candidate: &PruneFile,
) -> Result<SourceDeleteSummary> {
    let source = repo
        .get_source(
            candidate
                .file
                .source_id
                .as_deref()
                .context("missing source has no ID")?,
        )
        .await?
        .context("source not found; preview prune again")?;
    // Each earlier cascade may have taken time. Verify this path and its
    // ancestors again after the last await immediately before its own delete.
    verify_prune_candidate(folders, candidate)?;
    Ok(repo.delete_source(&source).await?)
}

pub(crate) async fn prune(
    repo: &Repository,
    config: &RuntimeConfig,
    config_path: Option<&Path>,
    command: FolderCommand,
) -> Result<()> {
    let FolderCommand::Prune {
        name,
        yes,
        revision,
        format,
    } = command
    else {
        unreachable!("config-only folder commands returned before DB startup")
    };
    let report = folders::discover(repo, selected_folders(config, Some(&name), false)?).await?;
    if report.failed() > 0 {
        print_report("folders.prune", format, &report)?;
        return Err(SyncPartialFailure.into());
    }
    let mut files = Vec::new();
    for file in &report.files {
        if file.status != FileSyncStatus::Missing {
            continue;
        }
        let source = repo
            .get_source(
                file.source_id
                    .as_deref()
                    .context("missing source has no ID")?,
            )
            .await?
            .context("source not found; preview prune again")?;
        files.push(PruneFile {
            file: file.clone(),
            generated_records: repo.preview_source_delete(&source).await?,
        });
    }
    let bytes = serde_json::to_vec(&(&report.folders, &files))?;
    let current_revision = format!("{:x}", Sha256::digest(bytes));
    if yes && revision.as_deref() != Some(&current_revision) {
        anyhow::bail!("prune confirmation requires the current preview token; rerun folders prune {name} and confirm its new --revision token");
    }
    let mut errors = Vec::new();
    if yes {
        // Reject an invalid batch before any cascade, then recheck each source
        // individually so a later restored file is retained after prior work.
        for candidate in &files {
            verify_prune_candidate(&report.folders, candidate)?;
        }
        for candidate in &mut files {
            match prune_candidate(repo, &report.folders, candidate).await {
                Ok(deleted) => {
                    candidate.generated_records = deleted;
                    candidate.file.status = FileSyncStatus::Pruned;
                }
                Err(error) => {
                    let message = format!("{}: {error:#}", candidate.file.path);
                    candidate.file.status = FileSyncStatus::Failed;
                    candidate.file.error = Some(message.clone());
                    candidate.file.retry_command = Some(format!(
                        "{} folders prune {}",
                        base_command(config, config_path),
                        quote(&name)
                    ));
                    errors.push(message);
                    break;
                }
            }
        }
    }
    let confirm = format!(
        "{} folders prune {} --yes --revision {}",
        base_command(config, config_path),
        quote(&name),
        quote(&current_revision)
    );
    let mut envelope = OutputEnvelope::success(
        "folders.prune",
        serde_json::json!({"dry_run":!yes,"folder":name,"revision":current_revision,"files":files,"confirm_command":confirm}),
    );
    envelope.success = errors.is_empty();
    envelope.errors = errors;
    output::print_envelope(format, &envelope, |writer| {
        writeln!(
            writer,
            "{} {} missing sources for {name}.",
            if yes { "Pruned" } else { "Preview:" },
            if yes {
                files
                    .iter()
                    .filter(|candidate| candidate.file.status == FileSyncStatus::Pruned)
                    .count()
            } else {
                files.len()
            }
        )?;
        for candidate in &files {
            writeln!(
                writer,
                "{:?}: {}: {} generated notes; manual notes retained",
                candidate.file.status, candidate.file.path, candidate.generated_records.notes
            )?;
            if let Some(error) = &candidate.file.error {
                writeln!(writer, "  {error}")?;
            }
            if let Some(retry) = &candidate.file.retry_command {
                writeln!(writer, "  Preview again: {retry}")?;
            }
        }
        if !yes {
            writeln!(writer, "Confirm: {confirm}")?;
        }
        Ok(())
    })?;
    if !envelope.success {
        return Err(SyncPartialFailure.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use graphrag_core::{normalize_file_uri, normalized_content_hash, Note, SourceType};

    #[tokio::test]
    async fn resume_invocation_errors_are_validation_missing_jobs_are_not_found() {
        use graphrag_db::{ProcessingJobStatus, ProcessingJobType, ProcessingJobUpdate};
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        let wrong = repo
            .create_processing_job_with_scope(ProcessingJobType::Embedding, None, 0, None, vec![])
            .await
            .unwrap();
        let complete = repo
            .create_processing_job_with_scope(ProcessingJobType::FolderSync, None, 0, None, vec![])
            .await
            .unwrap();
        repo.update_processing_job(
            complete.id.as_ref().unwrap(),
            ProcessingJobUpdate {
                status: Some(ProcessingJobStatus::Completed),
                finish: true,
                ..Default::default()
            },
        )
        .await
        .unwrap();
        for job in [wrong, complete] {
            let error = validated_resume_plan(
                &repo,
                &graphrag_core::record_id_to_string(job.id.as_ref().unwrap()),
            )
            .await
            .unwrap_err();
            assert_eq!(
                crate::app::exit_code_for(&error),
                output::ExitCode::Validation
            );
        }
        let error = validated_resume_plan(&repo, "processing_job:missing")
            .await
            .unwrap_err();
        assert_eq!(
            crate::app::exit_code_for(&error),
            output::ExitCode::NotFound
        );
        assert!(repo
            .list_processing_jobs(10)
            .await
            .unwrap()
            .iter()
            .all(|job| job.failed_count == 0));
    }

    #[tokio::test]
    async fn each_prune_rechecks_paths_after_prior_cascades_and_retains_restored_source() {
        let directory = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(directory.path()).unwrap();
        let folder = FolderSpec {
            name: "notes".into(),
            path: root.clone(),
            recursive: true,
            include: vec!["**/*.md".into()],
            exclude: vec![],
        };
        let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
        for name in ["a.md", "b.md"] {
            let path = root.join(name);
            let content = format!("# Note\nStored {name} content.");
            std::fs::write(&path, &content).unwrap();
            let mut plan = repo
                .begin_file_import(
                    SourceType::Markdown,
                    name.into(),
                    normalize_file_uri(&path).unwrap(),
                    content.clone(),
                    normalized_content_hash(&content),
                    false,
                )
                .await
                .unwrap();
            repo.create_note(
                Note::new(content)
                    .with_source(plan.source.id.clone().unwrap())
                    .with_source_generation(plan.source.generation),
            )
            .await
            .unwrap();
            repo.complete_file_import(&mut plan.source).await.unwrap();
            std::fs::remove_file(path).unwrap();
        }
        let report = folders::discover(&repo, vec![folder.clone()])
            .await
            .unwrap();
        let mut candidates = Vec::new();
        for file in report.files {
            let source = repo
                .get_source(file.source_id.as_deref().unwrap())
                .await
                .unwrap()
                .unwrap();
            candidates.push(PruneFile {
                file,
                generated_records: repo.preview_source_delete(&source).await.unwrap(),
            });
        }
        for candidate in &candidates {
            verify_prune_candidate(std::slice::from_ref(&folder), candidate).unwrap();
        }
        let deleted = prune_candidate(&repo, std::slice::from_ref(&folder), &candidates[0])
            .await
            .unwrap();
        assert_eq!(deleted.notes, 1);
        // A filesystem event during the previous cascade invalidates only the
        // later candidate. Its source and visible notes must remain.
        std::fs::write(
            root.join("b.md"),
            "# Restored\nReturned during the first cascade.",
        )
        .unwrap();
        assert!(prune_candidate(&repo, &[folder], &candidates[1])
            .await
            .unwrap_err()
            .to_string()
            .contains("returned"));
        assert_eq!(repo.list_sources().await.unwrap().len(), 1);
        assert_eq!(repo.list_notes(100).await.unwrap().len(), 1);
        assert!(repo.list_notes(100).await.unwrap()[0]
            .content
            .contains("b.md"));
    }
}
