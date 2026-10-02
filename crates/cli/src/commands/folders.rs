//! Explicit folder registration, sync reports, and confirmed missing-source pruning.
use crate::cli::FolderCommand;
use crate::output::{self, OutputEnvelope, OutputFormat};
use anyhow::{Context, Result};
use graphrag_agents::{ingestion::folders::{self, FileSyncResult, FileSyncStatus, FolderSpec, SyncReport}, LibrarianAgent};
use graphrag_config::{FolderConfig, RuntimeConfig};
use graphrag_db::{Repository, SourceDeleteSummary};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::{atomic::AtomicBool, Arc};

#[derive(Debug)]
pub(crate) struct SyncPartialFailure;
impl std::fmt::Display for SyncPartialFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str("folder operation had failed files; see its per-file report") }
}
impl std::error::Error for SyncPartialFailure {}

pub(crate) fn run_config_command(config: &RuntimeConfig, explicit: Option<&Path>, command: &FolderCommand) -> Result<bool> {
    match command {
        FolderCommand::Add { name, path, no_recursive, include, exclude, format } => {
            let config_path = graphrag_config::selected_config_path(explicit).context("folder registration requires --config or an available platform config path")?;
            if !config_path.exists() { anyhow::bail!("folder registration requires an existing config; run graphrag init --write first using the same --config path"); }
            let mut folder = FolderConfig { path: path.clone(), recursive: !no_recursive, ..Default::default() };
            if !include.is_empty() { folder.include = include.clone(); }
            folder.exclude.extend(exclude.iter().cloned());
            let folder = graphrag_config::register_folder(&config_path, name, folder)?;
            output::print(*format, "folders.add", serde_json::json!({"name":name,"config_path":config_path,"folder":folder}), |writer| {
                writeln!(writer, "Registered {name}: {}", folder.path.display())?;
                writeln!(writer, "Preview: graphrag sync {name} --dry-run")
            })?;
            Ok(true)
        }
        FolderCommand::List { format } => {
            output::print(*format, "folders.list", &config.folders, |writer| {
                if config.folders.is_empty() { writeln!(writer, "No folders registered. Use graphrag folders add NAME PATH.")?; }
                for (name, folder) in &config.folders { writeln!(writer, "{name}: {} (recursive={})", folder.path.display(), folder.recursive)?; }
                Ok(())
            })?;
            Ok(true)
        }
        FolderCommand::Prune { .. } => Ok(false),
    }
}

fn folder_spec(name: &str, folder: &FolderConfig) -> FolderSpec {
    FolderSpec { name: name.into(), path: folder.path.clone(), recursive: folder.recursive, include: folder.include.clone(), exclude: folder.exclude.clone() }
}
fn selected_folders(config: &RuntimeConfig, name: Option<&str>, all: bool) -> Result<Vec<FolderSpec>> {
    if all {
        if config.folders.is_empty() { anyhow::bail!("sync --all requires at least one registered folder"); }
        Ok(config.folders.iter().map(|(name, folder)| folder_spec(name, folder)).collect())
    } else {
        let name = name.context("sync requires NAME, --all, or --resume JOB")?;
        let folder = config.folders.get(name).with_context(|| format!("folder not found: {name}; use graphrag folders list"))?;
        Ok(vec![folder_spec(name, folder)])
    }
}

fn quote(value: &str) -> String { format!("'{}'", value.replace('\'', "'\\''")) }
fn base_command(config: &RuntimeConfig, config_path: Option<&Path>) -> String {
    let mut command = "graphrag".to_string();
    if let Some(path) = config_path {
        let path = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(path) };
        command.push_str(&format!(" --config {}", quote(&path.to_string_lossy())));
    }
    let path = if config.database.path.is_absolute() { config.database.path.clone() } else { std::env::current_dir().unwrap_or_default().join(&config.database.path) };
    command.push_str(&format!(" --db-path {}", quote(&path.to_string_lossy())));
    command
}

#[allow(clippy::too_many_arguments)]
pub(crate) async fn sync(repo: &Repository, librarian: &LibrarianAgent, config: &RuntimeConfig, config_path: Option<&Path>, name: Option<String>, all: bool, dry_run: bool, resume: Option<String>, format: OutputFormat, cancellation: Arc<AtomicBool>) -> Result<()> {
    let (plan, resumed) = if let Some(job_id) = &resume {
        let (plan, job) = folders::resume_plan(repo, job_id).await?; (plan, Some(job))
    } else { (folders::discover(repo, selected_folders(config, name.as_deref(), all)?).await?, None) };
    let mut report = if dry_run { plan } else { folders::execute(repo, librarian, plan, resumed, cancellation).await? };
    let base = base_command(config, config_path);
    for file in &mut report.files {
        if file.retry_command.is_some() {
            file.retry_command = Some(match &report.job_id {
                Some(id) if !dry_run => format!("{base} sync --resume {}", quote(id)),
                _ => format!("{base} sync {}", quote(&file.folder)),
            });
        }
    }
    print_report("sync", format, &report)?;
    if report.failed() > 0 || report.cancelled { return Err(SyncPartialFailure.into()); }
    Ok(())
}

fn print_report(command: &str, format: OutputFormat, report: &SyncReport) -> Result<()> {
    let mut envelope = OutputEnvelope::success(command, report);
    envelope.success = report.failed() == 0 && !report.cancelled;
    envelope.warnings = report.warnings.clone();
    envelope.errors = report.files.iter().filter(|file| file.status == FileSyncStatus::Failed).filter_map(|file| file.error.clone()).collect();
    output::print_envelope(format, &envelope, |writer| {
        writeln!(writer, "{}{}", if report.dry_run { "Folder sync preview" } else { "Folder sync" }, if report.cancelled { " (cancelled)" } else { "" })?;
        if let Some(job) = &report.job_id { writeln!(writer, "Job: {job}")?; }
        for file in &report.files {
            writeln!(writer, "{:?}: {} [{}]", file.status, file.path, file.folder)?;
            if let Some(error) = &file.error { writeln!(writer, "  {error}")?; }
            if let Some(retry) = &file.retry_command { writeln!(writer, "  Retry: {retry}")?; }
        }
        for warning in &report.warnings { writeln!(writer, "Warning: {warning}")?; }
        let missing = report.files.iter().filter(|file| file.status == FileSyncStatus::Missing).count();
        if missing > 0 { writeln!(writer, "{missing} missing sources retained. Preview removal separately with graphrag folders prune NAME.")?; }
        Ok(())
    })
}

#[derive(Serialize)]
struct PruneFile { file: FileSyncResult, generated_records: SourceDeleteSummary }

pub(crate) async fn prune(repo: &Repository, config: &RuntimeConfig, config_path: Option<&Path>, command: FolderCommand) -> Result<()> {
    let FolderCommand::Prune { name, yes, revision, format } = command else { unreachable!("config-only folder commands returned before DB startup") };
    let report = folders::discover(repo, selected_folders(config, Some(&name), false)?).await?;
    if report.failed() > 0 {
        print_report("folders.prune", format, &report)?;
        return Err(SyncPartialFailure.into());
    }
    let mut files = Vec::new();
    for file in &report.files {
        if file.status != FileSyncStatus::Missing { continue; }
        let source = repo.get_source(file.source_id.as_deref().context("missing source has no ID")?).await?.context("source not found; preview prune again")?;
        files.push(PruneFile { file: file.clone(), generated_records: repo.preview_source_delete(&source).await? });
    }
    let bytes = serde_json::to_vec(&(&report.folders, &files))?;
    let current_revision = format!("{:x}", Sha256::digest(bytes));
    if yes && revision.as_deref() != Some(&current_revision) { anyhow::bail!("prune preview changed; rerun folders prune {name} and confirm its new --revision token"); }
    if yes {
        // Recheck missing state immediately before each source cascade. A
        // restored file or incomplete root is never removed by an old plan.
        for candidate in &files {
            if std::fs::symlink_metadata(&candidate.file.path).is_ok() { anyhow::bail!("source file returned during prune; preview again before confirming"); }
        }
        for candidate in &mut files {
            let source = repo.get_source(candidate.file.source_id.as_deref().expect("checked source ID")).await?.context("source not found; preview prune again")?;
            candidate.generated_records = repo.delete_source(&source).await?;
            candidate.file.status = FileSyncStatus::Pruned;
        }
    }
    let confirm = format!("{} folders prune {} --yes --revision {}", base_command(config, config_path), quote(&name), quote(&current_revision));
    output::print(format, "folders.prune", serde_json::json!({"dry_run":!yes,"folder":name,"revision":current_revision,"files":files,"confirm_command":confirm}), |writer| {
        writeln!(writer, "{} {} missing sources for {name}.", if yes { "Pruned" } else { "Preview:" }, files.len())?;
        for candidate in &files { writeln!(writer, "{}: {} generated notes; manual notes retained", candidate.file.path, candidate.generated_records.notes)?; }
        if !yes { writeln!(writer, "Confirm: {confirm}")?; }
        Ok(())
    })
}
