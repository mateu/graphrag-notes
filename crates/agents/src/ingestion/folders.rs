//! Explicit, deterministic host-local Markdown sync. Discovery is read-only;
//! inference and source promotion happen only for changed files in `execute`.

use crate::{AgentError, LibrarianAgent, Result};
use glob::{MatchOptions, Pattern};
use graphrag_core::{normalize_file_uri, normalized_content_hash, record_id_to_string, SourceIngestionStatus};
use graphrag_db::{repository::FileSourceSnapshot, ProcessingJob, ProcessingJobStatus, ProcessingJobType, ProcessingJobUpdate, Repository, SourceImportAction};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{atomic::{AtomicBool, Ordering}, Arc};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct FolderSpec {
    pub name: String,
    pub path: PathBuf,
    pub recursive: bool,
    pub include: Vec<String>,
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FileSyncStatus { Created, Updated, Unchanged, Failed, Missing, Ignored, Pruned }

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSyncResult {
    pub folder: String,
    pub path: String,
    pub source_uri: Option<String>,
    pub source_id: Option<String>,
    pub status: FileSyncStatus,
    pub generation: Option<u64>,
    pub content_hash: Option<String>,
    pub error: Option<String>,
    pub retry_command: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncReport {
    pub dry_run: bool,
    pub job_id: Option<String>,
    pub cancelled: bool,
    pub folders: Vec<FolderSpec>,
    pub files: Vec<FileSyncResult>,
    pub warnings: Vec<String>,
}

impl SyncReport {
    pub fn failed(&self) -> usize { self.files.iter().filter(|file| file.status == FileSyncStatus::Failed).count() }
    pub fn needs_inference(&self) -> bool { self.files.iter().any(|file| matches!(file.status, FileSyncStatus::Created | FileSyncStatus::Updated)) }
    pub fn missing_revision(&self) -> Result<String> {
        let missing = self.files.iter().filter(|file| file.status == FileSyncStatus::Missing).collect::<Vec<_>>();
        let bytes = serde_json::to_vec(&(&self.folders, missing)).map_err(|error| AgentError::Processing(error.to_string()))?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }
}

struct Rules { include: Vec<Pattern>, exclude: Vec<Pattern> }
impl Rules {
    fn new(folder: &FolderSpec) -> Result<Self> {
        let compile = |patterns: &[String]| patterns.iter().map(|pattern| Pattern::new(pattern).map_err(|error| AgentError::Processing(format!("invalid folder glob {pattern:?}: {error}")))).collect::<Result<Vec<_>>>();
        Ok(Self { include: compile(&folder.include)?, exclude: compile(&folder.exclude)? })
    }
    fn matches(patterns: &[Pattern], relative: &str) -> bool {
        let options = MatchOptions { case_sensitive: true, require_literal_separator: true, require_literal_leading_dot: false };
        patterns.iter().any(|pattern| pattern.matches_with(relative, options))
    }
    fn excluded(&self, relative: &str) -> bool { Self::matches(&self.exclude, relative) }
    fn included(&self, relative: &str) -> bool { Self::matches(&self.include, relative) && !self.excluded(relative) }
}

fn row(folder: &FolderSpec, path: &Path, status: FileSyncStatus) -> FileSyncResult {
    FileSyncResult { folder: folder.name.clone(), path: path.to_string_lossy().into_owned(), source_uri: None, source_id: None, status, generation: None, content_hash: None, error: None, retry_command: None }
}
fn failed(folder: &FolderSpec, path: &Path, error: impl ToString) -> FileSyncResult {
    let mut result = row(folder, path, FileSyncStatus::Failed);
    result.error = Some(error.to_string());
    result.retry_command = Some(format!("graphrag sync {}", folder.name));
    result
}

/// Plan every selected folder before any inference, checkpoint, or source write.
/// Missing conclusions are suppressed for an incomplete/unavailable root.
pub async fn discover(repo: &Repository, folders: Vec<FolderSpec>) -> Result<SyncReport> {
    for (index, folder) in folders.iter().enumerate() {
        if !folder.path.is_absolute() || folder.path.to_str().is_none() || folder.include.is_empty() {
            return Err(AgentError::Processing("folder sync requires absolute UTF-8 roots and nonempty include rules".into()));
        }
        if folders[..index].iter().any(|other| folder.path.starts_with(&other.path) || other.path.starts_with(&folder.path)) {
            return Err(AgentError::Processing("folder sync requires distinct, nonoverlapping roots".into()));
        }
    }
    let sources = repo.file_source_snapshots().await?;
    let sources = sources.iter().filter_map(|source| source.normalized_uri.as_ref().or(source.uri.as_ref()).map(|uri| (uri.clone(), source.clone()))).collect::<BTreeMap<_,_>>();
    let mut report = SyncReport { dry_run: true, job_id: None, cancelled: false, folders, files: Vec::new(), warnings: Vec::new() };
    for folder in &report.folders {
        let rules = Rules::new(folder)?;
        let mut found = BTreeSet::new();
        let mut complete = true;
        match std::fs::symlink_metadata(&folder.path) {
            Ok(metadata) if metadata.is_dir() && !metadata.file_type().is_symlink() => {},
            Ok(_) => {
                report.files.push(failed(folder, &folder.path, "Registered root requires an actual directory; symlink roots must be registered through their canonical target."));
                continue;
            }
            Err(error) => {
                report.files.push(failed(folder, &folder.path, format!("Folder is unavailable: {error}. Restore the root and retry; no missing files were inferred.")));
                continue;
            }
        }
        scan(folder, &folder.path, &rules, &sources, &mut found, &mut report.files, &mut complete);
        if !complete {
            report.warnings.push(format!("Folder {} was not fully scanned; missing-file reporting and pruning are disabled for this folder.", folder.name));
            continue;
        }
        for (uri, source) in &sources {
            if found.contains(uri) { continue; }
            let Some(path) = uri.strip_prefix("file://").map(PathBuf::from) else { continue; };
            let Ok(relative) = path.strip_prefix(&folder.path) else { continue; };
            let Some(relative) = relative.to_str() else { continue; };
            let relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
            if (!folder.recursive && relative.contains('/')) || !rules.included(&relative) { continue; }
            // Excluded files, symlink replacements, and unreadable files are
            // retained. Absence is established only by an explicit NotFound.
            if std::fs::symlink_metadata(&path).is_err_and(|error| error.kind() == std::io::ErrorKind::NotFound) {
                let mut missing = row(folder, &path, FileSyncStatus::Missing);
                missing.source_uri = Some(uri.clone());
                missing.source_id = Some(record_id_to_string(&source.id));
                missing.generation = Some(source.successful_generation);
                missing.content_hash = source.content_hash.clone();
                report.files.push(missing);
            }
        }
    }
    report.files.sort_by(|a,b| (&a.folder, &a.path).cmp(&(&b.folder, &b.path)));
    Ok(report)
}

fn scan(folder: &FolderSpec, directory: &Path, rules: &Rules, sources: &BTreeMap<String, FileSourceSnapshot>, found: &mut BTreeSet<String>, files: &mut Vec<FileSyncResult>, complete: &mut bool) {
    let entries = match std::fs::read_dir(directory) {
        Ok(entries) => entries.collect::<std::io::Result<Vec<_>>>(),
        Err(error) => { files.push(failed(folder, directory, error)); *complete = false; return; }
    };
    let mut entries = match entries { Ok(entries) => entries, Err(error) => { files.push(failed(folder, directory, error)); *complete = false; return; } };
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let Some(relative) = path.strip_prefix(&folder.path).ok().and_then(Path::to_str) else {
            files.push(failed(folder, &path, "File path is not valid UTF-8; rename it before syncing.")); *complete = false; continue;
        };
        let relative = relative.replace(std::path::MAIN_SEPARATOR, "/");
        let kind = match entry.file_type() { Ok(kind) => kind, Err(error) => { files.push(failed(folder, &path, error)); *complete = false; continue; } };
        if kind.is_symlink() {
            let mut ignored = row(folder, &path, FileSyncStatus::Ignored);
            ignored.error = Some("Symlinks are never followed.".into()); files.push(ignored); continue;
        }
        if rules.excluded(&relative) || (kind.is_dir() && rules.excluded(&format!("{relative}/"))) { continue; }
        if kind.is_dir() {
            if folder.recursive { scan(folder, &path, rules, sources, found, files, complete); }
            continue;
        }
        if !kind.is_file() || !rules.included(&relative) { continue; }
        let uri = match normalize_file_uri(&path) { Ok(uri) => uri, Err(error) => { files.push(failed(folder, &path, error)); *complete = false; continue; } };
        if !uri.strip_prefix("file://").map(Path::new).is_some_and(|canonical| canonical.starts_with(&folder.path)) {
            files.push(failed(folder, &path, "File resolved outside the registered root; symlinks are never followed.")); *complete = false; continue;
        }
        found.insert(uri.clone());
        let content = match std::fs::read_to_string(&path) { Ok(content) => content, Err(error) => { let mut result = failed(folder, &path, error); result.source_uri = Some(uri); files.push(result); *complete = false; continue; } };
        let hash = normalized_content_hash(&content);
        let source = sources.get(&uri);
        let status = match source {
            Some(source) if source.status == SourceIngestionStatus::Ready && source.content_hash.as_deref() == Some(&hash) => FileSyncStatus::Unchanged,
            Some(_) => FileSyncStatus::Updated,
            None => FileSyncStatus::Created,
        };
        let mut result = row(folder, &path, status);
        result.source_uri = Some(uri); result.content_hash = Some(hash);
        result.source_id = source.map(|source| record_id_to_string(&source.id));
        result.generation = source.map(|source| source.successful_generation);
        files.push(result);
    }
}

/// Persisted job scope pins host roots and rules. Resume never substitutes the
/// current configuration or silently starts newly discovered files.
pub async fn resume_plan(repo: &Repository, job_id: &str) -> Result<(SyncReport, ProcessingJob)> {
    let job = repo.get_processing_job(job_id).await?.ok_or_else(|| AgentError::NotFound(format!("folder sync job {job_id}")))?;
    if job.job_type_enum() != Some(ProcessingJobType::FolderSync) || !matches!(job.status.as_str(), "running" | "failed" | "cancelled") {
        return Err(AgentError::Processing("sync --resume requires a running, failed, or cancelled folder sync job".into()));
    }
    let folders: Vec<FolderSpec> = serde_json::from_str(job.scope.as_deref().ok_or_else(|| AgentError::Processing("folder sync job has no pinned definitions".into()))?)
        .map_err(|error| AgentError::Processing(format!("invalid pinned folder definitions: {error}")))?;
    let mut report = discover(repo, folders).await?;
    let pinned = job.item_ids.iter().collect::<BTreeSet<_>>();
    report.files.retain(|file| file.status == FileSyncStatus::Failed && file.source_uri.is_none() || file.source_uri.as_ref().is_some_and(|uri| pinned.contains(uri)));
    for uri in &job.item_ids {
        if report.files.iter().any(|file| file.source_uri.as_ref() == Some(uri)) { continue; }
        let path = uri.strip_prefix("file://").map(PathBuf::from).ok_or_else(|| AgentError::Processing("folder sync job contains a non-file identity".into()))?;
        let folder = report.folders.iter().find(|folder| path.starts_with(&folder.path)).ok_or_else(|| AgentError::Processing("folder sync job contains an identity outside its pinned roots".into()))?;
        let mut missing = row(folder, &path, FileSyncStatus::Missing);
        missing.source_uri = Some(uri.clone());
        missing.error = Some("Pinned file is unavailable or no longer matches its original rules; restore it and resume, or start a new sync.".into());
        missing.retry_command = Some(format!("graphrag sync --resume {job_id}"));
        report.files.push(missing);
    }
    report.files.sort_by(|a,b| (&a.folder, &a.path).cmp(&(&b.folder, &b.path)));
    report.job_id = Some(record_id_to_string(job.id.as_ref().ok_or_else(|| AgentError::Processing("folder job has no ID".into()))?));
    Ok((report, job))
}

pub async fn execute(repo: &Repository, librarian: &LibrarianAgent, mut report: SyncReport, resume: Option<ProcessingJob>, cancellation: Arc<AtomicBool>) -> Result<SyncReport> {
    report.dry_run = false;
    let item_ids = report.files.iter().filter(|file| matches!(file.status, FileSyncStatus::Created | FileSyncStatus::Updated)).filter_map(|file| file.source_uri.clone()).collect::<Vec<_>>();
    if item_ids.is_empty() && resume.is_none() { return Ok(report); }
    let job = match resume {
        Some(job) => repo.resume_folder_sync_job(job.id.as_ref().ok_or_else(|| AgentError::Processing("folder sync job has no ID".into()))?).await?,
        None => repo.create_processing_job_with_scope(ProcessingJobType::FolderSync, None, item_ids.len() as u64, Some(serde_json::to_string(&report.folders).map_err(|error| AgentError::Processing(error.to_string()))?), item_ids).await?,
    };
    let id = job.id.as_ref().ok_or_else(|| AgentError::Processing("folder sync job has no ID".into()))?;
    let job_id = record_id_to_string(id); report.job_id = Some(job_id.clone());
    let mut completed = 0u64;
    let mut failures = 0u64;
    for file in &mut report.files {
        if cancellation.load(Ordering::Acquire) { report.cancelled = true; break; }
        if !job.item_ids.iter().any(|uri| file.source_uri.as_ref() == Some(uri)) { continue; }
        if matches!(file.status, FileSyncStatus::Created | FileSyncStatus::Updated) {
            let result = match read_planned_file(file) {
                Ok(content) if Some(normalized_content_hash(&content)) == file.content_hash => librarian.ingest_markdown_with_options(&file.path, content, false).await,
                Ok(_) => Err(AgentError::Processing("File changed after discovery; rerun sync to use its latest contents.".into())),
                Err(error) => Err(AgentError::Processing(error.to_string())),
            };
            match result {
                Ok(imported) => {
                    file.source_id = Some(imported.source_id); file.generation = Some(imported.generation);
                    file.status = match imported.action { SourceImportAction::Created => FileSyncStatus::Created, SourceImportAction::Updated => FileSyncStatus::Updated, SourceImportAction::Unchanged => FileSyncStatus::Unchanged };
                }
                Err(error) => {
                    file.status = FileSyncStatus::Failed; file.error = Some(error.to_string());
                    file.retry_command = Some(format!("graphrag sync --resume {job_id}"));
                }
            }
        }
        if file.status == FileSyncStatus::Failed || file.status == FileSyncStatus::Missing { failures += 1; }
        else { completed += 1; }
        repo.update_processing_job(id, ProcessingJobUpdate { completed_count: Some(completed), failed_count: Some(failures), checkpoint: Some(file.source_uri.clone()), last_error: Some(file.error.clone()), ..Default::default() }).await?;
    }
    let status = if report.cancelled { ProcessingJobStatus::Cancelled } else if failures > 0 || report.failed() > 0 { ProcessingJobStatus::Failed } else { ProcessingJobStatus::Completed };
    repo.update_processing_job(id, ProcessingJobUpdate { status: Some(status), completed_count: Some(completed), failed_count: Some(failures), finish: true, ..Default::default() }).await?;
    Ok(report)
}

fn read_planned_file(file: &FileSyncResult) -> std::result::Result<String, std::io::Error> {
    let path = Path::new(&file.path);
    let metadata = std::fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(std::io::Error::other("Planned path became a symlink or stopped being a regular file; rerun sync."));
    }
    let current_uri = normalize_file_uri(path).map_err(|error| std::io::Error::other(error.to_string()))?;
    if file.source_uri.as_deref() != Some(&current_uri) {
        return Err(std::io::Error::other("Planned file identity changed; rerun sync."));
    }
    std::fs::read_to_string(path)
}
