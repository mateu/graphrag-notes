use super::*;
use crate::{
    DeterministicEmbedder, Embedder, FixtureEntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig,
};
use async_trait::async_trait;
use graphrag_core::Note;
use std::sync::{atomic::AtomicUsize, Mutex};

#[derive(Default)]
struct Probe {
    provider: DeterministicEmbedder,
    requests: AtomicUsize,
    health_checks: AtomicUsize,
    reject: Mutex<Option<String>>,
    cancel_on: Mutex<Option<(String, Arc<AtomicBool>)>>,
}
impl Probe {
    fn record(&self, texts: &[String]) -> Result<()> {
        self.requests.fetch_add(1, Ordering::SeqCst);
        if let Some((needle, flag)) = self.cancel_on.lock().unwrap().as_ref() {
            if texts.iter().any(|text| text.contains(needle)) {
                flag.store(true, Ordering::Release);
            }
        }
        if self
            .reject
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|needle| texts.iter().any(|text| text.contains(needle)))
        {
            return Err(AgentError::InferenceService(
                "offline fixture rejected this file".into(),
            ));
        }
        Ok(())
    }
}
#[async_trait]
impl Embedder for Probe {
    async fn embed(&self, text: &str, query: bool) -> Result<Vec<f32>> {
        self.record(&[text.into()])?;
        self.provider.embed(text, query).await
    }
    async fn embed_batch(&self, texts: &[String], query: bool) -> Result<Vec<Vec<f32>>> {
        self.record(texts)?;
        self.provider.embed_batch(texts, query).await
    }
    async fn health(&self) -> Result<bool> {
        self.health_checks.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        self.provider.capabilities()
    }
}
async fn fixture() -> (Repository, Arc<Probe>, LibrarianAgent) {
    let repo = Repository::new(graphrag_db::init_memory().await.unwrap());
    let probe = Arc::new(Probe::default());
    let librarian = LibrarianAgent::new(
        repo.clone(),
        probe.clone(),
        Arc::new(FixtureEntityExtractor::default()),
    )
    .with_runtime_config(LibrarianRuntimeConfig {
        skip_entity_extraction: true,
        ..Default::default()
    });
    (repo, probe, librarian)
}
fn spec(root: &Path) -> FolderSpec {
    FolderSpec {
        name: "notes".into(),
        path: std::fs::canonicalize(root).unwrap(),
        recursive: true,
        include: vec!["**/*.md".into()],
        exclude: vec!["**/.git/**".into(), "private/**".into()],
    }
}
fn flag() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

#[tokio::test]
async fn unchanged_and_newline_only_sync_are_provider_free_and_create_no_job() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("café.md"),
        "# Launch\nThe searchable launch plan.\n",
    )
    .unwrap();
    let (repo, probe, librarian) = fixture().await;
    let folder = spec(dir.path());
    let preview = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert_eq!(preview.files[0].status, FileSyncStatus::Created);
    assert!(repo.list_sources().await.unwrap().is_empty());
    assert!(repo.list_processing_jobs(10).await.unwrap().is_empty());
    assert_eq!(probe.requests.load(Ordering::SeqCst), 0);
    let initial = execute(&repo, &librarian, preview, None, flag())
        .await
        .unwrap();
    assert!(initial.job_id.is_some());
    assert_eq!(initial.failed(), 0);
    let notes = repo.list_notes(100).await.unwrap();
    let jobs = repo.list_processing_jobs(10).await.unwrap().len();
    let requests = probe.requests.load(Ordering::SeqCst);
    std::fs::write(
        dir.path().join("café.md"),
        "# Launch\r\nThe searchable launch plan.\r\n",
    )
    .unwrap();
    let no_change = discover(&repo, vec![folder]).await.unwrap();
    assert!(!no_change.needs_inference());
    let no_change = execute(&repo, &librarian, no_change, None, flag())
        .await
        .unwrap();
    assert_eq!(no_change.files[0].status, FileSyncStatus::Unchanged);
    assert!(no_change.job_id.is_none());
    assert_eq!(
        repo.list_notes(100)
            .await
            .unwrap()
            .iter()
            .map(|note| &note.id)
            .collect::<Vec<_>>(),
        notes.iter().map(|note| &note.id).collect::<Vec<_>>()
    );
    assert_eq!(repo.list_processing_jobs(10).await.unwrap().len(), jobs);
    assert_eq!(probe.requests.load(Ordering::SeqCst), requests);
    assert_eq!(probe.health_checks.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn failed_refresh_keeps_old_generation_peers_complete_and_pinned_resume_recovers() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.md"), "# A\nStable first file.").unwrap();
    std::fs::write(dir.path().join("b.md"), "# B\nOld searchable content.").unwrap();
    let (repo, probe, librarian) = fixture().await;
    let folder = spec(dir.path());
    execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    let old_hit = repo
        .list_notes(100)
        .await
        .unwrap()
        .into_iter()
        .find(|note| note.content.contains("Old searchable"))
        .unwrap();
    let old = repo
        .get_note(&record_id_to_string(&old_hit.id))
        .await
        .unwrap()
        .unwrap();
    std::fs::write(dir.path().join("b.md"), "# B\nREJECT new content.").unwrap();
    std::fs::write(dir.path().join("c.md"), "# C\nA successful peer.").unwrap();
    *probe.reject.lock().unwrap() = Some("REJECT".into());
    let failed = execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    assert_eq!(failed.failed(), 1);
    assert!(failed
        .files
        .iter()
        .any(|file| file.path.ends_with("c.md") && file.status == FileSyncStatus::Created));
    assert!(repo
        .get_visible_note(&record_id_to_string(old.id.as_ref().unwrap()))
        .await
        .unwrap()
        .is_some());
    let source = repo
        .get_source(
            old.source_id
                .as_ref()
                .map(record_id_to_string)
                .as_ref()
                .unwrap(),
        )
        .await
        .unwrap()
        .unwrap();
    assert_eq!(source.successful_generation, 1);
    assert_eq!(source.status, SourceIngestionStatus::Failed);
    let job_id = failed.job_id.unwrap();
    let job = repo.get_processing_job(&job_id).await.unwrap().unwrap();
    assert_eq!(job.status, "failed");
    assert_eq!(job.completed_count, 1);
    assert_eq!(job.failed_count, 1);
    std::fs::write(dir.path().join("d.md"), "# D\nNew unpinned file.").unwrap();
    *probe.reject.lock().unwrap() = None;
    let requests = probe.requests.load(Ordering::SeqCst);
    let (plan, job) = resume_plan(&repo, &job_id).await.unwrap();
    assert!(!plan.files.iter().any(|file| file.path.ends_with("d.md")));
    let recovered = execute(&repo, &librarian, plan, Some(job), flag())
        .await
        .unwrap();
    assert_eq!(recovered.failed(), 0);
    assert_eq!(probe.requests.load(Ordering::SeqCst), requests + 1);
    assert!(repo
        .get_visible_note(&record_id_to_string(old.id.as_ref().unwrap()))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        repo.get_processing_job(&job_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        "completed"
    );
}

#[tokio::test]
async fn missing_failed_new_file_remains_a_failed_resume_until_restored() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("retry.md");
    std::fs::write(&path, "# Retry\nREJECT first import.").unwrap();
    let (repo, probe, librarian) = fixture().await;
    *probe.reject.lock().unwrap() = Some("REJECT".into());
    let failed = execute(
        &repo,
        &librarian,
        discover(&repo, vec![spec(dir.path())]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    let job_id = failed.job_id.unwrap();
    std::fs::remove_file(&path).unwrap();
    let (plan, job) = resume_plan(&repo, &job_id).await.unwrap();
    assert_eq!(plan.failed(), 1);
    assert_eq!(plan.files[0].status, FileSyncStatus::Missing);
    let retained = execute(&repo, &librarian, plan, Some(job), flag())
        .await
        .unwrap();
    assert_eq!(retained.failed(), 1);
    assert_eq!(
        repo.get_processing_job(&job_id)
            .await
            .unwrap()
            .unwrap()
            .status,
        "failed"
    );
    std::fs::write(&path, "# Retry\nRecovered first import.").unwrap();
    *probe.reject.lock().unwrap() = None;
    let (plan, job) = resume_plan(&repo, &job_id).await.unwrap();
    assert_eq!(
        execute(&repo, &librarian, plan, Some(job), flag())
            .await
            .unwrap()
            .failed(),
        0
    );
}

#[tokio::test]
async fn rules_recursion_symlinks_and_renames_have_deterministic_plans() {
    let dir = tempfile::tempdir().unwrap();
    for folder in ["nested", "private", ".git"] {
        std::fs::create_dir(dir.path().join(folder)).unwrap();
    }
    for path in [
        "first.md",
        "nested/second.md",
        "private/secret.md",
        ".git/ignored.md",
        "other.txt",
    ] {
        std::fs::write(dir.path().join(path), "# Content\nUseful text.").unwrap();
    }
    #[cfg(unix)]
    {
        std::os::unix::fs::symlink(dir.path().join("first.md"), dir.path().join("alias.md"))
            .unwrap();
        std::os::unix::fs::symlink(dir.path(), dir.path().join("cycle")).unwrap();
    }
    let (repo, _, librarian) = fixture().await;
    let folder = spec(dir.path());
    let plan = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert_eq!(
        plan.files
            .iter()
            .filter(|file| file.status == FileSyncStatus::Created)
            .count(),
        2
    );
    #[cfg(unix)]
    assert_eq!(
        plan.files
            .iter()
            .filter(|file| file.status == FileSyncStatus::Ignored)
            .count(),
        2
    );
    execute(&repo, &librarian, plan, None, flag())
        .await
        .unwrap();
    std::fs::rename(dir.path().join("first.md"), dir.path().join("renamed.md")).unwrap();
    let renamed = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert!(renamed
        .files
        .iter()
        .any(|file| file.path.ends_with("first.md") && file.status == FileSyncStatus::Missing));
    assert!(renamed
        .files
        .iter()
        .any(|file| file.path.ends_with("renamed.md") && file.status == FileSyncStatus::Created));
    assert_eq!(renamed.failed(), 0);
    let mut shallow = folder;
    shallow.recursive = false;
    assert!(!discover(&repo, vec![shallow])
        .await
        .unwrap()
        .files
        .iter()
        .any(|file| file.path.contains("nested/")));
}

#[tokio::test]
async fn unavailable_root_and_incomplete_scan_never_infer_missing_sources() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("notes");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("note.md"), "# Note\nStored note.").unwrap();
    let (repo, _, librarian) = fixture().await;
    let folder = spec(&root);
    execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    std::fs::remove_dir_all(&root).unwrap();
    let unavailable = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert_eq!(unavailable.failed(), 1);
    assert!(!unavailable
        .files
        .iter()
        .any(|file| file.status == FileSyncStatus::Missing));
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("bad.md"), [0xff, 0xfe]).unwrap();
    let incomplete = discover(&repo, vec![folder]).await.unwrap();
    assert_eq!(incomplete.failed(), 1);
    assert!(!incomplete
        .files
        .iter()
        .any(|file| file.status == FileSyncStatus::Missing));
    assert_eq!(repo.list_sources().await.unwrap().len(), 1);
}

#[tokio::test]
async fn excluded_sources_remain_retained_and_prune_preserves_manual_legacy_provenance() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("note.md");
    std::fs::write(&path, "# Note\nImported content.").unwrap();
    let (repo, _, librarian) = fixture().await;
    let mut folder = spec(dir.path());
    execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    let source = repo.list_sources().await.unwrap().pop().unwrap();
    let manual = repo
        .create_note(Note::new("Manual annotation"))
        .await
        .unwrap();
    let legacy = repo
        .create_note(Note::new("Legacy source annotation").with_source(source.id.clone().unwrap()))
        .await
        .unwrap();
    repo.create_edge(
        manual.id.as_ref().unwrap(),
        legacy.id.as_ref().unwrap(),
        graphrag_core::EdgeType::RelatedTo,
        None,
    )
    .await
    .unwrap();
    std::fs::remove_file(&path).unwrap();
    folder.exclude.push("note.md".into());
    assert!(!discover(&repo, vec![folder])
        .await
        .unwrap()
        .files
        .iter()
        .any(|file| file.status == FileSyncStatus::Missing));
    let preview = repo.preview_source_delete(&source).await.unwrap();
    assert_eq!(preview.notes, 1);
    let deleted = repo.delete_source(&source).await.unwrap();
    assert_eq!(deleted.notes, 1);
    assert_eq!(repo.list_notes(100).await.unwrap().len(), 2);
    assert!(repo
        .get_visible_note(&record_id_to_string(legacy.id.as_ref().unwrap()))
        .await
        .unwrap()
        .is_some());
    assert_eq!(
        repo.get_note_edges(&record_id_to_string(manual.id.as_ref().unwrap()))
            .await
            .unwrap()
            .len(),
        1
    );
}

#[tokio::test]
async fn changed_file_after_plan_is_rejected_before_inference() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("changing.md");
    std::fs::write(&path, "# Initial\nPlanned content.").unwrap();
    let (repo, probe, librarian) = fixture().await;
    let plan = discover(&repo, vec![spec(dir.path())]).await.unwrap();
    std::fs::write(path, "# Later\nChanged after plan.").unwrap();
    let rejected = execute(&repo, &librarian, plan, None, flag())
        .await
        .unwrap();
    assert_eq!(rejected.failed(), 1);
    assert!(rejected.files[0]
        .error
        .as_ref()
        .unwrap()
        .contains("changed after discovery"));
    assert_eq!(probe.requests.load(Ordering::SeqCst), 0);
    assert!(repo.list_sources().await.unwrap().is_empty());
}

#[tokio::test]
async fn cancellation_has_a_durable_explicit_resume_without_importing_new_files() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("first.md"), "# First\nUseful content.").unwrap();
    let (repo, probe, librarian) = fixture().await;
    let cancelled = execute(
        &repo,
        &librarian,
        discover(&repo, vec![spec(dir.path())]).await.unwrap(),
        None,
        Arc::new(AtomicBool::new(true)),
    )
    .await
    .unwrap();
    assert!(cancelled.cancelled);
    assert_eq!(cancelled.files[0].status, FileSyncStatus::Pending);
    assert_eq!(probe.requests.load(Ordering::SeqCst), 0);
    let id = cancelled.job_id.unwrap();
    assert_eq!(
        repo.get_processing_job(&id).await.unwrap().unwrap().status,
        "cancelled"
    );
    assert!(librarian
        .resume_processing_job(&id)
        .await
        .unwrap_err()
        .to_string()
        .contains("sync --resume"));
    let (plan, job) = resume_plan(&repo, &id).await.unwrap();
    assert_eq!(
        execute(&repo, &librarian, plan, Some(job), flag())
            .await
            .unwrap()
            .failed(),
        0
    );
}

#[tokio::test]
async fn cancellation_after_failed_peer_reports_completed_work_and_resume_cleans_hidden_generations(
) {
    let dir = tempfile::tempdir().unwrap();
    for (name, content) in [
        ("a.md", "REJECT first peer"),
        ("b.md", "CANCEL successful peer"),
        ("c.md", "Pending last peer"),
    ] {
        std::fs::write(dir.path().join(name), format!("# Note\n{content}")).unwrap();
    }
    let (repo, probe, librarian) = fixture().await;
    let cancellation = flag();
    *probe.reject.lock().unwrap() = Some("REJECT".into());
    *probe.cancel_on.lock().unwrap() = Some(("CANCEL".into(), cancellation.clone()));
    let cancelled = execute(
        &repo,
        &librarian,
        discover(&repo, vec![spec(dir.path())]).await.unwrap(),
        None,
        cancellation,
    )
    .await
    .unwrap();
    assert!(cancelled.cancelled);
    assert_eq!(
        cancelled
            .files
            .iter()
            .map(|file| file.status)
            .collect::<Vec<_>>(),
        vec![
            FileSyncStatus::Failed,
            FileSyncStatus::Created,
            FileSyncStatus::Pending
        ]
    );
    let id = cancelled.job_id.unwrap();
    let job = repo.get_processing_job(&id).await.unwrap().unwrap();
    assert_eq!((job.completed_count, job.failed_count), (1, 1));
    assert!(job.checkpoint.unwrap().ends_with("b.md"));
    let source = repo
        .get_source(&normalize_file_uri(dir.path().join("b.md")).unwrap())
        .await
        .unwrap()
        .unwrap();
    // Model a stop after promotion but before old-generation cleanup.
    let stale = repo
        .create_note(
            Note::new("stale hidden generation")
                .with_source(source.id.unwrap())
                .with_source_generation(0),
        )
        .await
        .unwrap();
    *probe.reject.lock().unwrap() = None;
    *probe.cancel_on.lock().unwrap() = None;
    let requests = probe.requests.load(Ordering::SeqCst);
    let (plan, job) = resume_plan(&repo, &id).await.unwrap();
    let resumed = execute(&repo, &librarian, plan, Some(job), flag())
        .await
        .unwrap();
    assert_eq!(resumed.failed(), 0);
    assert_eq!(probe.requests.load(Ordering::SeqCst), requests + 2);
    assert_eq!(probe.health_checks.load(Ordering::SeqCst), 0);
    assert!(repo
        .get_note(&record_id_to_string(stale.id.as_ref().unwrap()))
        .await
        .unwrap()
        .is_none());
    assert_eq!(
        repo.get_processing_job(&id).await.unwrap().unwrap().status,
        "completed"
    );
}

#[cfg(unix)]
#[tokio::test]
async fn directory_symlink_replacement_retains_nested_sources_and_refuses_missing_checks() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("nested");
    let redirected = dir.path().join("elsewhere");
    std::fs::create_dir(&nested).unwrap();
    std::fs::create_dir(&redirected).unwrap();
    let path = nested.join("stored.md");
    std::fs::write(&path, "# Stored\nRetain searchable content.").unwrap();
    let (repo, _, librarian) = fixture().await;
    let folder = spec(dir.path());
    execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    std::fs::remove_dir_all(&nested).unwrap();
    std::os::unix::fs::symlink(&redirected, &nested).unwrap();
    let plan = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert_eq!(plan.failed(), 0);
    assert!(!plan
        .files
        .iter()
        .any(|file| file.status == FileSyncStatus::Missing));
    assert!(!source_is_missing(&folder, &folder.path.join("nested/stored.md")).unwrap());
    assert_eq!(repo.list_notes(100).await.unwrap().len(), 1);
}

#[cfg(unix)]
#[tokio::test]
async fn root_ancestor_symlink_redirection_is_unavailable_and_never_prunable() {
    let dir = tempfile::tempdir().unwrap();
    let ancestor = dir.path().join("parent");
    let root = ancestor.join("notes");
    let redirected = dir.path().join("replacement");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::create_dir_all(redirected.join("notes")).unwrap();
    std::fs::write(
        root.join("stored.md"),
        "# Stored\nRetain searchable content.",
    )
    .unwrap();
    let (repo, _, librarian) = fixture().await;
    let folder = spec(&root);
    execute(
        &repo,
        &librarian,
        discover(&repo, vec![folder.clone()]).await.unwrap(),
        None,
        flag(),
    )
    .await
    .unwrap();
    std::fs::remove_dir_all(&ancestor).unwrap();
    std::os::unix::fs::symlink(&redirected, &ancestor).unwrap();
    let plan = discover(&repo, vec![folder.clone()]).await.unwrap();
    assert_eq!(plan.failed(), 1);
    assert!(!plan
        .files
        .iter()
        .any(|file| file.status == FileSyncStatus::Missing));
    assert!(source_is_missing(&folder, &folder.path.join("stored.md")).is_err());
    assert_eq!(repo.list_notes(100).await.unwrap().len(), 1);
}
