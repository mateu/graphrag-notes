//! Real CLI/HTTP/repository checks for revision guards and draft cleanup.
use graphrag_agents::{
    DeterministicEmbedder, Embedder, FixtureEntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig, SearchAgent, SharedEmbedder,
};
use graphrag_application::EmbeddedApplication;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};
use std::time::Duration;
use tokio::sync::Notify;

struct Fixture {
    temp: tempfile::TempDir,
    endpoint: String,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}
const TOKEN: &str = "synthetic-cli-http-mutation-token-1234567890";
impl Fixture {
    async fn new(repo: &Repository) -> Self {
        Self::with_embedder(repo, Arc::new(DeterministicEmbedder::default())).await
    }
    async fn with_embedder(repo: &Repository, embedder: SharedEmbedder) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let credentials_file = temp.path().join("credentials.json");
        let policy = CredentialFile {
            schema_version: 1,
            credentials: vec![Credential {
                instance_id: "cli-synthetic".into(),
                token_sha256: format!("{:x}", Sha256::digest(TOKEN.as_bytes())),
                capabilities: vec![
                    Capability::Read,
                    Capability::Edit,
                    Capability::Delete,
                    Capability::Capture,
                ],
            }],
        };
        std::fs::write(&credentials_file, serde_json::to_vec(&policy).unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&credentials_file, std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        let application = Arc::new(EmbeddedApplication::new(
            repo.clone(),
            SearchAgent::new(repo.clone(), embedder.clone()),
            embedder,
            Arc::new(FixtureEntityExtractor::default()),
            LibrarianRuntimeConfig {
                skip_entity_extraction: true,
                ..Default::default()
            },
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let listen = listener.local_addr().unwrap();
        let options = ServiceOptions {
            listen,
            credentials_file,
            ..Default::default()
        };
        let task = tokio::spawn(serve(listener, application, options, Default::default()));
        Self {
            temp,
            endpoint: format!("http://{listen}/mcp"),
            task,
        }
    }
    fn command(&self, args: &[&str]) -> tokio::process::Command {
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_graphrag"));
        command
            .current_dir(self.temp.path())
            .env("GRAPHRAG_TOKEN", TOKEN)
            .env(
                "GRAPHRAG_CONFIG",
                self.temp.path().join("missing-config.toml"),
            )
            .env(
                "GRAPHRAG_DB_PATH",
                self.temp.path().join("must-not-create-client-db"),
            )
            .args(["--server", &self.endpoint])
            .args(args);
        command
    }
    async fn cli(&self, args: &[&str]) -> std::process::Output {
        self.command(args).output().await.unwrap()
    }
    async fn success(&self, args: &[&str]) -> Value {
        let output = self.cli(args).await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let value: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(value["success"], true);
        assert_eq!(value["data"]["error"], Value::Null);
        value["data"]["data"].clone()
    }
}

#[cfg(unix)]
struct BlockedEmbedder {
    started: Notify,
    released: Notify,
    calls: AtomicUsize,
}
#[cfg(unix)]
#[async_trait::async_trait]
impl Embedder for BlockedEmbedder {
    async fn embed(&self, text: &str, query: bool) -> graphrag_agents::Result<Vec<f32>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.released.notified().await;
        DeterministicEmbedder::default().embed(text, query).await
    }
    async fn embed_batch(
        &self,
        texts: &[String],
        query: bool,
    ) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        self.released.notified().await;
        DeterministicEmbedder::default()
            .embed_batch(texts, query)
            .await
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        Ok(true)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lost_ack_edit_and_capture_retries_adopt_original_private_draft_only_after_verified_replay()
{
    for operation in ["edit", "capture"] {
        let repo = Repository::new(init_memory().await.unwrap());
        let note = repo
            .create_note(Note::new("original manual content"))
            .await
            .unwrap();
        let id = record_id_to_string(note.id.as_ref().unwrap());
        let provider = Arc::new(BlockedEmbedder {
            started: Notify::new(),
            released: Notify::new(),
            calls: AtomicUsize::new(0),
        });
        let fixture = Fixture::with_embedder(&repo, provider.clone()).await;
        let snapshot = fixture
            .success(&["notes", "show", &id, "--format", "json"])
            .await;
        let revision = snapshot["revision"].as_str().unwrap();
        let input = fixture.temp.path().join("ordinary-input.md");
        let drafts = fixture.temp.path().join("private drafts");
        let body = "valuable recovery body after an actual lost acknowledgment";
        std::fs::write(&input, body).unwrap();
        let request = format!("lost-{operation}");
        let mut args = vec!["--request-id", &request];
        if operation == "edit" {
            args.extend(["--expected-revision", revision, "notes", "edit", &id]);
        } else {
            args.push("capture");
        }
        args.extend([
            "--content-file",
            input.to_str().unwrap(),
            "--draft-dir",
            drafts.to_str().unwrap(),
            "--format",
            "json",
        ]);
        let mut command = fixture.command(&args);
        command
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        let mut child = command.spawn().unwrap();
        tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
            .await
            .unwrap();
        let retained = std::fs::read_dir(&drafts)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path();
        assert_eq!(std::fs::read_to_string(&retained).unwrap(), body);
        child.kill().await.unwrap();
        provider.released.notify_one();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let committed = if operation == "edit" {
                    repo.get_note(&id).await.unwrap().unwrap().content == body
                } else {
                    repo.get_stats().await.unwrap().note_count == 2
                };
                if committed {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let calls = provider.calls.load(Ordering::SeqCst);
        let mut retry = vec!["--recover-draft", "--request-id", &request];
        if operation == "edit" {
            retry.extend(["--expected-revision", revision, "notes", "edit", &id]);
        } else {
            retry.push("capture");
        }
        retry.extend([
            "--content-file",
            retained.to_str().unwrap(),
            "--draft-dir",
            drafts.to_str().unwrap(),
            "--format",
            "json",
        ]);
        let replay = fixture.success(&retry).await;
        assert_eq!(replay["replayed"], true, "{operation}");
        assert_eq!(provider.calls.load(Ordering::SeqCst), calls);
        assert!(!retained.exists());
        assert_eq!(std::fs::read_dir(&drafts).unwrap().count(), 0);
        assert_eq!(std::fs::read_to_string(&input).unwrap(), body);
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cli_edit_guards_revision_replays_and_keeps_only_failed_drafts() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo
        .create_note(Note::new("original manual content"))
        .await
        .unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let fixture = Fixture::new(&repo).await;
    let snapshot = fixture
        .success(&["notes", "show", &id, "--format", "json"])
        .await;
    let revision = snapshot["revision"].as_str().unwrap();
    let edited = fixture
        .success(&[
            "--request-id",
            "metadata-001",
            "--expected-revision",
            revision,
            "notes",
            "edit",
            &id,
            "--title",
            "reviewed title",
            "--format",
            "json",
        ])
        .await;
    assert_eq!(edited["outcome"]["actor"], "mcp:cli-synthetic");
    let replay = fixture
        .success(&[
            "--request-id",
            "metadata-001",
            "--expected-revision",
            revision,
            "notes",
            "edit",
            &id,
            "--title",
            "reviewed title",
            "--format",
            "json",
        ])
        .await;
    assert_eq!(replay["replayed"], true);
    assert_eq!(replay["outcome"], edited["outcome"]);
    let file = fixture.temp.path().join("replacement.md");
    let drafts = fixture.temp.path().join("private drafts");
    std::fs::write(&file, "valuable replacement content").unwrap();
    let result = fixture
        .cli(&[
            "--request-id",
            "content-001",
            "--expected-revision",
            revision,
            "notes",
            "edit",
            &id,
            "--content-file",
            file.to_str().unwrap(),
            "--draft-dir",
            drafts.to_str().unwrap(),
            "--format",
            "json",
        ])
        .await;
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("revision_conflict"));
    assert_eq!(std::fs::read_dir(&drafts).unwrap().count(), 1);
    assert_eq!(
        repo.get_note(id.split_once(':').unwrap().1)
            .await
            .unwrap()
            .unwrap()
            .content,
        "original manual content"
    );
    let current = fixture
        .success(&["notes", "show", &id, "--format", "json"])
        .await;
    let fresh = current["revision"].as_str().unwrap();
    fixture
        .success(&[
            "--request-id",
            "content-002",
            "--expected-revision",
            fresh,
            "notes",
            "edit",
            &id,
            "--content-file",
            file.to_str().unwrap(),
            "--draft-dir",
            drafts.to_str().unwrap(),
            "--format",
            "json",
        ])
        .await;
    // The old conflicting draft remains, while the successful new draft is removed.
    assert_eq!(std::fs::read_dir(&drafts).unwrap().count(), 1);
    assert!(!fixture
        .temp
        .path()
        .join("must-not-create-client-db")
        .exists());
}

#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unchanged_remote_editor_applies_explicit_metadata_without_losing_recovery_contract() {
    use std::os::unix::fs::PermissionsExt;
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo.create_note(Note::new("unchanged body")).await.unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let fixture = Fixture::new(&repo).await;
    let editor = fixture.temp.path().join("unchanged-editor");
    std::fs::write(&editor, "#!/bin/sh\nexit 0\n").unwrap();
    std::fs::set_permissions(&editor, std::fs::Permissions::from_mode(0o700)).unwrap();
    let snapshot = fixture
        .success(&["notes", "show", &id, "--format", "json"])
        .await;
    let drafts = fixture.temp.path().join("drafts");
    fixture
        .success(&[
            "--expected-revision",
            snapshot["revision"].as_str().unwrap(),
            "--request-id",
            "editor-metadata-001",
            "notes",
            "edit",
            &id,
            "--editor",
            "--editor-command",
            editor.to_str().unwrap(),
            "--draft-dir",
            drafts.to_str().unwrap(),
            "--title",
            "explicit metadata",
            "--format",
            "json",
        ])
        .await;
    let current = fixture
        .success(&["notes", "show", &id, "--format", "json"])
        .await;
    assert_eq!(current["title"], json!("explicit metadata"));
    assert_eq!(current["content"], json!("unchanged body"));
    assert_eq!(std::fs::read_dir(drafts).unwrap().count(), 0);
}
