//! Real CLI/HTTP/repository checks for revision guards and draft cleanup.
use graphrag_agents::{
    DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::EmbeddedApplication;
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

struct Fixture {
    temp: tempfile::TempDir,
    endpoint: String,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}
const TOKEN: &str = "synthetic-cli-http-mutation-token-1234567890";
impl Fixture {
    async fn new(repo: &Repository) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let credentials_file = temp.path().join("credentials.json");
        let policy = CredentialFile {
            schema_version: 1,
            credentials: vec![Credential {
                instance_id: "cli-synthetic".into(),
                token_sha256: format!("{:x}", Sha256::digest(TOKEN.as_bytes())),
                capabilities: vec![Capability::Read, Capability::Edit, Capability::Delete],
            }],
        };
        std::fs::write(&credentials_file, serde_json::to_vec(&policy).unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&credentials_file, std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        let embedder = Arc::new(DeterministicEmbedder::default());
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
    async fn cli(&self, args: &[&str]) -> std::process::Output {
        tokio::process::Command::new(env!("CARGO_BIN_EXE_graphrag"))
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
            .args(args)
            .output()
            .await
            .unwrap()
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
