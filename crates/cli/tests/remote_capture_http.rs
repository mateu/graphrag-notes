//! A lost remote capture acknowledgement must retain one draft until receipt replay.
use graphrag_agents::{
    DeterministicEmbedder, Embedder, FixtureEntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig, SearchAgent, SharedEmbedder,
};
use graphrag_application::EmbeddedApplication;
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::Value;
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
    async fn with_embedder(repo: &Repository, embedder: SharedEmbedder) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let credentials_file = temp.path().join("credentials.json");
        let policy = CredentialFile {
            schema_version: 1,
            credentials: vec![Credential {
                instance_id: "cli-synthetic".into(),
                token_sha256: format!("{:x}", Sha256::digest(TOKEN.as_bytes())),
                capabilities: vec![Capability::Read, Capability::Capture],
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
            .env_clear()
            .current_dir(self.temp.path())
            .env("HOME", self.temp.path())
            .env("XDG_CONFIG_HOME", self.temp.path())
            .env("PATH", "/usr/bin:/bin")
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
async fn lost_ack_capture_reuses_original_private_draft_until_verified_receipt_replay() {
    let repo = Repository::new(init_memory().await.unwrap());
    let provider = Arc::new(BlockedEmbedder {
        started: Notify::new(),
        released: Notify::new(),
        calls: AtomicUsize::new(0),
    });
    let fixture = Fixture::with_embedder(&repo, provider.clone()).await;
    let input = fixture.temp.path().join("ordinary-input.md");
    let relative_drafts = "relative private drafts";
    let drafts = fixture.temp.path().join(relative_drafts);
    let body = "valuable recovery body after an actual lost acknowledgment";
    std::fs::write(&input, body).unwrap();
    let request = "-lost-capture";
    let request_option = format!("--request-id={request}");
    let mut args = vec![request_option.as_str()];
    args.push("capture");
    args.extend([
        "--content-file",
        input.to_str().unwrap(),
        "--draft-dir",
        relative_drafts,
        "--format",
        "json",
    ]);
    let mut command = fixture.command(&args);
    command
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    command.kill_on_drop(true);
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
    child.wait().await.unwrap();
    provider.released.notify_one();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            let committed = repo.get_stats().await.unwrap().note_count == 1;
            if committed {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let calls = provider.calls.load(Ordering::SeqCst);
    let mut retry = vec!["--recover-draft", request_option.as_str()];
    retry.push("capture");
    retry.extend([
        "--content-file",
        retained.to_str().unwrap(),
        "--draft-dir",
        relative_drafts,
        "--format",
        "json",
    ]);
    // Obtain the actual copied command from another failed attempt, then
    // execute it with the synthetic token in an isolated environment.
    let mut failed = fixture.command(&retry);
    failed.env("GRAPHRAG_TOKEN", "").kill_on_drop(true);
    let failed_retry = tokio::time::timeout(Duration::from_secs(5), failed.output())
        .await
        .unwrap()
        .unwrap();
    assert!(!failed_retry.status.success());
    assert!(retained.exists());
    assert_eq!(std::fs::read_dir(&drafts).unwrap().count(), 1);
    let stderr = String::from_utf8(failed_retry.stderr).unwrap();
    let copied = stderr
        .lines()
        .find_map(|line| line.strip_prefix("Recover: "))
        .expect("failed retry must retain its copied command");
    assert!(copied.contains(&format!("--request-id='{request}'")));
    assert!(copied.contains(&format!(
        "--draft-dir '{}'",
        std::fs::canonicalize(&drafts).unwrap().display()
    )));
    let executable = env!("CARGO_BIN_EXE_graphrag").replace('\'', "'\\''");
    let copied = format!(
        "'{executable}' {}",
        copied.strip_prefix("graphrag ").unwrap()
    );
    let mut replay_command = tokio::process::Command::new("/bin/sh");
    let replay_cwd = fixture.temp.path().join("different working directory");
    std::fs::create_dir(&replay_cwd).unwrap();
    replay_command
        .env_clear()
        .current_dir(&replay_cwd)
        .env("HOME", fixture.temp.path())
        .env("XDG_CONFIG_HOME", fixture.temp.path())
        .env("PATH", "/usr/bin:/bin")
        .env("GRAPHRAG_TOKEN", TOKEN)
        .env(
            "GRAPHRAG_CONFIG",
            fixture.temp.path().join("missing-config.toml"),
        )
        .env(
            "GRAPHRAG_DB_PATH",
            fixture.temp.path().join("must-not-create-client-db"),
        )
        .args(["-c", &copied])
        .kill_on_drop(true);
    let output = tokio::time::timeout(Duration::from_secs(10), replay_command.output())
        .await
        .unwrap()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["success"], true);
    let replay = &envelope["data"]["data"];
    assert_eq!(replay["replayed"], true);
    assert_eq!(repo.get_stats().await.unwrap().note_count, 1);
    assert_eq!(provider.calls.load(Ordering::SeqCst), calls);
    assert!(!retained.exists());
    assert_eq!(std::fs::read_dir(&drafts).unwrap().count(), 0);
    assert_eq!(std::fs::read_to_string(&input).unwrap(), body);
    assert!(!replay_cwd.join(relative_drafts).exists());
}
impl Drop for Fixture {
    fn drop(&mut self) {
        self.task.abort();
    }
}
