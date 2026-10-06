//! Fictional in-memory walkthrough: diagnostics never bootstrap client storage or inference.
use async_trait::async_trait;
use graphrag_agents::{
    DeterministicEmbedder, Embedder, FixtureEntityExtractor, InferenceCapabilities,
    LibrarianRuntimeConfig, SearchAgent,
};
use graphrag_application::EmbeddedApplication;
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

const TOKEN: &str = "fictional-doctor-credential-1234567890123456789";
#[derive(Default)]
struct StoppedProvider {
    health_calls: AtomicUsize,
}
#[async_trait]
impl Embedder for StoppedProvider {
    async fn embed(&self, _: &str, _: bool) -> graphrag_agents::Result<Vec<f32>> {
        panic!("doctor must not embed")
    }
    async fn embed_batch(&self, _: &[String], _: bool) -> graphrag_agents::Result<Vec<Vec<f32>>> {
        panic!("doctor must not embed batch")
    }
    async fn health(&self) -> graphrag_agents::Result<bool> {
        self.health_calls.fetch_add(1, Ordering::SeqCst);
        Ok(false)
    }
    fn capabilities(&self) -> InferenceCapabilities {
        DeterministicEmbedder::default().capabilities()
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_walkthrough_distinguishes_unknown_stopped_auth_transport_and_partial_refresh() {
    let temp = tempfile::tempdir().unwrap();
    let credentials = temp.path().join("credentials.json");
    std::fs::write(
        &credentials,
        serde_json::to_vec(&CredentialFile {
            schema_version: 1,
            credentials: vec![Credential {
                instance_id: "fictional-reader".into(),
                token_sha256: format!("{:x}", Sha256::digest(TOKEN.as_bytes())),
                capabilities: vec![Capability::Read],
            }],
        })
        .unwrap(),
    )
    .unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&credentials, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let repo = Repository::new(init_memory().await.unwrap());
    let provider = Arc::new(StoppedProvider::default());
    let application = Arc::new(EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo, provider.clone()),
        provider.clone(),
        Arc::new(FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig::default(),
    ));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let listen = listener.local_addr().unwrap();
    let endpoint = format!("http://{listen}/mcp");
    let task = tokio::spawn(serve(
        listener,
        application,
        ServiceOptions {
            listen,
            credentials_file: credentials,
            ..Default::default()
        },
        Default::default(),
    ));
    let command = |server: &str, token: &str| {
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_graphrag"));
        command
            .env_clear()
            .env("HOME", temp.path())
            .env("PATH", "/usr/bin:/bin")
            .env("GRAPHRAG_TOKEN", token)
            .env("GRAPHRAG_CONFIG", temp.path().join("missing.toml"))
            .env("GRAPHRAG_DB_PATH", temp.path().join("must-not-exist"))
            .current_dir(temp.path())
            .args(["--server", server]);
        command
    };
    let output = command(&endpoint, TOKEN)
        .args(["doctor", "--format", "json"])
        .output()
        .await
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(report["transport"]["state"], "ready");
    assert_eq!(report["authorization"]["state"], "ready");
    assert_eq!(report["application"]["storage"]["state"], "ready");
    assert_eq!(report["application"]["embeddings"]["state"], "unknown");
    assert_eq!(report["application"]["backup"]["state"], "unknown");
    assert_eq!(report["service"]["jobs"]["readiness"]["state"], "forbidden");
    assert_eq!(provider.health_calls.load(Ordering::SeqCst), 0);
    assert!(!temp.path().join("must-not-exist").exists());

    let search = command(&endpoint, TOKEN)
        .args([
            "search",
            "--mode",
            "hybrid",
            "--scope",
            "notes",
            "--limit",
            "7",
            "--",
            "fictional atlas",
        ])
        .output()
        .await
        .unwrap();
    assert!(!search.status.success());
    let stderr = String::from_utf8_lossy(&search.stderr);
    assert!(stderr.contains("Explicit keyword retry:"));
    assert!(stderr.contains("--mode keyword --graph off --scope notes --limit 7"));
    assert!(!stderr.contains(TOKEN));
    let report = command(&endpoint, TOKEN)
        .args(["doctor", "--format", "json"])
        .output()
        .await
        .unwrap();
    let report: Value = serde_json::from_slice(&report.stdout).unwrap();
    assert_eq!(report["application"]["embeddings"]["state"], "unavailable");
    assert_eq!(provider.health_calls.load(Ordering::SeqCst), 1);

    let status = temp.path().join("refresh.json");
    let mut evidence = json!({"schema_version":1,"endpoint_sha256":format!("{:x}", Sha256::digest(endpoint.as_bytes())),"instance_id":"fictional-reader","collection_id":"fictional-memory","status":"partial","last_attempt_at":"2026-10-05T00:00:00Z","last_success_at":null,"counts":{"created":1,"changed":0,"unchanged":0,"failed":1,"missing":0},"pending_parts":1,"error_code":"provider_unavailable","retry":{"action":"resume"}});
    std::fs::write(&status, serde_json::to_vec(&evidence).unwrap()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&status, std::fs::Permissions::from_mode(0o600)).unwrap();
    }
    let partial = command(&endpoint, TOKEN)
        .args([
            "doctor",
            "--format",
            "json",
            "--refresh-status-file",
            status.to_str().unwrap(),
        ])
        .output()
        .await
        .unwrap();
    let report: Value = serde_json::from_slice(&partial.stdout).unwrap();
    assert_eq!(report["application"]["sources"]["state"], "partial");
    assert_eq!(report["client_refresh_evidence"]["counts"]["failed"], 1);
    evidence["instance_id"] = json!("another-principal");
    std::fs::write(&status, serde_json::to_vec(&evidence).unwrap()).unwrap();
    let wrong = command(&endpoint, TOKEN)
        .args([
            "doctor",
            "--format",
            "json",
            "--refresh-status-file",
            status.to_str().unwrap(),
        ])
        .output()
        .await
        .unwrap();
    let report: Value = serde_json::from_slice(&wrong.stdout).unwrap();
    assert_eq!(report["application"]["sources"]["state"], "unknown");
    assert!(report["client_refresh_evidence"].is_null());

    let denied = command(&endpoint, "fictional-wrong-credential-1234567890123456789")
        .args(["doctor", "--format", "json"])
        .output()
        .await
        .unwrap();
    assert_eq!(denied.status.code(), Some(2));
    let report: Value = serde_json::from_slice(&denied.stdout).unwrap();
    assert_eq!(report["transport"]["state"], "ready");
    assert_eq!(report["authorization"]["state"], "forbidden");
    assert!(report["service"].is_null());
    let unreachable = tokio::time::timeout(
        Duration::from_secs(10),
        command("http://127.0.0.1:0/mcp", TOKEN)
            .args(["doctor", "--format", "json"])
            .output(),
    )
    .await
    .unwrap()
    .unwrap();
    let report: Value = serde_json::from_slice(&unreachable.stdout).unwrap();
    assert_eq!(report["transport"]["state"], "unavailable");
    assert_eq!(report["authorization"]["state"], "unknown");
    assert!(!String::from_utf8_lossy(&unreachable.stdout).contains(TOKEN));
    assert!(!temp.path().join("must-not-exist").exists());
    task.abort();
}
