//! Real CLI/HTTP checks for the bounded remote candidate scan (fictional fixtures).
use graphrag_agents::{
    DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig, SearchAgent,
    SharedEmbedder,
};
use graphrag_application::EmbeddedApplication;
use graphrag_core::{record_id_to_string, Entity, EntityType, Note, Source, SourceType};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;

const GARDEN_TOKEN: &str = "synthetic-cli-garden-token-1234567890abcdef";
const READ_TOKEN: &str = "synthetic-cli-reader-token-1234567890abcdef";

struct Fixture {
    temp: tempfile::TempDir,
    endpoint: String,
    task: tokio::task::JoinHandle<Result<(), graphrag_service::ServiceError>>,
}

impl Fixture {
    async fn new(repo: &Repository) -> Self {
        let temp = tempfile::tempdir().unwrap();
        let credentials_file = temp.path().join("credentials.json");
        let credential = |name: &str, token: &str, capability| Credential {
            instance_id: name.into(),
            token_sha256: format!("{:x}", Sha256::digest(token.as_bytes())),
            capabilities: vec![capability],
        };
        let policy = CredentialFile {
            schema_version: 1,
            credentials: vec![
                credential("cli-gardener", GARDEN_TOKEN, Capability::Garden),
                credential("cli-reader", READ_TOKEN, Capability::Read),
            ],
        };
        std::fs::write(&credentials_file, serde_json::to_vec(&policy).unwrap()).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&credentials_file, std::fs::Permissions::from_mode(0o600))
                .unwrap();
        }
        let embedder: SharedEmbedder = Arc::new(DeterministicEmbedder::default());
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

    async fn cli(&self, token: &str, args: &[&str]) -> std::process::Output {
        tokio::process::Command::new(env!("CARGO_BIN_EXE_graphrag"))
            .env_clear()
            .current_dir(self.temp.path())
            .env("HOME", self.temp.path())
            .env("XDG_CONFIG_HOME", self.temp.path())
            .env("PATH", "/usr/bin:/bin")
            .env("GRAPHRAG_TOKEN", token)
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
}

fn vector(cosine: f32, axis: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[0] = cosine;
    vector[axis] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

async fn note(repo: &Repository, file: &str, cosine: f32, axis: usize, marker: &str) -> String {
    let source = repo
        .create_source(
            Source::from_file(format!("/fixtures/vault/{file}"), SourceType::Markdown).unwrap(),
        )
        .await
        .unwrap()
        .id
        .unwrap();
    let id = repo
        .create_note(
            Note::new(format!("Fictional {file} entry about the {marker}"))
                .with_embedding(vector(cosine, axis))
                .with_source(source),
        )
        .await
        .unwrap()
        .id
        .unwrap();
    let mut entity = Entity::new(marker, EntityType::Project);
    entity.metadata = json!({});
    repo.upsert_entities_and_link_note(&id, vec![entity])
        .await
        .unwrap();
    record_id_to_string(&id)
}

async fn proposal_count(repo: &Repository) -> usize {
    repo.portable_records_page("proposed_edge", 0, 100)
        .await
        .unwrap()
        .len()
}

#[tokio::test]
async fn remote_garden_scan_requires_explicit_budgets_and_echoes_its_receipt() {
    let repo = Repository::new(init_memory().await.unwrap());
    let anchor = note(&repo, "alpha.md", 1.0, 1, "Lantern relay").await;
    note(&repo, "beta.md", 0.93, 2, "Lantern relay").await;
    let fixture = Fixture::new(&repo).await;

    let missing = fixture
        .cli(GARDEN_TOKEN, &["garden", "scan", "--dry-run"])
        .await;
    assert!(!missing.status.success());
    assert!(String::from_utf8_lossy(&missing.stderr).contains("--max-candidates"));

    let scan = [
        "garden",
        "scan",
        "--max-candidates",
        "2",
        "--note",
        anchor.as_str(),
        "--dry-run",
        "--format",
        "json",
    ];
    let denied = fixture.cli(READ_TOKEN, &scan).await;
    assert!(!denied.status.success());
    assert!(String::from_utf8_lossy(&denied.stderr).contains("forbidden"));

    let preview = fixture.cli(GARDEN_TOKEN, &scan).await;
    assert!(
        preview.status.success(),
        "{}",
        String::from_utf8_lossy(&preview.stderr)
    );
    let envelope: Value = serde_json::from_slice(&preview.stdout).unwrap();
    let receipt = &envelope["data"]["data"];
    assert_eq!(receipt["actor"], "mcp:cli-gardener");
    assert_eq!(
        receipt["limits"],
        json!({"max_candidates": 2, "max_endpoints": 25, "deadline_ms": 10000})
    );
    assert_eq!(receipt["dry_run"], true);
    assert_eq!(receipt["candidates"].as_array().unwrap().len(), 1);
    assert_eq!(receipt["candidates"][0]["proposal"], Value::Null);
    assert_eq!(proposal_count(&repo).await, 0);

    let persisted = fixture
        .cli(
            GARDEN_TOKEN,
            &[
                "garden",
                "scan",
                "--max-candidates",
                "2",
                "--note",
                anchor.as_str(),
            ],
        )
        .await;
    assert!(
        persisted.status.success(),
        "{}",
        String::from_utf8_lossy(&persisted.stderr)
    );
    let human = String::from_utf8_lossy(&persisted.stdout);
    assert!(
        human.contains("Candidate scan recorded 1 pending proposal(s)"),
        "{human}"
    );
    assert!(human.contains("max_candidates 2, max_endpoints 25, deadline 10000 ms"));
    assert!(human.contains("lantern relay (2 notes)"), "{human}");
    let stored = repo
        .portable_records_page("proposed_edge", 0, 100)
        .await
        .unwrap();
    assert_eq!(stored.len(), 1);
    assert_eq!(stored[0]["status"], "pending");
    assert_eq!(stored[0]["generator"], "gardener-candidate");
    assert!(repo
        .portable_records_page("related_to", 0, 100)
        .await
        .unwrap()
        .is_empty());
    fixture.task.abort();
}
