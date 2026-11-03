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
    note_with(repo, file, vector(cosine, axis), marker).await
}

async fn note_with(repo: &Repository, file: &str, embedding: Vec<f32>, marker: &str) -> String {
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
                .with_embedding(embedding)
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

/// Unit vector with `cosine` on the `main` axis and the rest on `private`.
fn topic(main: usize, cosine: f32, private: usize) -> Vec<f32> {
    let mut vector = vec![0.0; 1024];
    vector[main] = cosine;
    vector[private] = (1.0 - cosine * cosine).max(0.0).sqrt();
    vector
}

/// Two anchors on orthogonal topics, in record-ID order: the first has one
/// cross-source counterpart sharing a rare marker, the second has two.
async fn two_windows(repo: &Repository) -> [String; 2] {
    let mut anchors = Vec::new();
    for (axis, marker) in [(100, "Quill relay"), (200, "Harbor beacon")] {
        let file = format!("anchor-{axis}.md");
        let id = note_with(repo, &file, topic(axis, 1.0, axis + 1), marker).await;
        anchors.push((id, axis, marker));
    }
    anchors.sort();
    for (rank, (_, axis, marker)) in anchors.iter().enumerate() {
        for (offset, cosine) in [0.95_f32, 0.92].into_iter().take(rank + 1).enumerate() {
            let file = format!("counterpart-{axis}-{offset}.md");
            note_with(repo, &file, topic(*axis, cosine, axis + 2 + offset), marker).await;
        }
    }
    [anchors[0].0.clone(), anchors[1].0.clone()]
}

fn paged_args(
    anchors: &[String; 2],
    dry_run: bool,
    after: Option<&str>,
    json: bool,
) -> Vec<String> {
    let mut args = [
        "garden",
        "scan",
        "--max-candidates",
        "1",
        "--max-endpoints",
        "1",
        "--note",
        anchors[0].as_str(),
        "--note",
        anchors[1].as_str(),
    ]
    .map(str::to_owned)
    .to_vec();
    if json {
        args.extend(["--format".to_owned(), "json".to_owned()]);
    }
    if dry_run {
        args.push("--dry-run".to_owned());
    }
    if let Some(after) = after {
        args.extend(["--after".to_owned(), after.to_owned()]);
    }
    args
}

/// Page the way a client follows the receipt contract: continue from
/// `next_after_note_id` (`null` repeats from the start) until the scope is
/// exhausted or a dry run reports `dry_run_cut`.
async fn page_until_done(fixture: &Fixture, anchors: &[String; 2], dry_run: bool) -> Vec<Value> {
    let mut after: Option<String> = None;
    let mut receipts = Vec::new();
    for _ in 0..6 {
        let args = paged_args(anchors, dry_run, after.as_deref(), true);
        let args = args.iter().map(String::as_str).collect::<Vec<_>>();
        let output = fixture.cli(GARDEN_TOKEN, &args).await;
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
        let receipt = envelope["data"]["data"].clone();
        let done = receipt["scope_exhausted"] == true || !receipt["dry_run_cut"].is_null();
        after = receipt["next_after_note_id"].as_str().map(str::to_owned);
        receipts.push(receipt);
        if done {
            return receipts;
        }
    }
    panic!("paging did not terminate: {receipts:#?}");
}

#[tokio::test]
async fn remote_garden_scan_paging_terminates_for_dry_and_persisting_runs() {
    let repo = Repository::new(init_memory().await.unwrap());
    let anchors = two_windows(&repo).await;
    let fixture = Fixture::new(&repo).await;

    // Dry run: the first window advances; the second cuts a candidate, so
    // paging stops there with the cut count instead of a cursor.
    let dry = page_until_done(&fixture, &anchors, true).await;
    assert_eq!(dry.len(), 2, "{dry:#?}");
    assert_eq!(dry[0]["next_after_note_id"], anchors[0].as_str());
    assert_eq!(dry[0]["dry_run_cut"], Value::Null);
    assert_eq!(dry[1]["stop_reason"], "candidate_budget");
    assert_eq!(dry[1]["next_after_note_id"], Value::Null);
    assert_eq!(dry[1]["scope_exhausted"], false);
    assert_eq!(dry[1]["dry_run_cut"]["candidates_cut"], 1);
    assert_eq!(proposal_count(&repo).await, 0);

    // Human output for that page explains the stop instead of a cursor.
    let args = paged_args(&anchors, true, Some(anchors[0].as_str()), false);
    let args = args.iter().map(String::as_str).collect::<Vec<_>>();
    let human = fixture.cli(GARDEN_TOKEN, &args).await;
    assert!(
        human.status.success(),
        "{}",
        String::from_utf8_lossy(&human.stderr)
    );
    let human = String::from_utf8_lossy(&human.stdout);
    assert!(
        human.contains("Dry run cut 1 candidate(s) at --max-candidates"),
        "{human}"
    );
    assert!(!human.contains("Resume with"), "{human}");

    // Persisting runs page as before: a cut repeats from the incoming cursor
    // and the recorded pending proposal lets the repeat progress.
    let persisted = page_until_done(&fixture, &anchors, false).await;
    assert_eq!(persisted.len(), 3, "{persisted:#?}");
    assert_eq!(persisted[0]["next_after_note_id"], anchors[0].as_str());
    assert_eq!(persisted[1]["rejections"]["over_budget"], 1);
    assert_eq!(persisted[1]["next_after_note_id"], anchors[0].as_str());
    assert_eq!(persisted[2]["rejections"]["already_proposed"], 1);
    assert_eq!(persisted[2]["scope_exhausted"], true);
    assert!(persisted
        .iter()
        .all(|receipt| receipt["dry_run_cut"].is_null()));
    assert_eq!(proposal_count(&repo).await, 3);
    fixture.task.abort();
}
