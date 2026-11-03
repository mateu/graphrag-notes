//! `graphrag synthesize`: disabled by default locally, and a verified,
//! receipted round trip through a server's `synthesize_notes` tool. Fictional
//! notes and a fixture provider only; no live model is contacted.
use assert_cmd::Command;
use graphrag_agents::alchemist::{CitationPolicy, FixtureSynthesisGenerator, SynthesisBudgets};
use graphrag_agents::{
    DeterministicEmbedder, FixtureEntityExtractor, LibrarianRuntimeConfig, SearchAgent,
    SharedEmbedder,
};
use graphrag_application::{AlchemistRuntime, EmbeddedApplication};
use graphrag_core::{record_id_to_string, Note};
use graphrag_db::{init_memory, Repository};
use graphrag_service::{serve, Capability, Credential, CredentialFile, ServiceOptions};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;

const SYNTHESIST: &str = "synthetic-cli-alchemist-token-1234567890abcdef";
const READER: &str = "synthetic-cli-reader-only-token-1234567890abcd";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_synthesis_renders_verified_claims_and_echoes_the_receipt() {
    let repo = Repository::new(init_memory().await.unwrap());
    let note = repo
        .create_note(Note::new(
            "Fictional orchard log: the cider press was cleaned on Friday evening.",
        ))
        .await
        .unwrap();
    let id = record_id_to_string(note.id.as_ref().unwrap());
    let draft = json!({
        "claims": [{"text": "The press was cleaned.", "citations": [
            {"chunk": "C1", "quote": "the cider press was cleaned on Friday"}
        ]}],
        "conflicts": [],
        "open_questions": []
    });
    let embedder: SharedEmbedder = Arc::new(DeterministicEmbedder::default());
    let application = EmbeddedApplication::new(
        repo.clone(),
        SearchAgent::new(repo.clone(), embedder.clone()),
        embedder,
        Arc::new(FixtureEntityExtractor::default()),
        LibrarianRuntimeConfig {
            skip_entity_extraction: true,
            ..Default::default()
        },
    )
    .with_alchemist(AlchemistRuntime {
        ceilings: SynthesisBudgets::default(),
        citation_policy: CitationPolicy::Drop,
        generator: Arc::new(FixtureSynthesisGenerator::fixed(draft.to_string())),
    });
    let temp = tempfile::tempdir().unwrap();
    let credentials_file = temp.path().join("credentials.json");
    let policy = CredentialFile {
        schema_version: 1,
        credentials: [
            (
                "cli-synthesist",
                SYNTHESIST,
                vec![Capability::Read, Capability::Alchemist],
            ),
            ("cli-reader", READER, vec![Capability::Read]),
        ]
        .into_iter()
        .map(|(instance, token, capabilities)| Credential {
            instance_id: instance.into(),
            token_sha256: format!("{:x}", Sha256::digest(token.as_bytes())),
            capabilities,
        })
        .collect(),
    };
    std::fs::write(&credentials_file, serde_json::to_vec(&policy).unwrap()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&credentials_file, std::fs::Permissions::from_mode(0o600))
            .unwrap();
    }
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let listen = listener.local_addr().unwrap();
    let options = ServiceOptions {
        listen,
        credentials_file,
        ..Default::default()
    };
    let task = tokio::spawn(serve(
        listener,
        Arc::new(application),
        options,
        Default::default(),
    ));
    let endpoint = format!("http://{listen}/mcp");
    let client_db = temp.path().join("must-not-create-client-db");
    let run = |token: &str, args: &[&str]| {
        let mut command = tokio::process::Command::new(env!("CARGO_BIN_EXE_graphrag"));
        command
            .env_clear()
            .current_dir(temp.path())
            .env("HOME", temp.path())
            .env("XDG_CONFIG_HOME", temp.path())
            .env("PATH", "/usr/bin:/bin")
            .env("GRAPHRAG_TOKEN", token)
            .env("GRAPHRAG_CONFIG", temp.path().join("missing-config.toml"))
            .env("GRAPHRAG_DB_PATH", &client_db)
            .args(["--server", &endpoint])
            .args(args);
        command
    };
    let output = tokio::time::timeout(
        Duration::from_secs(20),
        run(
            SYNTHESIST,
            &[
                "synthesize",
                "--seed",
                &id,
                "--max-chunks",
                "2",
                "--format",
                "json",
            ],
        )
        .output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    let data = &envelope["data"];
    assert_eq!(
        data["synthesis"]["claims"][0]["citations"][0]["chunk_id"],
        id
    );
    assert_eq!(data["receipt"]["budgets"]["max_chunks"], 2);
    assert_eq!(data["receipt"]["policy"]["writes"], "none");
    assert_eq!(data["receipt"]["policy"]["citation_policy"], "drop");

    let output = tokio::time::timeout(
        Duration::from_secs(20),
        run(SYNTHESIST, &["synthesize", "--seed", &id]).output(),
    )
    .await
    .unwrap()
    .unwrap();
    let human = String::from_utf8(output.stdout).unwrap();
    assert!(
        human.contains("read-only; not stored; not evidence"),
        "{human}"
    );
    assert!(human.contains("\"the cider press was cleaned on Friday\""));
    assert!(human.contains("max_chunks=8"));

    let output = tokio::time::timeout(
        Duration::from_secs(20),
        run(READER, &["synthesize", "--seed", &id]).output(),
    )
    .await
    .unwrap()
    .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("forbidden"));
    assert!(!client_db.exists());
    task.abort();
}

#[test]
fn local_synthesis_is_disabled_by_default() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("config.toml");
    std::fs::write(&config, "").unwrap();
    Command::cargo_bin("graphrag")
        .unwrap()
        .env_clear()
        .env("HOME", temp.path().join("home"))
        .env("XDG_CONFIG_HOME", temp.path().join("xdg-config"))
        .current_dir(temp.path())
        .arg("--config")
        .arg(&config)
        .arg("--db-path")
        .arg(temp.path().join("db"))
        .args(["synthesize", "orchard pressure"])
        .timeout(Duration::from_secs(30))
        .assert()
        .code(4)
        .stderr(predicates::str::contains("[alchemist] enabled = true"));
}
