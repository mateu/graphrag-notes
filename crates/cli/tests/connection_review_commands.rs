//! Offline review against real persistent databases, each owned by a child process.
use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use graphrag_core::{normalized_content_hash, record_id_to_string, EdgeType, Note, SourceType};
use graphrag_db::{init_persistent, repository::EdgeProposalDraft, Repository};
use serde_json::{json, Value};
use std::fs;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn fixture_operation(directory: &Path, request: Value) -> Value {
    let input = directory.join("request.json");
    let response = directory.join("response.json");
    fs::write(&input, serde_json::to_vec(&request).unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env("HOME", directory.join("home"))
        .env("GRAPHRAG_REVIEW_FIXTURE", &input)
        .current_dir(directory)
        .args([
            "--exact",
            "connection_review_fixture_process",
            "--nocapture",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&fs::read(response).unwrap()).unwrap()
}

#[test]
fn connection_review_fixture_process() {
    let Some(input) = std::env::var_os("GRAPHRAG_REVIEW_FIXTURE") else {
        return;
    };
    let input = PathBuf::from(input);
    let directory = input.parent().unwrap();
    let request: Value = serde_json::from_slice(&fs::read(&input).unwrap()).unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let response = runtime.block_on(async {
        let db = init_persistent(directory.join("database")).await.unwrap();
        let repo = Repository::new(db.clone());
        match request["operation"].as_str().unwrap() {
            "seed" => {
                let content = "# Atlas\n\nPilot observations 🦀 need a careful connection.";
                let source_path = directory.join("pilot's café 日本語.md");
                fs::write(&source_path, content).unwrap();
                let mut source = repo.begin_file_import(SourceType::Markdown, "Pilot source".into(), graphrag_core::normalize_file_uri(&source_path).unwrap(), content.into(), normalized_content_hash(content), false).await.unwrap().source;
                let mut imported = Note::new(content).with_title("Pilot source note").with_source(source.id.clone().unwrap()).with_source_generation(source.generation);
                imported.chunk_heading_path = vec!["Atlas".into()];
                imported.source_start_line = Some(1);
                imported.source_end_line = Some(3);
                imported.embedding = vec![1.0; 1024];
                let imported = repo.create_note(imported).await.unwrap();
                repo.complete_file_import(&mut source).await.unwrap();
                let first = repo.create_note(Note::new("Manual pilot review with original Unicode café context.").with_title("Manual pilot decision")).await.unwrap();
                let second = repo.create_note(Note::new("A separate evidence note for batch policy boundaries.").with_title("Other evidence")).await.unwrap();
                let imported_id = imported.id.unwrap();
                let first_id = first.id.unwrap();
                let second_id = second.id.unwrap();
                let pending = repo.upsert_gardener_proposal(&imported_id, &first_id, 0.9, "Shared pilot context".into(), Some("fixture-v1".into()), Some("fixture model".into())).await.unwrap();
                let lower = repo.upsert_gardener_proposal(&first_id, &second_id, 0.75, "Weaker context".into(), None, None).await.unwrap();
                let logical = repo.upsert_edge_proposal(EdgeProposalDraft {from_id: first_id.clone(), to_id: second_id, edge_type: EdgeType::Supports, confidence: 0.99, reason: "Explicit evidence required".into(), generator: "manual".into(), generator_version: None, model: None}).await.unwrap();
                json!({"pending": record_id_to_string(pending.id.as_ref().unwrap()), "lower": record_id_to_string(lower.id.as_ref().unwrap()), "logical": record_id_to_string(logical.id.as_ref().unwrap()), "manual_note": record_id_to_string(&first_id), "source_id": record_id_to_string(source.id.as_ref().unwrap())})
            }
            "snapshot" => {
                let proposals = repo.list_edge_proposals(None, 200).await.unwrap();
                let mut edges = Vec::new();
                for proposal in &proposals {
                    if let Some(edge) = &proposal.resulting_edge_id {
                        if repo.note_edge_exists(edge).await.unwrap() {
                            edges.push(record_id_to_string(edge));
                        }
                    }
                }
                json!({"proposals": proposals.into_iter().map(|proposal| json!({"id": record_id_to_string(proposal.id.as_ref().unwrap()), "proposal": proposal})).collect::<Vec<_>>(), "edges": edges})
            }
            "missing" => {
                // Simulate a legacy dangling proposal without lifecycle cleanup.
                let id = graphrag_db::repository::parse_record_id(request["note_id"].as_str().unwrap(), Some("note")).unwrap();
                db.query("DELETE $id").bind(("id", id)).await.unwrap().check().unwrap();
                json!({})
            }
            "replace_manual_content" => {
                let id = request["note_id"].as_str().unwrap();
                let mut note = repo.get_note(id).await.unwrap().unwrap();
                note.content = request["content"].as_str().unwrap().into();
                repo.update_note(id, note).await.unwrap();
                json!({})
            }
            _ => panic!("unknown fixture operation"),
        }
    });
    fs::write(
        directory.join("response.json"),
        serde_json::to_vec(&response).unwrap(),
    )
    .unwrap();
}

struct Fixture {
    directory: tempfile::TempDir,
    config: PathBuf,
    seed: Value,
    unused_provider: TcpListener,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("review config.toml");
        let unused_provider = TcpListener::bind("127.0.0.1:0").unwrap();
        unused_provider.set_nonblocking(true).unwrap();
        let mut settings = RuntimeConfig::default();
        settings.database.path = directory.path().join("database");
        let endpoint = format!("http://{}", unused_provider.local_addr().unwrap());
        settings.inference.embedding_url = endpoint.clone();
        settings.inference.extraction_url = endpoint;
        fs::write(&config, settings.redacted_toml().unwrap()).unwrap();
        let seed = fixture_operation(directory.path(), json!({"operation": "seed"}));
        Self {
            directory,
            config,
            seed,
            unused_provider,
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::cargo_bin("graphrag").unwrap();
        command
            .env_clear()
            .env("HOME", self.directory.path().join("home"))
            .env("XDG_CONFIG_HOME", self.directory.path().join("xdg-config"))
            .current_dir(self.directory.path())
            .arg("--config")
            .arg(&self.config)
            .timeout(Duration::from_secs(20));
        command
    }
    fn card(&self, id: &str) -> Value {
        let output = self
            .command()
            .args(["garden", "review", "--id", id, "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["command"], "garden.review");
        assert_eq!(envelope["success"], true);
        envelope["data"]["proposals"][0].clone()
    }
    fn snapshot(&self) -> Value {
        fixture_operation(self.directory.path(), json!({"operation": "snapshot"}))
    }
    fn no_inference(&self) {
        assert_eq!(
            self.unused_provider.accept().unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
    }
}

fn proposal<'a>(snapshot: &'a Value, id: &str) -> &'a Value {
    &snapshot["proposals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|row| row["id"] == id)
        .unwrap()["proposal"]
}

#[test]
fn inbox_shows_both_notes_provenance_and_replayable_inspection_offline() {
    let fixture = Fixture::new();
    let before = fixture.snapshot();
    let id = fixture.seed["pending"].as_str().unwrap();
    let card = fixture.card(id);
    assert_eq!(card["status"], "pending");
    assert_eq!(card["reason"], "Shared pilot context");
    assert_eq!(card["generator"], "gardener-similarity");
    assert!((card["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(card["accept_allowed"], true);
    let endpoints = [&card["from"], &card["to"]];
    assert!(endpoints
        .iter()
        .any(|note| note["title"] == "Pilot source note"));
    assert!(endpoints
        .iter()
        .any(|note| note["title"] == "Manual pilot decision"));
    let sourced = endpoints
        .iter()
        .find(|note| !note["provenance"]["source_id"].is_null())
        .unwrap();
    assert_eq!(sourced["provenance"]["heading_path"], json!(["Atlas"]));
    assert_eq!(sourced["provenance"]["source_generation"], 1);
    let elsewhere = fixture.directory.path().join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    let executable = assert_cmd::cargo::cargo_bin("graphrag");
    let path = format!("{}:/usr/bin:/bin", executable.parent().unwrap().display());
    for note in endpoints {
        let hint = format!(
            "{} --format json",
            note["inspect_command"].as_str().unwrap()
        );
        let output = std::process::Command::new("/bin/sh")
            .env_clear()
            .env("HOME", fixture.directory.path().join("home"))
            .env("PATH", &path)
            .current_dir(&elsewhere)
            .args(["-c", &hint])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let inspected: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(inspected["data"]["id"], note["id"]);
        assert_eq!(inspected["data"]["revision"], note["revision"]);
    }
    fixture
        .command()
        .args(["garden", "review", "--id", id])
        .assert()
        .success()
        .stdout(predicates::str::contains("Pilot source note"))
        .stdout(predicates::str::contains("Manual pilot decision"))
        .stdout(predicates::str::contains("Shared pilot context"))
        .stdout(predicates::str::contains("Source generation: 1"));
    let jsonl = fixture
        .command()
        .args(["garden", "review", "--format", "jsonl"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8_lossy(&jsonl).lines().count(), 3);
    for line in String::from_utf8_lossy(&jsonl).lines() {
        let line: Value = serde_json::from_str(line).unwrap();
        assert_eq!(line["command"], "garden.review");
        assert!(line["data"]["from"]["excerpt"].is_string());
        assert!(line["data"]["to"]["excerpt"].is_string());
    }
    assert_eq!(before, fixture.snapshot());
    fixture.no_inference();
}

#[test]
fn machine_excerpts_keep_unicode_and_ellipsis_within_the_character_limit() {
    let fixture = Fixture::new();
    let id = fixture.seed["pending"].as_str().unwrap();
    let note_id = fixture.seed["manual_note"].as_str().unwrap();
    for length in [499, 500, 501, 1_500] {
        let content = "🦀".repeat(length);
        fixture_operation(
            fixture.directory.path(),
            json!({"operation": "replace_manual_content", "note_id": note_id, "content": content}),
        );
        let before = fixture.snapshot();
        let expected = if length > 500 {
            format!("{}…", "🦀".repeat(499))
        } else {
            content.clone()
        };
        let card = fixture.card(id);
        let jsonl = fixture
            .command()
            .args(["garden", "review", "--id", id, "--format", "jsonl"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let jsonl: Value = serde_json::from_slice(&jsonl).unwrap();
        for card in [&card, &jsonl["data"]] {
            let endpoint = [&card["from"], &card["to"]]
                .into_iter()
                .find(|endpoint| endpoint["id"] == note_id)
                .unwrap();
            let excerpt = endpoint["excerpt"].as_str().unwrap();
            assert_eq!(excerpt.chars().count(), length.min(500));
            assert_eq!(excerpt, expected, "input length {length}");
        }
        assert_eq!(before, fixture.snapshot());
    }
    fixture.no_inference();
}

#[test]
fn skip_cancel_and_eof_leave_proposals_and_audit_unchanged() {
    let fixture = Fixture::new();
    let before = fixture.snapshot();
    let id = fixture.seed["pending"].as_str().unwrap();
    for input in ["s\n", "a\n\ns\n", "r\nreject\n", "q\n"] {
        fixture
            .command()
            .args(["garden", "review", "--id", id, "--interactive"])
            .write_stdin(input)
            .assert()
            .success();
        assert_eq!(before, fixture.snapshot());
    }
    fixture.no_inference();
}

#[test]
fn decisions_and_undo_keep_the_existing_audit_and_terminal_state() {
    let fixture = Fixture::new();
    let id = fixture.seed["pending"].as_str().unwrap();
    fixture
        .command()
        .args(["garden", "review", "--id", id, "--interactive"])
        .write_stdin("a\naccept\npilot evidence checked\n")
        .assert()
        .success()
        .stdout(predicates::str::contains("is accepted"));
    let accepted = fixture.snapshot();
    let audit = proposal(&accepted, id);
    assert_eq!(audit["status"], "accepted");
    assert_eq!(audit["reviewer"], "cli interactive review");
    assert_eq!(audit["action_reason"], "pilot evidence checked");
    assert_eq!(audit["acceptance_is_manual"], true);
    assert!(audit["reviewed_at"].is_string());
    assert_eq!(accepted["edges"].as_array().unwrap().len(), 1);
    let card = fixture.card(id);
    assert_eq!(card["accept_allowed"], false);
    assert!(card["undo_command"].as_str().unwrap().contains("--db-path"));
    fixture
        .command()
        .args(["garden", "review", "--id", id, "--interactive"])
        .write_stdin("u\nundo\nconnection reconsidered\n")
        .assert()
        .success()
        .stdout(predicates::str::contains("is superseded"));
    let undone = fixture.snapshot();
    let undone_audit = proposal(&undone, id);
    assert_eq!(undone_audit["status"], "superseded");
    assert!(undone_audit["resulting_edge_id"].is_null());
    assert_eq!(undone_audit["reviewed_at"], audit["reviewed_at"]);
    assert_eq!(undone_audit["action_reason"], audit["action_reason"]);
    assert_eq!(
        undone_audit["supersession_reason"],
        "connection reconsidered"
    );
    assert!(undone["edges"].as_array().unwrap().is_empty());
    fixture
        .command()
        .args(["garden", "review", "--id", id, "--interactive"])
        .write_stdin("a\ns\n")
        .assert()
        .success()
        .stderr(predicates::str::contains("Accept unavailable"));
    assert_eq!(undone, fixture.snapshot());
    let lower = fixture.seed["lower"].as_str().unwrap();
    fixture
        .command()
        .args(["garden", "review", "--id", lower, "--interactive"])
        .write_stdin("r\nreject\nnot enough evidence\n")
        .assert()
        .success();
    let rejected = fixture.snapshot();
    assert_eq!(proposal(&rejected, lower)["status"], "rejected");
    assert_eq!(
        proposal(&rejected, lower)["reviewer"],
        "cli interactive review"
    );
    assert_eq!(
        proposal(&rejected, lower)["action_reason"],
        "not enough evidence"
    );
    fixture.no_inference();
}

#[test]
fn missing_endpoints_block_acceptance_without_mutating_the_proposal() {
    let fixture = Fixture::new();
    fixture_operation(
        fixture.directory.path(),
        json!({"operation": "missing", "note_id": fixture.seed["manual_note"]}),
    );
    let before = fixture.snapshot();
    let id = fixture.seed["pending"].as_str().unwrap();
    let card = fixture.card(id);
    assert_eq!(card["accept_allowed"], false);
    assert!(card["from"]["available"] == false || card["to"]["available"] == false);
    fixture
        .command()
        .args(["garden", "review", "--id", id, "--interactive"])
        .write_stdin("a\ns\n")
        .assert()
        .success()
        .stderr(predicates::str::contains("Accept unavailable"));
    assert_eq!(before, fixture.snapshot());
    fixture.no_inference();
}

#[test]
fn batch_threshold_confirmation_and_generator_boundaries_are_preserved() {
    let fixture = Fixture::new();
    let before = fixture.snapshot();
    for args in [
        vec![
            "garden",
            "proposals",
            "accept",
            "--all",
            "--min-confidence",
            "0.8",
        ],
        vec!["garden", "proposals", "accept", "--all", "--yes"],
        vec![
            "garden",
            "proposals",
            "accept",
            "--all",
            "--min-confidence",
            "NaN",
            "--yes",
        ],
    ] {
        fixture.command().args(args).assert().code(2);
        assert_eq!(before, fixture.snapshot());
    }
    fixture
        .command()
        .args([
            "garden",
            "proposals",
            "accept",
            "--all",
            "--min-confidence",
            "0.8",
            "--yes",
        ])
        .assert()
        .success()
        .stdout(predicates::str::contains("Accepted 1"));
    let after = fixture.snapshot();
    assert_eq!(
        proposal(&after, fixture.seed["pending"].as_str().unwrap())["status"],
        "accepted"
    );
    assert_eq!(
        proposal(&after, fixture.seed["lower"].as_str().unwrap())["status"],
        "pending"
    );
    assert_eq!(
        proposal(&after, fixture.seed["logical"].as_str().unwrap())["status"],
        "pending"
    );
    fixture.no_inference();
}

#[test]
fn invalid_review_options_are_clear_and_machine_stdout_has_no_prompts() {
    let fixture = Fixture::new();
    for args in [
        vec!["garden", "review", "--interactive", "--format", "json"],
        vec!["garden", "review", "--interactive", "--format", "jsonl"],
        vec!["garden", "review", "--limit", "0"],
        vec!["garden", "review", "--limit", "201"],
        vec!["garden", "review", "--id", "1"],
        vec!["garden", "review", "--id", "proposed_edge:"],
    ] {
        fixture
            .command()
            .args(args)
            .assert()
            .code(2)
            .stdout(predicates::str::is_empty());
    }
    fixture
        .command()
        .args([
            "garden",
            "review",
            "--id",
            "proposed_edge:absent",
            "--format",
            "json",
        ])
        .assert()
        .code(3)
        .stdout(predicates::str::is_empty());
    fixture.no_inference();
}

#[test]
fn restored_imported_notes_keep_source_identity_when_local_uris_are_redacted() {
    let fixture = Fixture::new();
    let archive = fixture.directory.path().join("backup");
    let restored = fixture.directory.path().join("restored");
    fixture
        .command()
        .args(["backup", "create"])
        .arg(&archive)
        .assert()
        .success();
    fixture
        .command()
        .arg("--db-path")
        .arg(&restored)
        .args(["backup", "restore"])
        .arg(&archive)
        .assert()
        .success();
    let output = fixture
        .command()
        .arg("--db-path")
        .arg(&restored)
        .args([
            "garden",
            "review",
            "--id",
            fixture.seed["pending"].as_str().unwrap(),
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let output = String::from_utf8(output).unwrap();
    assert!(output.contains(&format!(
        "Source ID: {}",
        fixture.seed["source_id"].as_str().unwrap()
    )));
    assert!(output.contains("Source type: markdown"));
    assert!(output.contains("Source URI: unavailable; stored provenance retained"));
    assert_eq!(output.matches("Source: manual/local note").count(), 1);
    fixture.no_inference();
}
