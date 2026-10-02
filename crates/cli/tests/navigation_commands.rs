//! Real CLI navigation against isolated, directly seeded databases.
//! Inspection/opening tests never start inference services or open personal data.

use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use graphrag_core::{
    normalize_file_uri, normalized_content_hash, record_id_to_string, ChatExport, Note,
    SourceIngestionStatus, SourceType,
};
use graphrag_db::{init_persistent, Repository};
use serde_json::{json, Value};
use std::fs;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

struct Fixture {
    directory: tempfile::TempDir,
    config_path: PathBuf,
    db_path: PathBuf,
    source_path: PathBuf,
    source_uri: String,
    source_id: String,
    source_note_id: String,
    manual_note_id: String,
    unbacked_note_id: String,
    web_note_id: String,
    conversation_id: String,
    message_ids: Vec<String>,
    unused_provider: TcpListener,
}

fn fixture_operation(directory: &std::path::Path, mut request: Value) -> Value {
    let request_path = directory.join("fixture-request.json");
    let response_path = directory.join("fixture-response.json");
    request["response_path"] = json!(response_path);
    fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
    let output = std::process::Command::new(std::env::current_exe().unwrap())
        .env_clear()
        .env("HOME", directory.join("home"))
        .env("XDG_CONFIG_HOME", directory.join("xdg-config"))
        .env("GRAPHRAG_NAVIGATION_FIXTURE_REQUEST", &request_path)
        .current_dir(directory)
        .args(["--exact", "database_fixture_process", "--nocapture"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "fixture process failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&fs::read(response_path).unwrap()).unwrap()
}

#[test]
fn database_fixture_process() {
    let Some(path) = std::env::var_os("GRAPHRAG_NAVIGATION_FIXTURE_REQUEST") else {
        return;
    };
    let request: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    let db_path = PathBuf::from(request["db_path"].as_str().unwrap());
    let response_path = PathBuf::from(request["response_path"].as_str().unwrap());
    // SurrealDB keeps embedded stores in a process-scoped registry. Seeding
    // and mutations run in this child process, which exits and releases the
    // RocksDB lock before any real CLI invocation in the parent test.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let response = runtime.block_on(async {
        let db = init_persistent(&db_path).await.unwrap();
        operate_fixture(Repository::new(db), &request).await
    });
    fs::write(response_path, serde_json::to_vec(&response).unwrap()).unwrap();
}

async fn operate_fixture(repo: Repository, request: &Value) -> Value {
    match request["operation"].as_str().unwrap() {
        "seed" => {
            let source_content = request["source_content"].as_str().unwrap();
            let seeded_source_uri = request["source_uri"].as_str().unwrap().to_owned();

            let mut plan = repo
                .begin_file_import(
                    SourceType::Markdown,
                    "Atlas 🦀".into(),
                    seeded_source_uri.clone(),
                    source_content.into(),
                    normalized_content_hash(source_content),
                    false,
                )
                .await
                .unwrap();
            let source_id = plan.source.id.clone().unwrap();
            let mut note =
                Note::new("The launch plan keeps café decisions and 🚀 milestones together.")
                    .with_title("Launch 日本語")
                    .with_source(source_id.clone())
                    .with_source_generation(plan.source.generation);
            note.chunk_heading_path = vec!["Atlas 🦀".into(), "Launch 日本語".into()];
            note.source_start_line = Some(5);
            note.source_end_line = Some(5);
            let source_note = repo.create_note(note).await.unwrap();
            repo.complete_file_import(&mut plan.source).await.unwrap();
            assert_eq!(plan.source.status, SourceIngestionStatus::Ready);
            let manual_note = repo
                .create_note(Note::new("A manual café note with no original file."))
                .await
                .unwrap();
            let unbacked_note = repo
                .create_note(Note::new("A manual note without any provenance."))
                .await
                .unwrap();
            let web_content = "A web-backed note is inspectable without browsing.";
            let mut web_plan = repo
                .begin_file_import(
                    SourceType::Url,
                    "Atlas web reference".into(),
                    "https://example.invalid/atlas".into(),
                    web_content.into(),
                    normalized_content_hash(web_content),
                    false,
                )
                .await
                .unwrap();
            let web_note = repo
                .create_note(
                    Note::new(web_content)
                        .with_source(web_plan.source.id.clone().unwrap())
                        .with_source_generation(web_plan.source.generation),
                )
                .await
                .unwrap();

            repo.complete_file_import(&mut web_plan.source)
                .await
                .unwrap();

            let conversation = ChatExport::from_json(
                    &json!([{
                        "uuid": "atlas-conversation-uuid",
                        "name": "Atlas pilot discussion 🦀",
                        "summary": "The Atlas pilot depends on a clear launch plan.",
                        "created_at": "2026-01-01T00:00:00Z",
                        "updated_at": "2026-01-01T00:05:00Z",
                        "chat_messages": [
                            {"uuid": "atlas-message-0", "sender": "human", "text": "Before: what should the pilot test?"},
                            {"uuid": "atlas-message-1", "sender": "assistant", "text": "Before: install and capture meeting decisions."},
                            {"uuid": "atlas-message-2", "sender": "human", "text": "Selected: can we find the café launch plan?"},
                            {"uuid": "atlas-message-3", "sender": "assistant", "text": "After: search the Atlas notebook and inspect the source."},
                            {"uuid": "atlas-message-4", "sender": "human", "text": "After: keep the original Markdown."}
                        ]
                    }])
                    .to_string(),
                )
                .unwrap()
                .conversations
                .remove(0);
            let conversation_id = repo
                .upsert_conversation(
                    &conversation,
                    Some(seeded_source_uri.clone()),
                    json!({
                        "conversation_id": &conversation.uuid,
                        "created_at": &conversation.created_at,
                        "summary": &conversation.summary,
                    }),
                    None,
                )
                .await
                .unwrap();
            let mut message_ids = Vec::new();
            for (index, message) in conversation.messages.iter().enumerate() {
                let id = repo
                    .upsert_message(&conversation_id, &conversation.uuid, index, message, None)
                    .await
                    .unwrap();
                message_ids.push(record_id_to_string(&id));
            }
            repo.link_note_to_conversation(manual_note.id.as_ref().unwrap(), &conversation_id)
                .await
                .unwrap();
            repo.link_note_to_message(
                manual_note.id.as_ref().unwrap(),
                &graphrag_db::parse_record_id(&message_ids[2], Some("message")).unwrap(),
            )
            .await
            .unwrap();
            json!([
                record_id_to_string(&source_id),
                record_id_to_string(source_note.id.as_ref().unwrap()),
                record_id_to_string(manual_note.id.as_ref().unwrap()),
                record_id_to_string(unbacked_note.id.as_ref().unwrap()),
                record_id_to_string(web_note.id.as_ref().unwrap()),
                record_id_to_string(&conversation_id),
                message_ids,
            ])
        }
        "edit_note" => {
            let id = request["id"].as_str().unwrap();
            let mut note = repo.get_note(id).await.unwrap().unwrap();
            note.content = request["content"].as_str().unwrap().to_owned();
            repo.update_note(id, note).await.unwrap();
            Value::Null
        }
        "refresh_source" => {
            let content = "A different source chunk in the next successful generation.";
            let mut plan = repo
                .begin_file_import(
                    SourceType::Markdown,
                    "Refreshed Atlas".into(),
                    request["source_uri"].as_str().unwrap().to_owned(),
                    content.into(),
                    normalized_content_hash(content),
                    true,
                )
                .await
                .unwrap();
            let replacement = repo
                .create_note(
                    Note::new(content)
                        .with_source(plan.source.id.clone().unwrap())
                        .with_source_generation(plan.source.generation),
                )
                .await
                .unwrap();
            repo.complete_file_import(&mut plan.source).await.unwrap();
            json!(record_id_to_string(replacement.id.as_ref().unwrap()))
        }
        operation => panic!("unknown isolated fixture operation: {operation}"),
    }
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let db_path = directory.path().join("database");
        let config_path = directory.path().join("configuration.toml");
        let source_path = directory
            .path()
            .join("notes $(touch injected-marker) `id` #100% ' space 🦀.md");
        let source_content = "# Atlas 🦀\n\n## Launch 日本語\n\nThe launch plan keeps café decisions and 🚀 milestones together.\n";
        fs::write(&source_path, source_content).unwrap();
        let source_path = source_path.canonicalize().unwrap();
        let source_uri = normalize_file_uri(&source_path).unwrap();
        let unused_provider = TcpListener::bind("127.0.0.1:0").unwrap();
        unused_provider.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", unused_provider.local_addr().unwrap());
        let mut config = RuntimeConfig::default();
        config.database.path = db_path.clone();
        config.inference.embedding_url = endpoint.clone();
        config.inference.extraction_url = endpoint;
        config.inference.timeout_secs = 1;
        config.inference.retry_attempts = 1;
        fs::write(&config_path, config.redacted_toml().unwrap()).unwrap();

        let seeded = fixture_operation(
            directory.path(),
            json!({"operation": "seed", "db_path": db_path, "source_uri": source_uri, "source_content": source_content}),
        );
        let source_id = seeded[0].as_str().unwrap().to_owned();
        let source_note_id = seeded[1].as_str().unwrap().to_owned();
        let manual_note_id = seeded[2].as_str().unwrap().to_owned();
        let unbacked_note_id = seeded[3].as_str().unwrap().to_owned();
        let web_note_id = seeded[4].as_str().unwrap().to_owned();
        let conversation_id = seeded[5].as_str().unwrap().to_owned();
        let message_ids = seeded[6]
            .as_array()
            .unwrap()
            .iter()
            .map(|id| id.as_str().unwrap().to_owned())
            .collect();

        Self {
            directory,
            config_path,
            db_path,
            source_path,
            source_uri,
            source_id,
            source_note_id,
            manual_note_id,
            unbacked_note_id,
            web_note_id,
            conversation_id,
            message_ids,
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
            .arg(&self.config_path)
            .timeout(Duration::from_secs(15));
        command
    }

    fn inspect(&self, id: &str) -> Value {
        let output = self
            .command()
            .args(["inspect", id, "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["schema_version"], 1);
        assert_eq!(envelope["command"], "inspect");
        assert_eq!(envelope["success"], true);
        assert_eq!(envelope["data"]["id"], id);
        envelope["data"].clone()
    }

    fn assert_no_provider_requests(&self) {
        let error = self.unused_provider.accept().unwrap_err();
        assert_eq!(error.kind(), ErrorKind::WouldBlock);
    }

    #[cfg(unix)]
    fn fake_opener(&self) -> (PathBuf, PathBuf) {
        use std::os::unix::fs::PermissionsExt;
        let executable = self.directory.path().join("fake opener executable");
        let log = self.directory.path().join("opener-arguments.txt");
        fs::write(
            &executable,
            "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$GRAPHRAG_TEST_OPENER_LOG\"\nprintf 'opener diagnostic output\\n'\n",
        )
        .unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        (executable, log)
    }
}

#[test]
fn inspection_keeps_complete_unicode_content_and_versioned_machine_envelopes() {
    let fixture = Fixture::new();
    let note = fixture.inspect(&fixture.source_note_id);
    assert_eq!(note["hit_type"], "note");
    assert_eq!(note["title"], "Launch 日本語");
    assert_eq!(
        note["content"],
        "The launch plan keeps café decisions and 🚀 milestones together."
    );
    assert!(!note["revision"].as_str().unwrap().is_empty());
    assert_eq!(note["provenance"]["source_id"], fixture.source_id);
    assert_eq!(note["provenance"]["source_uri"], fixture.source_uri);
    assert_eq!(note["provenance"]["source_generation"], 1);
    assert_eq!(
        note["provenance"]["heading_path"],
        json!(["Atlas 🦀", "Launch 日本語"])
    );
    assert_eq!(note["provenance"]["start_line"], 5);
    assert_eq!(note["provenance"]["end_line"], 5);

    let output = fixture
        .command()
        .args(["inspect", &fixture.source_note_id, "--format", "jsonl"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let records: Vec<_> = std::str::from_utf8(&output).unwrap().lines().collect();
    assert_eq!(records.len(), 1);
    let envelope: Value = serde_json::from_str(records[0]).unwrap();
    assert_eq!(envelope["command"], "inspect");
    assert_eq!(envelope["data"]["id"], fixture.source_note_id);
    fixture.assert_no_provider_requests();
}

#[test]
fn message_and_conversation_inspection_preserves_bounded_original_chat_context() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args([
            "inspect",
            &fixture.message_ids[2],
            "--neighbors",
            "1",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    let message = &envelope["data"];
    assert_eq!(message["hit_type"], "message");
    assert_eq!(
        message["content"],
        "Selected: can we find the café launch plan?"
    );
    assert_eq!(
        message["provenance"]["conversation_id"],
        fixture.conversation_id
    );
    assert_eq!(
        message["provenance"]["conversation_uuid"],
        "atlas-conversation-uuid"
    );
    assert_eq!(message["provenance"]["message_index"], 2);
    assert_eq!(message["provenance"]["role"], "human");
    let neighbors = message["messages"].as_array().unwrap();
    assert_eq!(neighbors.len(), 3);
    assert_eq!(neighbors[0]["id"], fixture.message_ids[1]);
    assert_eq!(neighbors[1]["id"], fixture.message_ids[2]);
    assert_eq!(neighbors[2]["id"], fixture.message_ids[3]);
    assert_eq!(neighbors[0]["role"], "assistant");
    assert_eq!(neighbors[1]["role"], "human");
    assert_eq!(neighbors[2]["role"], "assistant");
    assert!(neighbors
        .iter()
        .all(|neighbor| neighbor["conversation_uuid"] == "atlas-conversation-uuid"));
    for neighbor in neighbors {
        let standalone = fixture.inspect(neighbor["id"].as_str().unwrap());
        assert_eq!(
            neighbor["revision"], standalone["revision"],
            "context messages must carry the same usable guard as direct inspection"
        );
    }

    let output = fixture
        .command()
        .args([
            "inspect",
            &fixture.conversation_id,
            "--neighbors",
            "1",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    let conversation = &envelope["data"];
    assert_eq!(conversation["hit_type"], "conversation-summary");
    assert_eq!(conversation["title"], "Atlas pilot discussion 🦀");
    assert_eq!(
        conversation["content"],
        "The Atlas pilot depends on a clear launch plan."
    );
    assert_eq!(conversation["messages"].as_array().unwrap().len(), 3);
    assert_eq!(conversation["messages"][0]["id"], fixture.message_ids[0]);
    assert_eq!(conversation["messages"][2]["id"], fixture.message_ids[2]);
    assert_eq!(conversation["messages_truncated"], true);
    fixture.assert_no_provider_requests();
}

#[test]
fn a_derived_note_retains_original_conversation_and_message_identity() {
    let fixture = Fixture::new();
    let note = fixture.inspect(&fixture.manual_note_id);
    let conversations = note["conversations"].as_array().unwrap();
    assert_eq!(conversations.len(), 1);
    assert_eq!(conversations[0]["id"], fixture.conversation_id);
    assert_eq!(conversations[0]["uuid"], "atlas-conversation-uuid");
    assert!(note["messages"]
        .as_array()
        .unwrap()
        .iter()
        .any(|message| message["id"] == fixture.message_ids[2]
            && message["message_index"] == 2
            && message["role"] == "human"));
    fixture.assert_no_provider_requests();
}

#[test]
fn full_ids_are_required_and_neighbor_bounds_fail_before_provider_calls() {
    let fixture = Fixture::new();
    for id in ["1", "2", "unqualified-id", "entity:not-an-inspectable-hit"] {
        fixture.command().args(["inspect", id]).assert().code(2);
    }
    fixture
        .command()
        .args(["inspect", &fixture.message_ids[2], "--neighbors", "21"])
        .assert()
        .code(2);
    fixture.assert_no_provider_requests();
}

#[test]
fn revision_guard_rejects_changed_records_instead_of_inspecting_new_content() {
    let fixture = Fixture::new();
    let first = fixture.inspect(&fixture.manual_note_id);
    let revision = first["revision"].as_str().unwrap();
    fixture_operation(
        fixture.directory.path(),
        json!({
            "operation": "edit_note", "db_path": fixture.db_path,
            "id": fixture.manual_note_id, "content": "Changed after the displayed search result.",
        }),
    );
    fixture
        .command()
        .args(["inspect", &fixture.manual_note_id, "--revision", revision])
        .assert()
        .code(2);
    let current = fixture.inspect(&fixture.manual_note_id);
    assert_ne!(current["revision"], first["revision"]);
    assert_eq!(
        current["content"],
        "Changed after the displayed search result."
    );
    fixture.assert_no_provider_requests();
}

#[test]
fn source_refresh_never_redirects_a_printed_id_to_a_new_chunk() {
    let fixture = Fixture::new();
    let first = fixture.inspect(&fixture.source_note_id);
    let replacement = fixture_operation(
        fixture.directory.path(),
        json!({
            "operation": "refresh_source", "db_path": fixture.db_path, "source_uri": fixture.source_uri,
        }),
    );
    let replacement_id = replacement.as_str().unwrap();
    assert_ne!(replacement_id, fixture.source_note_id);
    fixture
        .command()
        .args([
            "inspect",
            &fixture.source_note_id,
            "--revision",
            first["revision"].as_str().unwrap(),
        ])
        .assert()
        .code(3);
    fixture.inspect(replacement_id);
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn opening_a_source_passes_one_literal_path_argument_without_a_shell() {
    let fixture = Fixture::new();
    let inspection = fixture.inspect(&fixture.source_note_id);
    let (opener, log) = fixture.fake_opener();
    let output = fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .arg("open")
        .arg(&fixture.source_note_id)
        .arg("--revision")
        .arg(inspection["revision"].as_str().unwrap())
        .arg("--opener")
        .arg(&opener)
        .args(["--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "open");
    assert_eq!(envelope["data"]["launched"], true);
    assert_eq!(
        envelope["data"]["path"],
        fixture.source_path.to_str().unwrap()
    );
    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        format!("{}\n", fixture.source_path.display())
    );
    assert!(!fixture.directory.path().join("injected-marker").exists());
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn missing_source_files_and_deleted_sources_do_not_launch_an_opener() {
    let fixture = Fixture::new();
    let (opener, log) = fixture.fake_opener();
    fs::remove_file(&fixture.source_path).unwrap();
    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .arg("open")
        .arg(&fixture.source_note_id)
        .arg("--opener")
        .arg(&opener)
        .assert()
        .code(3)
        .stderr(predicates::str::contains("file"));
    // Stored text remains inspectable even when its external file is gone.
    fixture.inspect(&fixture.source_note_id);
    assert!(!log.exists());
    fixture
        .command()
        .args(["sources", "delete", &fixture.source_id, "--yes"])
        .assert()
        .success();
    fixture
        .command()
        .args(["inspect", &fixture.source_note_id])
        .assert()
        .code(3);
    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .arg("open")
        .arg(&fixture.source_note_id)
        .arg("--opener")
        .arg(&opener)
        .assert()
        .code(3);
    assert!(!log.exists());
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn opening_manual_or_remote_records_does_not_launch_an_opener() {
    let fixture = Fixture::new();
    let (opener, log) = fixture.fake_opener();
    for id in [&fixture.unbacked_note_id, &fixture.web_note_id] {
        fixture.inspect(id);
        fixture
            .command()
            .env("GRAPHRAG_TEST_OPENER_LOG", &log)
            .arg("open")
            .arg(id)
            .arg("--opener")
            .arg(&opener)
            .assert()
            .code(2);
        assert!(!log.exists());
    }
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn opener_argv_and_executable_overrides_preserve_literal_arguments() {
    let fixture = Fixture::new();
    let (opener, log) = fixture.fake_opener();
    let mut config = RuntimeConfig::from_file(&fixture.config_path).unwrap();
    config.navigation.opener = vec![
        opener.to_string_lossy().into_owned(),
        "--literal".into(),
        "$(touch injected-argument-marker)".into(),
        "`id`".into(),
    ];
    fs::write(&fixture.config_path, config.redacted_toml().unwrap()).unwrap();
    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .args(["open", &fixture.source_note_id])
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        format!(
            "--literal\n$(touch injected-argument-marker)\n`id`\n{}\n",
            fixture.source_path.display()
        )
    );
    assert!(!fixture
        .directory
        .path()
        .join("injected-argument-marker")
        .exists());

    // An environment override is one executable path, not a shell command and
    // not a string to split at the spaces inside its pathname.
    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .env("GRAPHRAG_OPENER", &opener)
        .args(["open", &fixture.source_note_id])
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        format!("{}\n", fixture.source_path.display())
    );

    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .env("GRAPHRAG_OPENER", "/no-such-opener")
        .args(["open", &fixture.source_note_id, "--opener"])
        .arg(&opener)
        .assert()
        .success();
    assert_eq!(
        fs::read_to_string(&log).unwrap(),
        format!("{}\n", fixture.source_path.display())
    );
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn stale_revision_is_checked_before_launching_a_source_opener() {
    let fixture = Fixture::new();
    let (opener, log) = fixture.fake_opener();
    let first = fixture.inspect(&fixture.source_note_id);
    fixture_operation(
        fixture.directory.path(),
        json!({
            "operation": "edit_note", "db_path": fixture.db_path,
            "id": fixture.source_note_id, "content": "This note changed since its printed selection.",
        }),
    );
    fixture
        .command()
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .arg("open")
        .arg(&fixture.source_note_id)
        .arg("--revision")
        .arg(first["revision"].as_str().unwrap())
        .arg("--opener")
        .arg(&opener)
        .assert()
        .code(2)
        .stderr(predicates::str::contains("changed"));
    assert!(!log.exists());
    fixture.assert_no_provider_requests();
}
