//! Provider-free real CLI retrieval. Database seeding runs in a separate
//! process because the embedded datastore cache survives a runtime's drop.
use assert_cmd::Command;
use chrono::{Duration as ChronoDuration, Utc};
use graphrag_config::RuntimeConfig;
use graphrag_core::{record_id_to_string, ChatExport, Note, SourceType};
use graphrag_db::{init_persistent, Repository};
use serde_json::{json, Value};
use std::fs;
use std::io::ErrorKind;
use std::net::TcpListener;
use std::path::PathBuf;
use std::time::Duration;

#[test]
#[ignore = "spawned explicitly by isolated CLI fixtures"]
fn keyword_fixture_seed() {
    let Some(directory) = std::env::var_os("GRAPHRAG_KEYWORD_FIXTURE") else {
        return;
    };
    let directory = PathBuf::from(directory);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let repo = Repository::new(init_persistent(directory.join("database")).await.unwrap());
        let mut records = Vec::new();
        for (label, age) in [("recent", 0), ("old", 90)] {
            let timestamp = Utc::now() - ChronoDuration::days(age);
            let source_path = directory.join(format!("{label} café 日本語.md"));
            fs::write(&source_path, "cafelexeme launch notes").unwrap();
            let uri = graphrag_core::normalize_file_uri(&source_path).unwrap();
        let mut source = repo.begin_file_import(SourceType::Markdown, label.into(), uri.clone(), "cafelexeme launch notes".into(), graphrag_core::normalized_content_hash("cafelexeme launch notes"), false).await.unwrap().source;
            let mut note = Note::new("cafelexeme launch notes").with_title(format!("{label} notes")).with_source(source.id.clone().unwrap()).with_source_generation(source.generation);
            note.created_at = timestamp;
            // Legacy vectors intentionally have no model metadata. Keyword
            // search must bypass compatibility and never touch the cache.
            note.embedding = vec![1.0; 1024];
            let note = repo.create_note(note).await.unwrap();
            repo.complete_file_import(&mut source).await.unwrap();
            let conversation = ChatExport::from_json(&json!([{
                "uuid": format!("keyword-{label}"), "name": format!("{label} discussion"),
                "summary": "cafelexeme launch summary",
                "created_at": timestamp.to_rfc3339(), "updated_at": timestamp.to_rfc3339(),
                "chat_messages": [{"uuid": format!("keyword-message-{label}"), "sender": "human", "text": "cafelexeme launch message", "created_at": timestamp.to_rfc3339()}]
            }]).to_string()).unwrap().conversations.remove(0);
            let conversation_id = repo.upsert_conversation(&conversation, Some(uri.clone()), json!({"conversation_id": conversation.uuid, "created_at": conversation.created_at, "summary": conversation.summary}), None).await.unwrap();
            let message_id = repo.upsert_message(&conversation_id, &conversation.uuid, 0, &conversation.messages[0], None).await.unwrap();
            records.push(json!({"label": label, "uri": uri, "note": record_id_to_string(note.id.as_ref().unwrap()), "message": record_id_to_string(&message_id), "conversation": record_id_to_string(&conversation_id)}));
        }
        fs::write(directory.join("seed.json"), serde_json::to_vec(&records).unwrap()).unwrap();
    });
}

struct Fixture {
    directory: tempfile::TempDir,
    config: PathBuf,
    unused_provider: TcpListener,
    records: Value,
}
impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("configuration.toml");
        let unused_provider = TcpListener::bind("127.0.0.1:0").unwrap();
        unused_provider.set_nonblocking(true).unwrap();
        let mut settings = RuntimeConfig::default();
        settings.database.path = directory.path().join("database");
        let endpoint = format!("http://{}", unused_provider.local_addr().unwrap());
        settings.inference.embedding_url = endpoint.clone();
        settings.inference.extraction_url = endpoint;
        settings.inference.timeout_secs = 1;
        settings.inference.retry_attempts = 1;
        fs::write(&config, settings.redacted_toml().unwrap()).unwrap();
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .env_clear()
            .env("HOME", directory.path().join("home"))
            .env("GRAPHRAG_KEYWORD_FIXTURE", directory.path())
            .current_dir(directory.path())
            .args([
                "--ignored",
                "--exact",
                "keyword_fixture_seed",
                "--test-threads=1",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "seed process failed: {}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        let records =
            serde_json::from_slice(&fs::read(directory.path().join("seed.json")).unwrap()).unwrap();
        Self {
            directory,
            config,
            unused_provider,
            records,
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
    fn keyword(&self, extra: &[&str]) -> Value {
        let output = self
            .command()
            .args([
                "search",
                "cafelexeme",
                "--mode",
                "keyword",
                "--scope",
                extra
                    .windows(2)
                    .find(|args| args[0] == "--scope")
                    .map(|args| args[1])
                    .unwrap_or("all"),
                "--format",
                "json",
            ])
            .args(extra.iter().enumerate().filter_map(|(index, argument)| {
                if *argument == "--scope" || (index > 0 && extra[index - 1] == "--scope") {
                    None
                } else {
                    Some(*argument)
                }
            }))
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["command"], "search");
        assert_eq!(envelope["success"], true);
        envelope["data"].clone()
    }
    fn assert_no_provider_requests(&self) {
        assert_eq!(
            self.unused_provider.accept().unwrap_err().kind(),
            ErrorKind::WouldBlock
        );
    }
}

#[test]
fn keyword_cli_reports_all_hit_kinds_and_consistent_filters_offline() {
    let fixture = Fixture::new();
    let data = fixture.keyword(&[]);
    assert_eq!(data["mode"], "keyword");
    assert_eq!(data["channels"], json!(["full_text"]));
    let results = data["results"].as_array().unwrap();
    assert_eq!(results.len(), 6);
    for kind in ["note", "message", "conversation-summary"] {
        assert_eq!(
            results.iter().filter(|hit| hit["hit_type"] == kind).count(),
            2
        );
    }
    let uri = fixture.records[0]["uri"].as_str().unwrap();
    let filtered = fixture.keyword(&["--source-uri", uri, "--since-days", "1"]);
    assert_eq!(filtered["results"].as_array().unwrap().len(), 3);
    assert!(filtered["results"]
        .as_array()
        .unwrap()
        .iter()
        .all(|hit| hit["source_uri"] == uri));
    let stale_uri = fixture.records[1]["uri"].as_str().unwrap();
    assert!(
        fixture.keyword(&["--source-uri", stale_uri, "--since-days", "1"])["results"]
            .as_array()
            .unwrap()
            .is_empty()
    );
    for scope in ["notes", "messages"] {
        assert_eq!(
            fixture.keyword(&["--scope", scope])["results"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
    fixture.assert_no_provider_requests();
}

#[test]
fn keyword_explain_jsonl_keeps_empty_pipeline_and_only_full_text_evidence() {
    let fixture = Fixture::new();
    let data = fixture.keyword(&["--explain"]);
    assert_eq!(data["pipeline"]["filters"]["mode"], "keyword");
    assert_eq!(data["pipeline"]["filters"]["graph"], "Off");
    for hit in data["results"].as_array().unwrap() {
        assert!(hit["vector"].is_null());
        assert!(hit["graph"].is_null());
        assert_eq!(hit["full_text"]["kind"], "bm25");
        assert_eq!(hit["fused"]["kind"], "bm25");
        assert_eq!(hit["final_score"]["kind"], "bm25");
        assert!(hit.get("embedding_provider").is_none());
    }
    let output = fixture
        .command()
        .args([
            "search",
            "unmatchedlexeme",
            "--mode",
            "keyword",
            "--scope",
            "all",
            "--explain",
            "--format",
            "jsonl",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8_lossy(&output).lines().count(), 1);
    let empty: Value = serde_json::from_slice(&output).unwrap();
    assert!(empty["data"]["result"].is_null());
    assert_eq!(empty["data"]["pipeline"]["filters"]["mode"], "keyword");
    fixture.assert_no_provider_requests();
}

#[test]
fn keyword_search_remains_available_after_vectorless_backup_restore() {
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
            "search",
            "cafelexeme",
            "--mode",
            "keyword",
            "--scope",
            "all",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let data: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(data["data"]["results"].as_array().unwrap().len(), 6);
    fixture.assert_no_provider_requests();
}

#[test]
fn keyword_rejects_explicit_graph_on_without_contacting_providers() {
    let fixture = Fixture::new();
    fixture
        .command()
        .args(["search", "cafelexeme", "--mode", "keyword", "--graph", "on"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains("--graph off"));
    fixture.assert_no_provider_requests();
}

#[test]
fn ordinary_keyword_jsonl_preserves_hit_envelopes_and_human_empty_names_mode() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args([
            "search",
            "cafelexeme",
            "--mode",
            "keyword",
            "--scope",
            "all",
            "--format",
            "jsonl",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let records = String::from_utf8(output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(records.len(), 6);
    assert!(records.iter().all(|record| record["data"]["id"].is_string()
        && record["data"]["mode"] == "keyword"
        && record["data"]["channels"] == json!(["full_text"])));
    fixture
        .command()
        .args(["search", "unmatchedlexeme", "--mode", "keyword"])
        .assert()
        .success()
        .stdout(predicates::str::contains(
            "Search mode: keyword (full_text)",
        ))
        .stdout(predicates::str::contains("No results found."));
    let output = fixture
        .command()
        .args([
            "search",
            "unmatchedlexeme",
            "--mode",
            "keyword",
            "--format",
            "jsonl",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert!(output.is_empty());
    fixture.assert_no_provider_requests();
}

#[test]
fn hybrid_provider_failure_suggests_a_runnable_keyword_command_with_same_filters() {
    let fixture = Fixture::new();
    let uri = fixture.records[0]["uri"].as_str().unwrap();
    let output = fixture
        .command()
        .arg("--db-path")
        .arg(fixture.directory.path().join("database"))
        .args([
            "search",
            "cafelexeme",
            "--scope",
            "all",
            "--since-days",
            "1",
            "--source-uri",
            uri,
            "--limit",
            "3",
            "--format",
            "json",
        ])
        .assert()
        .code(1)
        .get_output()
        .clone();
    assert!(
        output.stdout.is_empty(),
        "hybrid failure must not silently emit keyword results"
    );
    let stderr = String::from_utf8(output.stderr).unwrap();
    let recovery = stderr
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("graphrag "))
        .expect("exact keyword command on provider failure");
    assert!(recovery.contains("--mode keyword"));
    assert!(recovery.contains("--since-days 1"));
    assert!(recovery.contains("--limit 3"));
    assert!(recovery.contains("--scope all"));
    assert!(recovery.contains("--format json"));
    assert!(recovery.contains("--source-uri"));
    assert!(recovery.contains("--config"));
    assert!(recovery.contains("--db-path"));
    let executable = assert_cmd::cargo::cargo_bin("graphrag");
    let path = format!("{}:/usr/bin:/bin", executable.parent().unwrap().display());
    // Execute the exact displayed recovery from a different directory. All
    // config/database/source/query arguments must survive shell quoting.
    let elsewhere = fixture.directory.path().join("elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    let recovered = std::process::Command::new("/bin/sh")
        .env_clear()
        .env("HOME", fixture.directory.path().join("home"))
        .env("PATH", path)
        .current_dir(&elsewhere)
        .args(["-c", recovery])
        .output()
        .unwrap();
    assert!(
        recovered.status.success(),
        "{}",
        String::from_utf8_lossy(&recovered.stderr)
    );
    let envelope: Value = serde_json::from_slice(&recovered.stdout).unwrap();
    assert_eq!(envelope["data"]["mode"], "keyword");
    assert_eq!(envelope["data"]["results"].as_array().unwrap().len(), 3);
    assert!(envelope["data"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .all(|hit| hit["source_uri"] == uri));
}

#[test]
fn hybrid_embedding_request_failure_also_offers_explicit_keyword_recovery() {
    use std::io::{Read, Write};
    let fixture = Fixture::new();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = listener.local_addr().unwrap();
    let server = std::thread::spawn(move || {
        let deadline = std::time::Instant::now() + Duration::from_secs(10);
        let mut requests = Vec::new();
        while requests.len() < 2 {
            let (mut stream, _) = match listener.accept() {
                Ok(accepted) => accepted,
                Err(error) if error.kind() == ErrorKind::WouldBlock => {
                    assert!(
                        std::time::Instant::now() < deadline,
                        "expected provider requests"
                    );
                    std::thread::sleep(Duration::from_millis(10));
                    continue;
                }
                Err(error) => panic!("accept provider request: {error}"),
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let header_end = loop {
                let mut bytes = [0_u8; 1024];
                let read = stream.read(&mut bytes).unwrap();
                assert!(read > 0, "complete request headers");
                request.extend_from_slice(&bytes[..read]);
                if let Some(index) = request.windows(4).position(|bytes| bytes == b"\r\n\r\n") {
                    break index + 4;
                }
            };
            let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
            let body_len = headers
                .lines()
                .filter_map(|line| line.split_once(':'))
                .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
                .map(|(_, value)| value.trim().parse::<usize>().unwrap())
                .unwrap_or(0);
            let remaining = (header_end + body_len).saturating_sub(request.len());
            stream.read_exact(&mut vec![0_u8; remaining]).unwrap();
            requests.push(headers.lines().next().unwrap().to_owned());
            let status = if requests.len() == 1 {
                "200 OK"
            } else {
                "503 Service Unavailable"
            };
            let body = if requests.len() == 1 {
                "{}"
            } else {
                "{\"error\":\"fixture embedding unavailable\"}"
            };
            stream.write_all(format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len()).as_bytes()).unwrap();
        }
        requests
    });
    let output = fixture
        .command()
        .env("TEI_URL", format!("http://{address}"))
        .args(["search", "cafelexeme", "--scope", "all", "--format", "json"])
        .assert()
        .code(1)
        .get_output()
        .clone();
    assert!(output.stdout.is_empty());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("Hybrid search failed"));
    assert!(stderr.contains("--mode keyword"));
    let requests = server.join().unwrap();
    assert!(requests[0].starts_with("GET /health"));
    assert!(requests[1].starts_with("POST /embed"));
    fixture.assert_no_provider_requests();
}

#[cfg(unix)]
#[test]
fn keyword_navigation_hints_inspect_each_hit_kind_and_open_the_exact_source() {
    use std::os::unix::fs::PermissionsExt;
    let fixture = Fixture::new();
    let opener = fixture.directory.path().join("keyword test opener");
    let log = fixture.directory.path().join("opened-path.txt");
    fs::write(
        &opener,
        "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$GRAPHRAG_TEST_OPENER_LOG\"\n",
    )
    .unwrap();
    fs::set_permissions(&opener, fs::Permissions::from_mode(0o755)).unwrap();
    let mut config = RuntimeConfig::from_file(&fixture.config).unwrap();
    config.navigation.opener = vec![opener.to_string_lossy().into_owned()];
    fs::write(&fixture.config, config.redacted_toml().unwrap()).unwrap();
    let data = fixture.keyword(&[]);
    let executable = assert_cmd::cargo::cargo_bin("graphrag");
    let path = format!("{}:/usr/bin:/bin", executable.parent().unwrap().display());
    let elsewhere = fixture.directory.path().join("navigation-elsewhere");
    fs::create_dir(&elsewhere).unwrap();
    for hit in data["results"].as_array().unwrap() {
        let hint = hit["navigation"]["inspect_command"]
            .as_str()
            .expect("keyword inspect hint");
        assert!(hint.contains("--revision"));
        let output = std::process::Command::new("/bin/sh")
            .env_clear()
            .env("HOME", fixture.directory.path().join("home"))
            .env("PATH", &path)
            .current_dir(&elsewhere)
            .args(["-c", &format!("{hint} --format json")])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let inspected: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(inspected["data"]["id"], hit["id"]);
        assert_eq!(inspected["data"]["content"], hit["content"]);
        assert_eq!(inspected["data"]["revision"], hit["navigation"]["revision"]);
    }
    let hit = data["results"]
        .as_array()
        .unwrap()
        .iter()
        .find(|hit| hit["hit_type"] == "note")
        .unwrap();
    let hint = hit["navigation"]["open_command"]
        .as_str()
        .expect("keyword source hint");
    let output = std::process::Command::new("/bin/sh")
        .env_clear()
        .env("HOME", fixture.directory.path().join("home"))
        .env("PATH", &path)
        .env("GRAPHRAG_TEST_OPENER_LOG", &log)
        .current_dir(&elsewhere)
        .args(["-c", &format!("{hint} --format json")])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let opened: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(opened["data"]["id"], hit["id"]);
    assert_eq!(opened["data"]["launched"], true);
    assert_eq!(
        fs::read_to_string(log).unwrap().trim_end(),
        opened["data"]["path"].as_str().unwrap()
    );
    fixture.assert_no_provider_requests();
}

#[test]
fn hybrid_recovery_preserves_option_like_query_and_source_filter_values() {
    let fixture = Fixture::new();
    let output = fixture
        .command()
        .args([
            "search",
            "--source-uri=-missing-uri",
            "--format",
            "json",
            "--",
            "--cafelexeme",
        ])
        .assert()
        .code(1)
        .get_output()
        .clone();
    let stderr = String::from_utf8(output.stderr).unwrap();
    let recovery = stderr
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("graphrag "))
        .unwrap();
    assert!(recovery.ends_with(" -- '--cafelexeme'"));
    assert!(recovery.contains("--source-uri='-missing-uri'"));
    let executable = assert_cmd::cargo::cargo_bin("graphrag");
    let path = format!("{}:/usr/bin:/bin", executable.parent().unwrap().display());
    let output = std::process::Command::new("/bin/sh")
        .env_clear()
        .env("HOME", fixture.directory.path().join("home"))
        .env("PATH", path)
        .current_dir(fixture.directory.path())
        .args(["-c", recovery])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let envelope: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(envelope["data"]["mode"], "keyword");
    assert!(envelope["data"]["results"].as_array().unwrap().is_empty());
}
