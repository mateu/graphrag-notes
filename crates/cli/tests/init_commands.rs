use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use serde_json::Value;
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::TcpListener;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Isolate both dotenv discovery and every inherited configuration variable.
/// A developer's running providers and personal database must never affect
/// these first-run tests.
fn graphrag(directory: &Path) -> Command {
    let mut command = Command::cargo_bin("graphrag").unwrap();
    command
        .env_clear()
        .env("HOME", directory.join("home"))
        .env("XDG_CONFIG_HOME", directory.join("xdg-config"))
        .current_dir(directory)
        .timeout(Duration::from_secs(10));
    command
}

fn json_output(command: &mut Command) -> Value {
    let output = command.assert().success().get_output().stdout.clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], "init");
    assert_eq!(envelope["success"], true);
    envelope["data"].clone()
}

fn assert_json_path(value: &Value, expected: &Path) {
    assert_eq!(value.as_str(), expected.to_str());
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn native_default_config(directory: &Path) -> std::path::PathBuf {
    if cfg!(target_os = "macos") {
        directory.join("home/Library/Application Support/graphrag/config.toml")
    } else {
        directory.join("xdg-config/graphrag/config.toml")
    }
}

#[test]
fn new_config_preview_is_offline_and_creates_no_directories() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("configuration/nested/graphrag.toml");
    let db_path = temp.path().join("database/nested/data");
    let data = json_output(
        graphrag(temp.path())
            .arg("--config")
            .arg(&config_path)
            .arg("--db-path")
            .arg(&db_path)
            .args(["init", "--backend", "ollama", "--format", "json"]),
    );

    assert_json_path(&data["config_path"], &config_path);
    assert_json_path(&data["database_path"], &db_path);
    assert_eq!(data["config_exists"], false);
    assert_eq!(data["config_written"], false);
    assert_eq!(data["embedding"]["provider"], "ollama");
    assert_eq!(data["extraction"]["provider"], "ollama");
    assert!(data["embedding"]["model"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    assert!(data["extraction"]["model"]
        .as_str()
        .is_some_and(|s| !s.is_empty()));
    assert!(data["diagnostics"].is_null());
    assert!(data["next_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command.as_str().unwrap().contains("doctor")));
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn explicit_write_creates_each_valid_backend_config_without_opening_database() {
    for (backend, embedding, extraction) in
        [("ollama", "ollama", "ollama"), ("tei-tgi", "tei", "tgi")]
    {
        let temp = tempfile::tempdir().unwrap();
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let config_path = temp.path().join("configuration/graphrag.toml");
        let db_path = temp.path().join("database/nested/data");
        let data = json_output(
            graphrag(temp.path())
                .env("TEI_URL", &endpoint)
                .env("TGI_URL", &endpoint)
                .arg("--config")
                .arg(&config_path)
                .arg("--db-path")
                .arg(&db_path)
                .args(["init", "--backend", backend, "--write", "--format", "json"]),
        );

        assert_eq!(data["config_written"], true);
        assert_json_path(&data["config_path"], &config_path);
        assert_json_path(&data["database_path"], &db_path);
        let config = RuntimeConfig::from_file(&config_path).unwrap();
        config.validate().unwrap();
        assert_eq!(config.database.path, db_path);
        assert_eq!(config.inference.embedding_provider, embedding);
        assert_eq!(config.inference.extraction_provider, extraction);
        assert_eq!(data["embedding"]["endpoint"], endpoint);
        assert_eq!(data["extraction"]["endpoint"], endpoint);
        assert_eq!(
            data["file_settings"]["embedding"]["endpoint"],
            config.inference.embedding_url
        );
        assert_eq!(
            data["file_settings"]["extraction"]["endpoint"],
            config.inference.extraction_url
        );
        assert_ne!(config.inference.embedding_url, endpoint);
        assert_ne!(config.inference.extraction_url, endpoint);
        assert_eq!(listener.accept().unwrap_err().kind(), ErrorKind::WouldBlock);
        assert!(!temp.path().join("database").exists());

        // Check the new file through the same normal config path that the
        // user's subsequent commands use, not only the template serializer.
        graphrag(temp.path())
            .arg("--config")
            .arg(&config_path)
            .args(["config", "validate"])
            .assert()
            .success();
        assert!(!temp.path().join("database").exists());
    }
}

#[test]
fn existing_preview_preserves_config_and_honors_environment_and_cli_precedence() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("graphrag.toml");
    let original = "# Keep this comment and formatting.\n[database]\npath = \"file-db\"\n\n[inference]\nembedding_model = \"file-model\"\n";
    fs::write(&config_path, original).unwrap();
    let env_db = temp.path().join("env-db");
    let cli_db = temp.path().join("cli-db");

    let data = json_output(
        graphrag(temp.path())
            .env("GRAPHRAG_DB_PATH", &env_db)
            .env("TEI_MODEL", "environment-model")
            .arg("--config")
            .arg(&config_path)
            .arg("--db-path")
            .arg(&cli_db)
            .args(["init", "--format", "json"]),
    );
    assert_eq!(data["config_exists"], true);
    assert_eq!(data["config_written"], false);
    assert_json_path(&data["database_path"], &cli_db);
    assert_eq!(data["embedding"]["model"], "environment-model");
    assert!(data["next_commands"]
        .as_array()
        .unwrap()
        .iter()
        .all(|command| {
            let command = command.as_str().unwrap();
            command.contains("--db-path") && command.contains(cli_db.to_str().unwrap())
        }));
    assert_eq!(fs::read(&config_path).unwrap(), original.as_bytes());
    assert!(!env_db.exists());
    assert!(!cli_db.exists());

    let data = json_output(
        graphrag(temp.path())
            .env("GRAPHRAG_DB_PATH", &env_db)
            .arg("--config")
            .arg(&config_path)
            .args(["init", "--format", "json"]),
    );
    assert_json_path(&data["database_path"], &env_db);
    assert_eq!(fs::read(&config_path).unwrap(), original.as_bytes());
}

#[test]
fn existing_config_cannot_be_overwritten_or_changed_by_a_backend_preset() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("graphrag.toml");
    let original = "# Existing user configuration.\n[logging]\nlevel = \"info\"\n";
    fs::write(&config_path, original).unwrap();

    for args in [
        vec!["init", "--write"],
        vec!["init", "--backend", "ollama"],
        vec!["init", "--backend", "tei-tgi", "--write"],
    ] {
        graphrag(temp.path())
            .arg("--config")
            .arg(&config_path)
            .args(args)
            .assert()
            .code(2);
        assert_eq!(fs::read(&config_path).unwrap(), original.as_bytes());
    }
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
}

#[test]
fn invalid_existing_configuration_fails_without_changing_it() {
    for original in [
        "[inference\ntimeout_secs = nope",
        "[inference]\nembedding_provider = \"unknown\"\n",
    ] {
        let temp = tempfile::tempdir().unwrap();
        let config_path = temp.path().join("graphrag.toml");
        fs::write(&config_path, original).unwrap();
        graphrag(temp.path())
            .arg("--config")
            .arg(&config_path)
            .arg("init")
            .assert()
            .code(2);
        assert_eq!(fs::read(&config_path).unwrap(), original.as_bytes());
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 1);
    }
}

#[test]
fn invalid_environment_prevents_preview_and_write() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("configuration/graphrag.toml");
    for args in [vec!["init"], vec!["init", "--write"]] {
        graphrag(temp.path())
            .env("GRAPHRAG_AUGMENT_MAX_TOKENS", "not-a-number")
            .arg("--config")
            .arg(&config_path)
            .args(args)
            .assert()
            .code(2)
            .stderr(predicates::str::contains("GRAPHRAG_AUGMENT_MAX_TOKENS"));
        assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
    }
}

#[test]
fn memory_mode_is_rejected_without_creating_config() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = temp.path().join("configuration/graphrag.toml");
    graphrag(temp.path())
        .arg("--config")
        .arg(&config_path)
        .args(["--memory", "init", "--write"])
        .assert()
        .code(2)
        .stderr(predicates::str::contains("memory"));
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[test]
fn config_path_selection_uses_environment_and_explicit_override() {
    let temp = tempfile::tempdir().unwrap();
    let env_path = temp.path().join("environment/config.toml");
    let explicit_path = temp.path().join("explicit/config.toml");

    let data = json_output(
        graphrag(temp.path())
            .env("GRAPHRAG_CONFIG", &env_path)
            .args(["init", "--format", "json"]),
    );
    assert_json_path(&data["config_path"], &env_path);
    let data = json_output(
        graphrag(temp.path())
            .env("GRAPHRAG_CONFIG", &env_path)
            .arg("--config")
            .arg(&explicit_path)
            .args(["init", "--format", "json"]),
    );
    assert_json_path(&data["config_path"], &explicit_path);
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
#[test]
fn native_default_config_can_be_previewed_and_written_in_isolated_home() {
    let temp = tempfile::tempdir().unwrap();
    let config_path = native_default_config(temp.path());
    let data = json_output(graphrag(temp.path()).args(["init", "--format", "json"]));
    assert_json_path(&data["config_path"], &config_path);
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);

    let data = json_output(graphrag(temp.path()).args([
        "init",
        "--backend",
        "ollama",
        "--write",
        "--format",
        "json",
    ]));
    assert_json_path(&data["config_path"], &config_path);
    assert!(config_path.is_file());
    assert!(!temp.path().join("home/.graphrag").exists());
}

#[test]
fn preview_neither_contacts_configured_providers_nor_opens_existing_database() {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    let config_path = temp.path().join("graphrag.toml");
    fs::write(&config_path, "").unwrap();
    let db_path = temp.path().join("database");
    fs::create_dir(&db_path).unwrap();
    let sentinel = db_path.join("existing-user-data");
    fs::write(&sentinel, "untouched").unwrap();

    let data = json_output(
        graphrag(temp.path())
            .env("TEI_URL", &endpoint)
            .env("TGI_URL", &endpoint)
            .arg("--config")
            .arg(&config_path)
            .arg("--db-path")
            .arg(&db_path)
            .args(["init", "--format", "json"]),
    );
    assert_eq!(data["embedding"]["endpoint"], endpoint);
    assert_eq!(data["extraction"]["endpoint"], endpoint);
    assert_eq!(listener.accept().unwrap_err().kind(), ErrorKind::WouldBlock);
    assert_eq!(fs::read_to_string(&sentinel).unwrap(), "untouched");
    assert_eq!(fs::read_dir(&db_path).unwrap().count(), 1);
}

#[test]
fn explicit_provider_check_reports_unavailable_ollama_with_setup_commands() {
    let temp = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let endpoint = format!("http://{}", listener.local_addr().unwrap());
    drop(listener);
    let config_path = temp.path().join("configuration/graphrag.toml");
    let db_path = temp.path().join("database/nested/data");
    let output = graphrag(temp.path())
        .env("TEI_URL", &endpoint)
        .env("TGI_URL", &endpoint)
        .env("GRAPHRAG_INFERENCE_TIMEOUT_SECS", "1")
        .arg("--config")
        .arg(&config_path)
        .arg("--db-path")
        .arg(&db_path)
        .args(["init", "--backend", "ollama", "--check", "--format", "json"])
        .assert()
        .code(1)
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["success"], false);
    assert!(!envelope["warnings"].as_array().unwrap().is_empty());
    let data = &envelope["data"];
    assert_eq!(data["diagnostics"]["status"], "warning");
    assert_eq!(data["diagnostics"]["read_only"], true);
    let checks = data["diagnostics"]["checks"].as_array().unwrap();
    assert!(checks
        .iter()
        .any(|check| { check["name"] == "embedding_provider" && check["status"] == "warning" }));
    assert!(checks
        .iter()
        .any(|check| { check["name"] == "extraction_provider" && check["status"] == "warning" }));
    assert!(!checks
        .iter()
        .any(|check| check["name"].as_str().unwrap().starts_with("database")));
    let recovery = serde_json::to_string(&data["prerequisites"]).unwrap();
    assert!(recovery.contains("ollama serve"));
    assert!(recovery.contains("ollama pull"));
    assert_eq!(fs::read_dir(temp.path()).unwrap().count(), 0);
}

/// A local Ollama protocol fixture, either missing its models or returning
/// deterministic inference. Requests are recorded to distinguish metadata
/// checks from actual import/search inference and accidental model downloads.
struct FakeOllama {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    server: Option<thread::JoinHandle<()>>,
}

impl FakeOllama {
    fn start(healthy: bool) -> Self {
        Self::start_with_embedding_failure(healthy, None)
    }

    fn start_with_embedding_failure(healthy: bool, failure_text: Option<&'static str>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stop = Arc::new(AtomicBool::new(false));
        let server_requests = requests.clone();
        let server_stop = stop.clone();
        let server = thread::spawn(move || {
            let mut embedding = vec![0.0_f32; 1024];
            embedding[0] = 1.0;
            while !server_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // BSD/macOS can inherit the listener's nonblocking
                        // mode. The request reader below requires blocking I/O
                        // so a split POST body cannot be mistaken for EOF.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_secs(2)))
                            .unwrap();
                        let mut request = Vec::new();
                        let mut buffer = [0; 2048];
                        while !request.windows(4).any(|bytes| bytes == b"\r\n\r\n") {
                            let count = stream
                                .read(&mut buffer)
                                .expect("fake Ollama must receive complete request headers");
                            assert_ne!(count, 0, "request closed before its headers completed");
                            request.extend_from_slice(&buffer[..count]);
                        }
                        if let Some(header_end) = request
                            .windows(4)
                            .position(|bytes| bytes == b"\r\n\r\n")
                            .map(|position| position + 4)
                        {
                            let content_length = String::from_utf8_lossy(&request[..header_end])
                                .lines()
                                .find_map(|line| {
                                    let (name, value) = line.split_once(':')?;
                                    name.eq_ignore_ascii_case("content-length")
                                        .then(|| value.trim().parse::<usize>().ok())
                                        .flatten()
                                })
                                .unwrap_or(0);
                            let received = request.len();
                            if received < header_end + content_length {
                                request.resize(header_end + content_length, 0);
                                stream
                                    .read_exact(&mut request[received..])
                                    .expect("fake Ollama must consume the complete request body");
                            }
                        }
                        let request_line = String::from_utf8_lossy(&request)
                            .lines()
                            .next()
                            .unwrap_or_default()
                            .to_owned();
                        server_requests.lock().unwrap().push(request_line.clone());
                        let (status, body) = if request_line.starts_with("GET ")
                            && request_line.contains("/api/tags ")
                        {
                            let models = if healthy {
                                serde_json::json!([
                                    {"name": "bge-m3:latest"},
                                    {"name": "phi4-mini:latest"}
                                ])
                            } else {
                                serde_json::json!([])
                            };
                            ("200 OK", serde_json::json!({"models": models}).to_string())
                        } else if healthy && request_line.starts_with("POST /api/show ") {
                            ("200 OK", "{}".to_owned())
                        } else if failure_text.is_some_and(|text| {
                            (request_line.starts_with("POST /api/embed ")
                                || request_line.starts_with("POST /api/embeddings "))
                                && String::from_utf8_lossy(&request).contains(text)
                        }) {
                            (
                                "400 Bad Request",
                                r#"{"error":"fixture refuses this text"}"#.to_owned(),
                            )
                        } else if healthy && request_line.starts_with("POST /api/embed ") {
                            (
                                "200 OK",
                                serde_json::json!({"embeddings": [embedding]}).to_string(),
                            )
                        } else if healthy && request_line.starts_with("POST /api/embeddings ") {
                            (
                                "200 OK",
                                serde_json::json!({"embedding": embedding}).to_string(),
                            )
                        } else if healthy && request_line.starts_with("POST /api/chat ") {
                            (
                                "200 OK",
                                serde_json::json!({
                                    "message": {
                                        "role": "assistant",
                                        "content": "{\"entities\":[],\"relationships\":[]}"
                                    },
                                    "done": true,
                                    "done_reason": "stop"
                                })
                                .to_string(),
                            )
                        } else {
                            (
                                "404 Not Found",
                                r#"{"error":"model is not installed"}"#.to_owned(),
                            )
                        };
                        let response = format!(
                            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        );
                        stream
                            .write_all(response.as_bytes())
                            .expect("fake Ollama must write a complete HTTP response");
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(error) => panic!("fake Ollama listener failed: {error}"),
                }
            }
        });
        Self {
            endpoint,
            requests,
            stop,
            server: Some(server),
        }
    }
}

impl Drop for FakeOllama {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.server.take().unwrap().join().unwrap();
    }
}

#[test]
fn explicit_provider_check_reports_missing_model_without_mutating_existing_data() {
    let temp = tempfile::tempdir().unwrap();
    let server = FakeOllama::start(false);
    let config_path = temp.path().join("graphrag.toml");
    let original = "# Existing user settings remain byte-for-byte intact.\n";
    fs::write(&config_path, original).unwrap();
    let db_path = temp.path().join("database");
    fs::create_dir(&db_path).unwrap();
    let sentinel = db_path.join("existing-user-data");
    fs::write(&sentinel, "untouched").unwrap();
    let output = graphrag(temp.path())
        .env("TEI_PROVIDER", "ollama")
        .env("TGI_PROVIDER", "ollama")
        .env("OLLAMA_URL", &server.endpoint)
        .env("TEI_MODEL", "missing-embedding")
        .env("TGI_MODEL", "missing-extraction")
        .env("GRAPHRAG_INFERENCE_TIMEOUT_SECS", "1")
        .arg("--config")
        .arg(&config_path)
        .arg("--db-path")
        .arg(&db_path)
        .args(["init", "--check", "--format", "json"])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["success"], false);
    assert!(!envelope["errors"].as_array().unwrap().is_empty());
    let data = &envelope["data"];
    assert_eq!(data["diagnostics"]["status"], "failed");
    assert_eq!(data["diagnostics"]["read_only"], true);
    assert!(data["diagnostics"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "embedding_provider" && check["status"] == "failed"));
    assert!(data["diagnostics"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "extraction_model" && check["status"] == "failed"));
    let recovery = serde_json::to_string(&data["prerequisites"]).unwrap();
    assert!(recovery.contains("ollama pull 'missing-embedding'"));
    assert!(recovery.contains("ollama pull 'missing-extraction'"));
    assert!(recovery.contains("ollama serve"));
    assert_eq!(fs::read(&config_path).unwrap(), original.as_bytes());
    assert_eq!(fs::read_to_string(&sentinel).unwrap(), "untouched");
    assert_eq!(fs::read_dir(&db_path).unwrap().count(), 1);
    let requests = server.requests.lock().unwrap();
    assert!(requests
        .iter()
        .any(|request| request.starts_with("GET /api/tags ")));
    assert!(requests.iter().all(|request| {
        request.starts_with("GET /api/tags ")
            || request.starts_with("POST /api/embed ")
            || request.starts_with("POST /api/embeddings ")
            || request.starts_with("POST /api/show ")
    }));
}

#[test]
fn fresh_install_can_check_import_search_inspect_and_reimport_sample() {
    let temp = tempfile::tempdir().unwrap();
    let server = FakeOllama::start(true);
    let config_path = temp.path().join("configuration/graphrag.toml");
    let db_path = temp.path().join("database/data");
    let sample_path = temp.path().join("samples/first-notes.md");
    fs::create_dir(sample_path.parent().unwrap()).unwrap();
    fs::write(
        &sample_path,
        include_str!("../../../samples/first-notes.md"),
    )
    .unwrap();
    let canonical_sample_path = sample_path.canonicalize().unwrap();
    let configured = || {
        let mut command = graphrag(temp.path());
        command
            .env("TEI_URL", &server.endpoint)
            .env("TGI_URL", &server.endpoint)
            .arg("--config")
            .arg(&config_path);
        command
    };

    let preview = json_output(configured().arg("--db-path").arg(&db_path).args([
        "init",
        "--backend",
        "ollama",
        "--format",
        "json",
    ]));
    assert_json_path(&preview["sample_path"], &canonical_sample_path);
    assert!(preview["next_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command.as_str().unwrap().contains(" import ")));
    assert!(preview["next_commands"]
        .as_array()
        .unwrap()
        .iter()
        .any(|command| command.as_str().unwrap().contains(" search ")));
    assert!(!config_path.exists());
    assert!(!db_path.exists());

    let setup = json_output(configured().arg("--db-path").arg(&db_path).args([
        "init",
        "--backend",
        "ollama",
        "--write",
        "--check",
        "--format",
        "json",
    ]));
    assert_eq!(setup["config_written"], true);
    assert_eq!(setup["diagnostics"]["status"], "healthy");
    assert!(!db_path.exists());
    configured().args(["config", "validate"]).assert().success();

    configured()
        .arg("import")
        .arg(&sample_path)
        .assert()
        .success()
        .stdout(predicates::str::contains("created"));
    assert!(db_path.is_dir());
    let search = configured()
        .args(["search", "Atlas launch plan", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let search: Value = serde_json::from_slice(&search).unwrap();
    assert_eq!(search["command"], "search");
    let results = search["data"]["results"].as_array().unwrap();
    let launch = results
        .iter()
        .find(|result| {
            result["content"]
                .as_str()
                .unwrap()
                .contains("launch plan has three stages")
        })
        .expect("sample launch-plan chunk is retrievable");
    assert_eq!(
        launch["source_uri"],
        graphrag_core::normalize_file_uri(&sample_path).unwrap()
    );
    let navigation = &launch["navigation"];
    assert!(navigation["preview"].as_str().unwrap().contains("launch"));
    assert_eq!(navigation["provenance"]["source_uri"], launch["source_uri"]);
    assert!(navigation["provenance"]["start_line"].as_u64().is_some());
    assert!(navigation["inspect_command"]
        .as_str()
        .unwrap()
        .contains("--config"));
    assert!(navigation["inspect_command"]
        .as_str()
        .unwrap()
        .contains("--db-path"));
    assert!(navigation["open_command"]
        .as_str()
        .unwrap()
        .contains(" open "));
    let requests_before = server.requests.lock().unwrap().len();
    let inspected = configured()
        .args([
            "inspect",
            launch["id"].as_str().unwrap(),
            "--revision",
            navigation["revision"].as_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let inspected: Value = serde_json::from_slice(&inspected).unwrap();
    assert_eq!(inspected["data"]["content"], launch["content"]);
    assert_eq!(inspected["data"]["provenance"], navigation["provenance"]);
    assert_eq!(server.requests.lock().unwrap().len(), requests_before);

    #[cfg(unix)]
    {
        // Run the exact printed command from another directory; quoted config
        // and absolute database selection must still reach this same corpus.
        let elsewhere = temp.path().join("another-working-directory");
        fs::create_dir(&elsewhere).unwrap();
        let executable = env!("CARGO_BIN_EXE_graphrag").replace('\'', "'\\''");
        let command = navigation["inspect_command"].as_str().unwrap().replacen(
            "graphrag ",
            &format!("'{executable}' "),
            1,
        ) + " --format json";
        let output = Command::new("/bin/sh")
            .env_clear()
            .env("HOME", temp.path().join("home"))
            .current_dir(&elsewhere)
            .args(["-c", &command])
            .timeout(Duration::from_secs(10))
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["data"]["id"], launch["id"]);
        assert_eq!(envelope["data"]["content"], launch["content"]);
    }

    // Both retrieval renderer branches and both machine formats retain
    // navigation alongside the pre-existing evidence/context pipeline.
    for arguments in [
        vec![
            "search",
            "Atlas launch plan",
            "--explain",
            "--format",
            "json",
        ],
        vec![
            "search",
            "Atlas launch plan",
            "--explain",
            "--format",
            "jsonl",
        ],
        vec![
            "search",
            "Atlas launch plan",
            "--context",
            "--graph",
            "off",
            "--format",
            "json",
        ],
        vec![
            "search",
            "Atlas launch plan",
            "--context",
            "--graph",
            "off",
            "--explain",
            "--format",
            "jsonl",
        ],
    ] {
        let output = configured()
            .args(&arguments)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        if arguments.contains(&"jsonl") {
            for line in std::str::from_utf8(&output).unwrap().lines() {
                let envelope: Value = serde_json::from_str(line).unwrap();
                assert!(envelope["data"]["result"]["navigation"]["inspect_command"]
                    .as_str()
                    .unwrap()
                    .contains("--revision"));
                assert!(envelope["data"]["pipeline"]["filters"].is_object());
            }
        } else {
            let envelope: Value = serde_json::from_slice(&output).unwrap();
            let hit = &envelope["data"]["results"][0];
            assert!(hit["navigation"]["inspect_command"]
                .as_str()
                .unwrap()
                .contains("--revision"));
            if arguments.contains(&"--explain") {
                assert!(hit["final_score"].is_object());
                assert!(envelope["data"]["pipeline"]["filters"].is_object());
            } else {
                assert!(hit["content"].is_string());
            }
        }
    }
    let note = configured()
        .args([
            "notes",
            "show",
            launch["id"].as_str().unwrap(),
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let note: Value = serde_json::from_slice(&note).unwrap();
    assert!(note["data"]["content"]
        .as_str()
        .unwrap()
        .contains("launch plan has three stages"));

    let list_before = configured()
        .args(["notes", "list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let list_before: Value = serde_json::from_slice(&list_before).unwrap();
    let notes_before = list_before["data"]["notes"].as_array().unwrap();
    configured()
        .arg("import")
        .arg(&sample_path)
        .assert()
        .success()
        .stdout(predicates::str::contains("unchanged"));
    let list_after = configured()
        .args(["notes", "list", "--format", "json"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let list_after: Value = serde_json::from_slice(&list_after).unwrap();
    assert_eq!(
        list_after["data"]["notes"].as_array().unwrap(),
        notes_before
    );
    configured().arg("doctor").assert().success();
    assert!(server
        .requests
        .lock()
        .unwrap()
        .iter()
        .all(|request| !request.contains("/api/pull")));
}

#[test]
fn failed_provider_checks_redact_credentials_in_nested_diagnostics() {
    let temp = tempfile::tempdir().unwrap();
    let server = FakeOllama::start(false);
    let endpoint = format!(
        "http://user:secret@{}/?token=secret",
        server.endpoint.strip_prefix("http://").unwrap()
    );
    let config_path = temp.path().join("configuration/graphrag.toml");
    let output = graphrag(temp.path())
        .env("TEI_URL", &endpoint)
        .env("TGI_URL", &endpoint)
        .arg("--config")
        .arg(&config_path)
        .args(["init", "--backend", "ollama", "--check", "--format", "json"])
        .assert()
        .code(2)
        .get_output()
        .stdout
        .clone();
    let output = String::from_utf8(output).unwrap();
    assert!(!output.contains("secret"));
    assert!(!output.contains("user:"));
    let envelope: Value = serde_json::from_str(&output).unwrap();
    assert_eq!(envelope["success"], false);
    assert!(envelope["data"]["diagnostics"]["checks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|check| check["name"] == "embedding_provider" && check["status"] == "failed"));
    assert!(!config_path.exists());
    assert!(!temp.path().join("home/.graphrag").exists());
}

#[test]
fn real_chat_import_and_backfill_create_inspectable_search_results() {
    let temp = tempfile::tempdir().unwrap();
    let server = FakeOllama::start(true);
    let config = temp.path().join("config.toml");
    let database = temp.path().join("database");
    let chat = temp.path().join("chat export.json");
    fs::write(&chat, serde_json::json!([{
        "uuid": "atlas-navigation-chat", "name": "Atlas launch discussion",
        "summary": "Atlas launch plan and pilot feedback.",
        "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:03:00Z",
        "chat_messages": [
            {"uuid": "atlas-message-0", "sender": "human", "text": "What is the Atlas launch plan?"},
            {"uuid": "atlas-message-1", "sender": "assistant", "text": "Start with internal testing, then a small pilot, then open access after pilot feedback."},
            {"uuid": "atlas-message-2", "sender": "human", "text": "Please preserve the pilot feedback with its original Atlas context."}
        ]
    }]).to_string()).unwrap();
    let configured = || {
        let mut command = graphrag(temp.path());
        command
            .env("TEI_URL", &server.endpoint)
            .env("TGI_URL", &server.endpoint)
            .arg("--config")
            .arg(&config);
        command
    };
    configured()
        .arg("--db-path")
        .arg(&database)
        .args(["init", "--backend", "ollama", "--write"])
        .assert()
        .success();
    configured()
        .arg("import-chats")
        .arg(&chat)
        .args(["--mode", "hybrid", "--skip-extraction"])
        .assert()
        .success()
        .stdout(predicates::str::contains("Conversations imported: 1"));
    configured()
        .arg("migrate-chats")
        .arg(&chat)
        .arg("--skip-extraction")
        .assert()
        .success()
        .stdout(predicates::str::contains("Conversations imported: 1"));
    let searched = configured()
        .args([
            "search", "Atlas", "--scope", "all", "--limit", "20", "--format", "json",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let searched: Value = serde_json::from_slice(&searched).unwrap();
    let hits = searched["data"]["results"].as_array().unwrap();
    for kind in ["note", "message", "conversation-summary"] {
        let hit = hits
            .iter()
            .find(|hit| hit["hit_type"] == kind)
            .expect("every chat search kind is available");
        let inspected = configured()
            .args([
                "inspect",
                hit["id"].as_str().unwrap(),
                "--revision",
                hit["navigation"]["revision"].as_str().unwrap(),
                "--format",
                "json",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let inspected: Value = serde_json::from_slice(&inspected).unwrap();
        assert_eq!(inspected["data"]["hit_type"], kind);
        assert_eq!(
            inspected["data"]["provenance"]["conversation_uuid"],
            "atlas-navigation-chat"
        );
        if kind == "note" {
            // A summary-derived note has a direct conversation link but no
            // message links. Tied hybrid scores can select either that note
            // or a message-derived note, so validate their shared context.
            assert!(inspected["data"]["conversations"]
                .as_array()
                .unwrap()
                .iter()
                .any(
                    |conversation| conversation["uuid"] == "atlas-navigation-chat"
                        && conversation["summary"]
                            .as_str()
                            .is_some_and(|summary| !summary.is_empty())
                ));
        } else {
            assert!(!inspected["data"]["messages"].as_array().unwrap().is_empty());
        }
    }
}

#[test]
fn chat_import_failures_exit_five_and_keep_successful_peers_inspectable() {
    for action in ["import-chats", "migrate-chats"] {
        let temp = tempfile::tempdir().unwrap();
        let server = FakeOllama::start_with_embedding_failure(true, Some("FAIL_THIS_MESSAGE"));
        let config = temp.path().join("config.toml");
        let database = temp.path().join("database");
        let chat = temp.path().join("partial chat.json");
        fs::write(&chat, serde_json::json!([
            {
                "uuid": "atlas-successful-peer", "name": "Atlas successful conversation",
                "summary": "Atlas successful peer stays available.",
                "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:03:00Z",
                "chat_messages": [{"uuid": "atlas-success-message", "sender": "human", "text": "Keep this Atlas context available."}]
            },
            {
                "uuid": "atlas-partial-peer", "name": "Atlas incomplete conversation",
                "summary": "Atlas summary was written before its message failed.",
                "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:03:00Z",
                "chat_messages": [{"uuid": "atlas-failing-message", "sender": "human", "text": "FAIL_THIS_MESSAGE"}]
            }
        ]).to_string()).unwrap();
        let configured = || {
            let mut command = graphrag(temp.path());
            command
                .env("TEI_URL", &server.endpoint)
                .env("TGI_URL", &server.endpoint)
                .env("GRAPHRAG_INFERENCE_RETRY_ATTEMPTS", "1")
                .arg("--config")
                .arg(&config);
            command
        };
        configured()
            .arg("--db-path")
            .arg(&database)
            .args(["init", "--backend", "ollama", "--write"])
            .assert()
            .success();
        configured()
            .arg(action)
            .arg(&chat)
            .arg("--skip-extraction")
            .assert()
            .code(5)
            .stdout(predicates::str::contains("Conversations imported: 1"))
            .stdout(predicates::str::contains("Conversations failed: 1"))
            .stdout(predicates::str::contains(
                "failed conversations may have partial writes",
            ));
        let output = configured()
            .args([
                "search", "Atlas", "--scope", "all", "--limit", "20", "--format", "json",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let searched: Value = serde_json::from_slice(&output).unwrap();
        let hits = searched["data"]["results"].as_array().unwrap();
        for uuid in ["atlas-successful-peer", "atlas-partial-peer"] {
            let hit = hits
                .iter()
                .find(|hit| {
                    hit["hit_type"] == "conversation-summary"
                        && hit["navigation"]["provenance"]["conversation_uuid"] == uuid
                })
                .expect("successful and already-durable partial records remain available");
            configured()
                .args([
                    "inspect",
                    hit["id"].as_str().unwrap(),
                    "--revision",
                    hit["navigation"]["revision"].as_str().unwrap(),
                    "--format",
                    "json",
                ])
                .assert()
                .success()
                .stdout(predicates::str::contains(uuid));
        }
    }
}
