//! Folder sync acceptance checks against real CLI subprocesses and isolated
//! temporary files/databases. The local HTTP fixture supplies deterministic
//! vectors and never contacts live inference services or a personal corpus.

use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use graphrag_core::normalize_file_uri;
use serde_json::{json, Value};
use std::fs;
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[derive(Clone, Debug)]
struct ProviderRequest {
    line: String,
    body: String,
}

struct ProviderFixture {
    endpoint: String,
    requests: Arc<Mutex<Vec<ProviderRequest>>>,
    failing_text: Arc<Mutex<Option<String>>>,
    remove_on_embedding: Arc<Mutex<Option<(String, PathBuf)>>>,
    stop: Arc<AtomicBool>,
    server: Option<thread::JoinHandle<()>>,
}

fn read_http_request(stream: &mut TcpStream) -> (String, String) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 2048];
    let header_end = loop {
        let count = stream.read(&mut buffer).unwrap();
        assert_ne!(count, 0, "provider request ended before its headers");
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|chunk| chunk == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..header_end]);
    let line = headers.lines().next().unwrap().to_owned();
    let content_length = headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    if bytes.len() < header_end + content_length {
        let received = bytes.len();
        bytes.resize(header_end + content_length, 0);
        stream.read_exact(&mut bytes[received..]).unwrap();
    }
    let body = String::from_utf8(bytes[header_end..header_end + content_length].to_vec()).unwrap();
    (line, body)
}

impl ProviderFixture {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let failing_text = Arc::new(Mutex::new(None::<String>));
        let remove_on_embedding = Arc::new(Mutex::new(None::<(String, PathBuf)>));
        let stop = Arc::new(AtomicBool::new(false));
        let recorded = requests.clone();
        let failures = failing_text.clone();
        let remove_file = remove_on_embedding.clone();
        let stopping = stop.clone();
        let server = thread::spawn(move || {
            let mut vector = vec![0.0_f32; 1024];
            vector[0] = 1.0;
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("provider fixture accept failed: {error}"),
                };
                let (line, body) = read_http_request(&mut stream);
                recorded.lock().unwrap().push(ProviderRequest {
                    line: line.clone(),
                    body: body.clone(),
                });
                let embedding_request = line.starts_with("POST /api/embeddings ")
                    || line.starts_with("POST /api/embed ");
                // Remove a later planned file during an earlier provider
                // request. This creates a real discovery/execution race
                // without modifying the CLI or seeding internal job state.
                let removed = {
                    let mut hook = remove_file.lock().unwrap();
                    if embedding_request
                        && hook
                            .as_ref()
                            .is_some_and(|(needle, _)| body.contains(needle))
                    {
                        hook.take().map(|(_, path)| path)
                    } else {
                        None
                    }
                };
                if let Some(path) = removed {
                    fs::remove_file(path).unwrap();
                }
                let should_fail = embedding_request
                    && failures
                        .lock()
                        .unwrap()
                        .as_ref()
                        .is_some_and(|needle| body.contains(needle));
                let (status, payload) = if should_fail {
                    (
                        "503 Service Unavailable",
                        json!({"error": "isolated embedding failure"}),
                    )
                } else if line.starts_with("GET /api/tags ") {
                    (
                        "200 OK",
                        json!({"models": [{"name": "bge-m3:latest"}, {"name": "phi4-mini:latest"}]}),
                    )
                } else if line.starts_with("POST /api/embeddings ") {
                    ("200 OK", json!({"embedding": &vector}))
                } else if line.starts_with("POST /api/embed ") {
                    ("200 OK", json!({"embeddings": [&vector]}))
                } else if line.starts_with("POST /api/chat ") {
                    (
                        "200 OK",
                        json!({
                            "message": {"role": "assistant", "content": "{\"entities\":[],\"relationships\":[]}"},
                            "done": true,
                            "done_reason": "stop",
                        }),
                    )
                } else if line.starts_with("POST /api/show ") {
                    ("200 OK", json!({}))
                } else {
                    (
                        "404 Not Found",
                        json!({"error": "unexpected fixture route"}),
                    )
                };
                let payload = payload.to_string();
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                    payload.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        Self {
            endpoint,
            requests,
            failing_text,
            remove_on_embedding,
            stop,
            server: Some(server),
        }
    }

    fn fail_embeddings_containing(&self, text: Option<&str>) {
        *self.failing_text.lock().unwrap() = text.map(str::to_owned);
    }

    fn requests(&self) -> Vec<ProviderRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn remove_file_during_embedding(&self, text: &str, path: &std::path::Path) {
        *self.remove_on_embedding.lock().unwrap() = Some((text.into(), path.into()));
    }
}

impl Drop for ProviderFixture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.server.take().unwrap().join().unwrap();
    }
}

struct Fixture {
    directory: tempfile::TempDir,
    root: PathBuf,
    config_path: PathBuf,
    db_path: PathBuf,
    provider: ProviderFixture,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let root = directory.path().join("Atlas notes 🦀");
        fs::create_dir(&root).unwrap();
        let root = root.canonicalize().unwrap();
        let config_path = directory.path().join("selected-user-config.toml");
        let db_path = directory.path().join("database");
        let provider = ProviderFixture::new();
        let mut config = RuntimeConfig::default();
        config.database.path = db_path.clone();
        config.inference.embedding_provider = "ollama".into();
        config.inference.extraction_provider = "ollama".into();
        config.inference.embedding_url = provider.endpoint.clone();
        config.inference.extraction_url = provider.endpoint.clone();
        config.inference.embedding_model = "bge-m3".into();
        config.inference.extraction_model = "phi4-mini".into();
        config.inference.timeout_secs = 2;
        config.inference.ollama_timeout_secs = 2;
        config.inference.retry_attempts = 1;
        config.inference.cache_enabled = false;
        config.logging.level = "error".into();
        config.search.default_limit = 37;
        fs::write(
            &config_path,
            format!(
                "# Preserve this user comment and Unicode café label.\n{}",
                config.redacted_toml().unwrap()
            ),
        )
        .unwrap();
        Self {
            directory,
            root,
            config_path,
            db_path,
            provider,
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

    fn register(&self) {
        let data = envelope_data(
            self.command()
                .args(["folders", "add", "notes"])
                .arg(&self.root)
                .args(["--format", "json"]),
            "folders.add",
            0,
        );
        assert_eq!(data["name"], "notes");
        assert_eq!(data["folder"]["path"], json!(&self.root));
    }

    fn write_note(&self, name: &str, marker: &str) -> PathBuf {
        let path = self.root.join(name);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(
            &path,
            format!("# Atlas {marker}\n\nThe Atlas notebook records {marker} decisions and the café launch plan for the next team meeting.\n"),
        )
        .unwrap();
        path
    }

    fn sync(&self, dry_run: bool, expected_code: i32) -> Value {
        let mut command = self.command();
        command.args(["sync", "notes", "--format", "json"]);
        if dry_run {
            command.arg("--dry-run");
        }
        envelope_data(&mut command, "sync", expected_code)
    }

    fn notes(&self) -> Vec<Value> {
        envelope_data(
            self.command()
                .args(["notes", "list", "--limit", "100", "--format", "json"]),
            "notes.list",
            0,
        )["notes"]
            .as_array()
            .unwrap()
            .clone()
    }

    fn source(&self, uri: &str) -> Value {
        raw_json(
            self.command()
                .args(["sources", "show", uri, "--format", "json"]),
        )
    }

    fn job(&self, id: &str) -> Value {
        raw_json(
            self.command()
                .args(["jobs", "show", id, "--format", "json"]),
        )
    }

    fn prune_preview(&self) -> Value {
        envelope_data(
            self.command()
                .args(["folders", "prune", "notes", "--format", "json"]),
            "folders.prune",
            0,
        )
    }
}

fn raw_json(command: &mut Command) -> Value {
    let output = command.assert().success().get_output().stdout.clone();
    serde_json::from_slice(&output).unwrap()
}

fn envelope_data(command: &mut Command, name: &str, expected_code: i32) -> Value {
    let output = command
        .assert()
        .code(expected_code)
        .get_output()
        .stdout
        .clone();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["command"], name);
    assert_eq!(envelope["success"], expected_code == 0);
    if expected_code != 0 {
        assert!(!envelope["errors"].as_array().unwrap().is_empty());
    }
    envelope["data"].clone()
}

fn files(report: &Value) -> &[Value] {
    report["files"].as_array().unwrap()
}

fn status_paths(report: &Value) -> Vec<(String, String)> {
    files(report)
        .iter()
        .map(|file| {
            (
                file["status"].as_str().unwrap().to_owned(),
                file["path"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

fn assert_hint_context(hint: &str, fixture: &Fixture) {
    assert!(hint.contains("--config"), "selected config missing: {hint}");
    assert!(hint.contains(fixture.config_path.to_str().unwrap()));
    assert!(hint.contains("--db-path"));
    assert!(hint.contains(fixture.db_path.to_str().unwrap()));
}

#[test]
fn registration_preserves_config_and_dry_run_discovers_without_providers_or_writes() {
    let fixture = Fixture::new();
    let original = fs::read_to_string(&fixture.config_path).unwrap();
    let alpha = fixture.write_note("alpha.md", "alpha");
    let nested = fixture.write_note("nested/café.markdown", "nested");
    fixture.write_note(".obsidian/settings.md", "excluded");
    fixture.write_note("ignored.txt", "plain-text");
    fixture.register();
    let registered = fs::read_to_string(&fixture.config_path).unwrap();
    assert!(registered.starts_with(&original));
    let config = RuntimeConfig::from_file(&fixture.config_path).unwrap();
    assert_eq!(config.database.path, fixture.db_path);
    assert_eq!(config.search.default_limit, 37);
    assert_eq!(config.inference.embedding_url, fixture.provider.endpoint);
    assert_eq!(config.folders["notes"].path, fixture.root);
    assert!(fixture.provider.requests().is_empty());
    assert!(!fixture.db_path.exists(), "registration opened a database");

    let listed = envelope_data(
        fixture
            .command()
            .args(["folders", "list", "--format", "json"]),
        "folders.list",
        0,
    );
    assert_eq!(listed["notes"]["path"], json!(&fixture.root));
    assert!(!fixture.db_path.exists());
    let preview = fixture.sync(true, 0);
    assert_eq!(preview["dry_run"], true);
    assert!(preview["job_id"].is_null());
    assert_eq!(
        status_paths(&preview),
        vec![
            ("created".into(), alpha.to_str().unwrap().into()),
            ("created".into(), nested.to_str().unwrap().into()),
        ]
    );
    assert!(fixture.notes().is_empty());
    assert_eq!(
        raw_json(fixture.command().args(["jobs", "list", "--format", "json"])),
        json!([])
    );
    assert!(fixture.provider.requests().is_empty());
    assert_eq!(
        fs::read_to_string(&fixture.config_path).unwrap(),
        registered
    );
}

#[test]
fn unchanged_sync_is_provider_free_and_does_not_create_another_checkpoint() {
    let fixture = Fixture::new();
    fixture.write_note("pilot.md", "initialpilot");
    fixture.register();
    let imported = fixture.sync(false, 0);
    assert_eq!(files(&imported)[0]["status"], "created");
    let before = fixture.notes();
    let job_id = imported["job_id"].as_str().unwrap();
    let job_before = fixture.job(job_id);
    assert_eq!(job_before["status"], "completed");
    assert_eq!(job_before["completed_count"], 1);
    let request_count = fixture.provider.requests().len();
    assert!(request_count > 0, "initial import did not embed its note");

    let unchanged = fixture.sync(false, 0);
    assert_eq!(files(&unchanged)[0]["status"], "unchanged");
    assert!(unchanged["job_id"].is_null());
    assert_eq!(fixture.notes(), before);
    assert_eq!(fixture.job(job_id), job_before);
    assert_eq!(fixture.provider.requests().len(), request_count);
}

#[test]
fn a_failed_changed_file_retains_the_last_searchable_generation_and_can_resume() {
    let fixture = Fixture::new();
    let path = fixture.write_note("launch.md", "oldlaunch");
    fixture.register();
    let imported = fixture.sync(false, 0);
    let source_id = files(&imported)[0]["source_id"].clone();
    let original_notes = fixture.notes();
    fixture.write_note("launch.md", "retrylaunch");
    fixture
        .provider
        .fail_embeddings_containing(Some("retrylaunch"));
    let failed = fixture.sync(false, 5);
    assert_eq!(files(&failed)[0]["status"], "failed");
    assert_eq!(files(&failed)[0]["source_id"], source_id);
    assert_eq!(files(&failed)[0]["generation"], 1);
    assert_eq!(fixture.notes(), original_notes);
    let uri = normalize_file_uri(&path).unwrap();
    let source = fixture.source(&uri);
    assert_eq!(source["status"], "failed");
    assert_eq!(source["generation"], 2);
    assert_eq!(source["successful_generation"], 1);
    let job_id = failed["job_id"].as_str().unwrap();
    let job = fixture.job(job_id);
    assert_eq!(job["status"], "failed");
    assert_eq!(job["failed_count"], 1);
    assert_eq!(job["checkpoint"], uri);
    let hint = files(&failed)[0]["retry_command"].as_str().unwrap();
    assert_hint_context(hint, &fixture);
    assert!(hint.contains(job_id));

    fixture.provider.fail_embeddings_containing(None);
    let search = envelope_data(
        fixture.command().args([
            "search",
            "oldlaunch",
            "--scope",
            "notes",
            "--graph",
            "off",
            "--format",
            "json",
        ]),
        "search",
        0,
    );
    assert!(search["results"]
        .as_array()
        .unwrap()
        .iter()
        .any(|hit| { hit["content"].as_str().unwrap().contains("oldlaunch") }));
    let resumed = envelope_data(
        fixture
            .command()
            .args(["sync", "--resume", job_id, "--format", "json"]),
        "sync",
        0,
    );
    assert_eq!(resumed["job_id"], job_id);
    assert_eq!(files(&resumed)[0]["status"], "updated");
    assert_eq!(files(&resumed)[0]["source_id"], source_id);
    assert_eq!(files(&resumed)[0]["generation"], 3);
    let current_notes = fixture.notes();
    assert!(current_notes
        .iter()
        .all(|note| !note["content"].as_str().unwrap().contains("oldlaunch")));
    assert!(current_notes
        .iter()
        .any(|note| note["content"].as_str().unwrap().contains("retrylaunch")));
    assert_eq!(fixture.job(job_id)["status"], "completed");
}

#[test]
fn resume_pins_roots_and_file_identities_while_skipping_completed_peers() {
    let fixture = Fixture::new();
    fixture.write_note("a.md", "firstpeer");
    let failing = fixture.write_note("b.md", "failedpeer");
    let last = fixture.write_note("c.md", "lastpeer");
    fixture.register();
    fixture
        .provider
        .fail_embeddings_containing(Some("failedpeer"));
    let failed = fixture.sync(false, 5);
    assert_eq!(
        files(&failed)
            .iter()
            .map(|file| file["status"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["created", "failed", "created"]
    );
    let job_id = failed["job_id"].as_str().unwrap();
    let job = fixture.job(job_id);
    assert_eq!(job["status"], "failed");
    assert_eq!(job["total_count"], 3);
    assert_eq!(job["item_count"], 3);
    assert_eq!(job["completed_count"], 2);
    assert_eq!(job["failed_count"], 1);
    assert_eq!(job["checkpoint"], normalize_file_uri(&last).unwrap());

    let another_root = fixture.directory.path().join("new-configured-root");
    fs::create_dir(&another_root).unwrap();
    let another_root = another_root.canonicalize().unwrap();
    fs::write(
        another_root.join("new.md"),
        "Do not include this new configured root.",
    )
    .unwrap();
    fixture.write_note("d.md", "newlydiscovered");
    let original_config = fs::read_to_string(&fixture.config_path).unwrap();
    let changed_config = original_config.replace(
        fixture.root.to_str().unwrap(),
        another_root.to_str().unwrap(),
    );
    assert_ne!(changed_config, original_config);
    fs::write(&fixture.config_path, &changed_config).unwrap();
    fixture.provider.fail_embeddings_containing(None);
    let requests_before = fixture.provider.requests().len();
    let resumed = envelope_data(
        fixture
            .command()
            .args(["sync", "--resume", job_id, "--format", "json"]),
        "sync",
        0,
    );
    assert_eq!(resumed["folders"][0]["path"], json!(&fixture.root));
    assert_eq!(files(&resumed).len(), 3);
    assert_eq!(
        files(&resumed)
            .iter()
            .map(|file| file["status"].as_str().unwrap())
            .collect::<Vec<_>>(),
        ["unchanged", "updated", "unchanged"]
    );
    assert_eq!(
        files(&resumed)[1]["source_uri"],
        normalize_file_uri(&failing).unwrap()
    );
    assert_eq!(
        fs::read_to_string(&fixture.config_path).unwrap(),
        changed_config
    );
    let requests = fixture.provider.requests();
    let embeddings_after_resume = requests[requests_before..]
        .iter()
        .filter(|request| request.line.starts_with("POST /api/embeddings "))
        .collect::<Vec<_>>();
    assert!(!embeddings_after_resume.is_empty());
    assert!(embeddings_after_resume
        .iter()
        .all(|request| request.body.contains("failedpeer")));
    assert_eq!(fixture.notes().len(), 3);
    let completed = fixture.job(job_id);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["completed_count"], 3);
    assert_eq!(completed["failed_count"], 0);
    let request_count = fixture.provider.requests().len();
    fixture
        .command()
        .args(["jobs", "resume", job_id])
        .assert()
        .failure()
        .stderr(predicates::str::contains("sync --resume"));
    assert_eq!(fixture.provider.requests().len(), request_count);
}

#[test]
fn missing_sources_require_a_current_deterministic_prune_preview() {
    let fixture = Fixture::new();
    let alpha = fixture.write_note("alpha.md", "missingalpha");
    let beta = fixture.write_note("beta.md", "missingbeta");
    fixture.register();
    fixture.sync(false, 0);
    let before = fixture.notes();
    let request_count = fixture.provider.requests().len();
    fs::remove_file(&alpha).unwrap();
    fs::remove_file(&beta).unwrap();
    let missing = fixture.sync(false, 0);
    assert_eq!(
        status_paths(&missing),
        vec![
            ("missing".into(), alpha.to_str().unwrap().into()),
            ("missing".into(), beta.to_str().unwrap().into()),
        ]
    );
    assert!(missing["job_id"].is_null());
    assert_eq!(fixture.notes(), before);
    let preview = fixture.prune_preview();
    let repeated = fixture.prune_preview();
    assert_eq!(preview, repeated);
    assert_eq!(preview["dry_run"], true);
    assert_eq!(files(&preview).len(), 2);
    assert_eq!(files(&preview)[0]["generated_records"]["notes"], 1);
    assert_hint_context(preview["confirm_command"].as_str().unwrap(), &fixture);
    let revision = preview["revision"].as_str().unwrap();

    // A restored file invalidates the preview token; nothing else is deleted.
    fixture.write_note("beta.md", "missingbeta");
    fixture
        .command()
        .args([
            "folders",
            "prune",
            "notes",
            "--yes",
            "--revision",
            revision,
            "--format",
            "json",
        ])
        .assert()
        .code(2);
    assert_eq!(fixture.notes(), before);
    let current = fixture.prune_preview();
    assert_ne!(current["revision"], preview["revision"]);
    let pruned = envelope_data(
        fixture.command().args([
            "folders",
            "prune",
            "notes",
            "--yes",
            "--revision",
            current["revision"].as_str().unwrap(),
            "--format",
            "json",
        ]),
        "folders.prune",
        0,
    );
    assert_eq!(pruned["dry_run"], false);
    assert_eq!(files(&pruned).len(), 1);
    assert_eq!(files(&pruned)[0]["file"]["status"], "pruned");
    let retained = fixture.notes();
    assert_eq!(retained.len(), 1);
    assert!(retained[0]["content"]
        .as_str()
        .unwrap()
        .contains("missingbeta"));
    assert_eq!(fixture.provider.requests().len(), request_count);
}

#[test]
fn an_unavailable_root_never_turns_its_sources_into_missing_or_prunable_records() {
    let fixture = Fixture::new();
    fixture.write_note("kept.md", "retainedroot");
    fixture.register();
    fixture.sync(false, 0);
    let before = fixture.notes();
    let requests_before = fixture.provider.requests().len();
    let moved_root = fixture.root.with_file_name("temporarily-offline");
    fs::rename(&fixture.root, &moved_root).unwrap();
    let unavailable = fixture.sync(true, 5);
    assert_eq!(files(&unavailable).len(), 1);
    assert_eq!(files(&unavailable)[0]["status"], "failed");
    assert!(files(&unavailable)[0]["source_id"].is_null());
    assert!(files(&unavailable)[0]["error"]
        .as_str()
        .unwrap()
        .contains("no missing files"));
    let refused = envelope_data(
        fixture
            .command()
            .args(["folders", "prune", "notes", "--format", "json"]),
        "folders.prune",
        5,
    );
    assert_eq!(files(&refused)[0]["status"], "failed");
    assert_eq!(fixture.notes(), before);
    assert_eq!(fixture.provider.requests().len(), requests_before);
}

#[test]
fn pinned_files_that_disappear_before_source_creation_stay_failed_until_restored() {
    let fixture = Fixture::new();
    fixture.write_note("a.md", "racefirst");
    let disappearing = fixture.write_note("b.md", "racelater");
    fixture.register();
    fixture
        .provider
        .remove_file_during_embedding("racefirst", &disappearing);
    let failed = fixture.sync(false, 5);
    assert_eq!(files(&failed)[0]["status"], "created");
    assert_eq!(files(&failed)[1]["status"], "failed");
    assert!(files(&failed)[1]["source_id"].is_null());
    assert!(!disappearing.exists());
    let job_id = failed["job_id"].as_str().unwrap();
    let uri = normalize_file_uri(&disappearing).unwrap();
    assert_eq!(fixture.job(job_id)["checkpoint"], uri);
    assert_eq!(
        raw_json(
            fixture
                .command()
                .args(["sources", "list", "--format", "json"])
        )
        .as_array()
        .unwrap()
        .len(),
        1,
        "the disappearing file never created a source record"
    );

    let request_count = fixture.provider.requests().len();
    let missing = envelope_data(
        fixture
            .command()
            .args(["sync", "--resume", job_id, "--format", "json"]),
        "sync",
        5,
    );
    assert_eq!(files(&missing).len(), 2);
    let missing_file = files(&missing)
        .iter()
        .find(|file| file["source_uri"] == uri)
        .unwrap();
    assert_eq!(missing_file["status"], "missing");
    assert!(missing_file["source_id"].is_null());
    assert!(missing_file["error"].is_string());
    assert_hint_context(missing_file["retry_command"].as_str().unwrap(), &fixture);
    let still_failed = fixture.job(job_id);
    assert_eq!(still_failed["status"], "failed");
    assert_eq!(still_failed["completed_count"], 1);
    assert_eq!(still_failed["failed_count"], 1);
    assert_eq!(fixture.provider.requests().len(), request_count);

    fixture.write_note("b.md", "racelater");
    let restored = envelope_data(
        fixture
            .command()
            .args(["sync", "--resume", job_id, "--format", "json"]),
        "sync",
        0,
    );
    assert_eq!(files(&restored)[0]["status"], "unchanged");
    assert_eq!(files(&restored)[1]["status"], "created");
    assert_eq!(fixture.job(job_id)["status"], "completed");
    assert_eq!(fixture.notes().len(), 2);
}

#[test]
fn custom_globs_and_non_recursive_registration_bound_public_discovery() {
    let fixture = Fixture::new();
    let selected = fixture.write_note("selected.md", "selectedrule");
    fixture.write_note("skipped.markdown", "extensionrule");
    fixture.write_note("skip-private.md", "excluderule");
    fixture.write_note("nested/child.md", "recursionrule");
    let registration = envelope_data(
        fixture
            .command()
            .args(["folders", "add", "notes"])
            .arg(&fixture.root)
            .args([
                "--no-recursive",
                "--include",
                "*.md",
                "--exclude",
                "skip*.md",
                "--format",
                "json",
            ]),
        "folders.add",
        0,
    );
    assert_eq!(registration["folder"]["recursive"], false);
    assert_eq!(registration["folder"]["include"], json!(["*.md"]));
    assert!(registration["folder"]["exclude"]
        .as_array()
        .unwrap()
        .contains(&json!("skip*.md")));
    let preview = fixture.sync(true, 0);
    assert_eq!(
        status_paths(&preview),
        [("created".into(), selected.to_str().unwrap().into())]
    );
    assert!(fixture.provider.requests().is_empty());
    assert!(fixture.notes().is_empty());
}

#[test]
fn discovery_failed_utf8_peer_stays_in_the_job_and_its_retry_imports_it() {
    let fixture = Fixture::new();
    let successful = fixture.write_note("a.md", "readablepeer");
    let unreadable = fixture.root.join("b.md");
    fs::write(&unreadable, [0xff, 0xfe]).unwrap();
    fixture.register();
    let failed = fixture.sync(false, 5);
    assert_eq!(files(&failed)[0]["status"], "created");
    assert_eq!(files(&failed)[1]["status"], "failed");
    let uri = normalize_file_uri(&unreadable).unwrap();
    assert_eq!(files(&failed)[1]["source_uri"], uri);
    assert!(files(&failed)[1]["source_id"].is_null());
    let job_id = failed["job_id"].as_str().unwrap();
    let job = fixture.job(job_id);
    assert_eq!(job["status"], "failed");
    assert_eq!(job["total_count"], 2);
    assert_eq!(job["completed_count"], 1);
    assert_eq!(job["failed_count"], 1);
    assert_eq!(fixture.notes().len(), 1);
    let hint = files(&failed)[1]["retry_command"].as_str().unwrap();
    assert_hint_context(hint, &fixture);
    assert!(hint.contains(job_id));

    fixture.write_note("b.md", "restoredutf8peer");
    let requests_before = fixture.provider.requests().len();
    #[cfg(unix)]
    let resumed = {
        let executable = env!("CARGO_BIN_EXE_graphrag").replace('\'', "'\\''");
        let command = hint.replacen("graphrag ", &format!("'{executable}' "), 1) + " --format json";
        let elsewhere = fixture
            .directory
            .path()
            .join("retry-from-another-directory");
        fs::create_dir(&elsewhere).unwrap();
        envelope_data(
            Command::new("/bin/sh")
                .env_clear()
                .env("HOME", fixture.directory.path().join("home"))
                .env(
                    "XDG_CONFIG_HOME",
                    fixture.directory.path().join("xdg-config"),
                )
                .current_dir(&elsewhere)
                .args(["-c", &command])
                .timeout(Duration::from_secs(15)),
            "sync",
            0,
        )
    };
    #[cfg(not(unix))]
    let resumed = envelope_data(
        fixture
            .command()
            .args(["sync", "--resume", job_id, "--format", "json"]),
        "sync",
        0,
    );
    assert_eq!(resumed["job_id"], job_id);
    assert_eq!(
        status_paths(&resumed),
        [
            ("unchanged".into(), successful.to_str().unwrap().into()),
            ("created".into(), unreadable.to_str().unwrap().into()),
        ]
    );
    assert_eq!(fixture.notes().len(), 2);
    let completed = fixture.job(job_id);
    assert_eq!(completed["status"], "completed");
    assert_eq!(completed["completed_count"], 2);
    assert_eq!(completed["failed_count"], 0);
    let requests = fixture.provider.requests();
    let embeddings = requests[requests_before..]
        .iter()
        .filter(|request| request.line.starts_with("POST /api/embeddings "))
        .collect::<Vec<_>>();
    assert!(!embeddings.is_empty());
    assert!(embeddings
        .iter()
        .all(|request| request.body.contains("restoredutf8peer")));
}

#[cfg(unix)]
#[test]
fn a_nested_directory_symlink_retains_its_sources_while_other_missing_sources_prune() {
    let fixture = Fixture::new();
    fixture.write_note("nested/kept.md", "nestedsymlinkretained");
    let removable = fixture.write_note("remove.md", "intentionalmissing");
    fixture.register();
    fixture.sync(false, 0);
    fs::remove_file(&removable).unwrap();
    let before = fixture.prune_preview();
    assert_eq!(files(&before).len(), 1);
    let requests_before = fixture.provider.requests().len();

    // Discovery ignores this directory symlink. Its stored child URI must
    // also remain retained, even though following the symlink yields ENOENT.
    let nested = fixture.root.join("nested");
    let saved = fixture.directory.path().join("saved-nested-directory");
    let unrelated = fixture.directory.path().join("unrelated-empty-directory");
    fs::create_dir(&unrelated).unwrap();
    fs::rename(&nested, &saved).unwrap();
    std::os::unix::fs::symlink(&unrelated, &nested).unwrap();
    let discovered = fixture.sync(true, 0);
    assert!(files(&discovered)
        .iter()
        .any(|file| file["status"] == "ignored" && file["path"] == json!(&nested)));
    let after = fixture.prune_preview();
    assert_eq!(files(&after).len(), 1);
    assert_eq!(files(&after)[0]["file"]["path"], json!(&removable));
    assert_eq!(after["revision"], before["revision"]);
    let pruned = envelope_data(
        fixture.command().args([
            "folders",
            "prune",
            "notes",
            "--yes",
            "--revision",
            after["revision"].as_str().unwrap(),
            "--format",
            "json",
        ]),
        "folders.prune",
        0,
    );
    assert_eq!(files(&pruned).len(), 1);
    assert_eq!(files(&pruned)[0]["file"]["status"], "pruned");
    let notes = fixture.notes();
    assert_eq!(notes.len(), 1);
    assert!(notes[0]["content"]
        .as_str()
        .unwrap()
        .contains("nestedsymlinkretained"));
    assert_eq!(fixture.provider.requests().len(), requests_before);
}

#[cfg(unix)]
#[test]
fn replacing_a_registered_root_ancestor_with_a_symlink_refuses_pruning() {
    let mut fixture = Fixture::new();
    let host = fixture.directory.path().join("registered-host-directory");
    fs::create_dir(&host).unwrap();
    let root = host.join("notes");
    fs::rename(&fixture.root, &root).unwrap();
    fixture.root = root.canonicalize().unwrap();
    fixture.write_note("kept.md", "rootancestorretained");
    fixture.register();
    fixture.sync(false, 0);
    let notes_before = fixture.notes();
    let requests_before = fixture.provider.requests().len();

    let saved = fixture.directory.path().join("saved-host-directory");
    let unrelated = fixture.directory.path().join("unrelated-host-directory");
    fs::create_dir_all(unrelated.join("notes")).unwrap();
    fs::rename(&host, &saved).unwrap();
    std::os::unix::fs::symlink(&unrelated, &host).unwrap();
    // The leaf root still appears to be a regular directory when its
    // ancestor is followed; only a canonical identity guard catches this.
    assert!(fs::symlink_metadata(&fixture.root).unwrap().is_dir());
    let unavailable = fixture.sync(true, 5);
    assert!(files(&unavailable)
        .iter()
        .all(|file| file["status"] != "missing"));
    let refused = envelope_data(
        fixture
            .command()
            .args(["folders", "prune", "notes", "--format", "json"]),
        "folders.prune",
        5,
    );
    assert!(files(&refused)
        .iter()
        .all(|file| file["status"] != "missing"));
    assert_eq!(fixture.notes(), notes_before);
    assert_eq!(fixture.provider.requests().len(), requests_before);
}

#[test]
fn pruning_a_detached_notes_source_preserves_portable_provenance_and_reimports_restored_files() {
    let fixture = Fixture::new();
    let path = fixture.write_note("source.md", "originalsource");
    fixture.register();
    let synced = fixture.sync(false, 0);
    let source_id = files(&synced)[0]["source_id"].as_str().unwrap();
    let uri = normalize_file_uri(&path).unwrap();
    let original_source = fixture.source(&uri);
    let generated = fixture.notes();
    assert_eq!(generated.len(), 1);
    let manual_content = "My manual café annotation preserves the original source provenance.\n";
    let edit_content = fixture.directory.path().join("manual annotation.txt");
    fs::write(&edit_content, manual_content).unwrap();
    let detached = envelope_data(
        fixture
            .command()
            .args([
                "notes",
                "edit",
                generated[0]["id"].as_str().unwrap(),
                "--detach",
                "--title",
                "Manual provenance annotation",
                "--content-file",
            ])
            .arg(&edit_content)
            .args(["--format", "json"]),
        "notes.edit",
        0,
    );
    assert_eq!(detached["detached"], true);
    assert_eq!(detached["note"]["source_id"], original_source["id"]);
    assert!(detached["note"]["source_generation"].is_null());
    let manual = fixture
        .notes()
        .into_iter()
        .find(|note| note["content"] == manual_content)
        .unwrap();
    let manual_id = manual["id"].as_str().unwrap();
    assert_ne!(manual_id, generated[0]["id"].as_str().unwrap());
    let requests_before = fixture.provider.requests().len();
    fs::remove_file(&path).unwrap();
    let preview = fixture.prune_preview();
    assert_eq!(files(&preview).len(), 1);
    assert_eq!(files(&preview)[0]["generated_records"]["notes"], 1);
    let pruned = envelope_data(
        fixture.command().args([
            "folders",
            "prune",
            "notes",
            "--yes",
            "--revision",
            preview["revision"].as_str().unwrap(),
            "--format",
            "json",
        ]),
        "folders.prune",
        0,
    );
    assert_eq!(files(&pruned)[0]["file"]["status"], "pruned");
    assert_eq!(fixture.notes(), vec![manual.clone()]);
    let retained_source = fixture.source(source_id);
    assert_eq!(retained_source["id"], original_source["id"]);
    assert_eq!(retained_source["uri"], uri);
    assert_eq!(retained_source["normalized_uri"], uri);
    assert_eq!(retained_source["content"], original_source["content"]);
    let retained = envelope_data(
        fixture
            .command()
            .args(["inspect", manual_id, "--format", "json"]),
        "inspect",
        0,
    );
    assert_eq!(retained["content"], manual_content);
    assert_eq!(retained["provenance"]["source_id"], source_id);
    assert_eq!(retained["provenance"]["source_uri"], uri);
    assert!(retained["provenance"]["source_generation"].is_null());

    // Metadata retained for a manual note is not a new deletion candidate on
    // every run. A second confirmed empty prune leaves provenance intact.
    let repeated = fixture.prune_preview();
    assert!(files(&repeated).is_empty());
    let repeated_confirmation = envelope_data(
        fixture.command().args([
            "folders",
            "prune",
            "notes",
            "--yes",
            "--revision",
            repeated["revision"].as_str().unwrap(),
            "--format",
            "json",
        ]),
        "folders.prune",
        0,
    );
    assert!(files(&repeated_confirmation).is_empty());
    assert_eq!(fixture.source(source_id), retained_source);

    let archive = fixture.directory.path().join("pruned-manual-backup");
    let created = raw_json(
        fixture
            .command()
            .args(["backup", "create"])
            .arg(&archive)
            .args(["--format", "json"]),
    );
    assert_eq!(created["includes_embeddings"], false);
    assert_eq!(created["record_counts"]["note"], 1);
    assert_eq!(created["record_counts"]["source"], 1);
    let verified = raw_json(
        fixture
            .command()
            .args(["backup", "verify"])
            .arg(&archive)
            .args(["--format", "json"]),
    );
    assert_eq!(verified, created);
    let restored_db = fixture.directory.path().join("fresh-restored-database");
    let restored = raw_json(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["backup", "restore"])
            .arg(&archive)
            .args(["--format", "json"]),
    );
    assert_eq!(restored["includes_embeddings"], false);
    assert_eq!(restored["record_counts"], created["record_counts"]);
    let restored_manual = envelope_data(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["inspect", manual_id, "--format", "json"]),
        "inspect",
        0,
    );
    assert_eq!(restored_manual["content"], manual_content);
    assert_eq!(restored_manual["provenance"], retained["provenance"]);
    let restored_source = raw_json(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["sources", "show", source_id, "--format", "json"]),
    );
    assert_eq!(restored_source["id"], original_source["id"]);
    assert_eq!(restored_source["content"], original_source["content"]);
    let restored_preview = envelope_data(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["folders", "prune", "notes", "--format", "json"]),
        "folders.prune",
        0,
    );
    assert!(files(&restored_preview).is_empty());
    assert_eq!(fixture.provider.requests().len(), requests_before);

    // Restoring identical file contents must create generated notes again,
    // while reusing the retained source identity and leaving the manual copy.
    fixture.write_note("source.md", "originalsource");
    let reimported = envelope_data(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["sync", "notes", "--format", "json"]),
        "sync",
        0,
    );
    assert_eq!(files(&reimported).len(), 1);
    assert_eq!(files(&reimported)[0]["status"], "updated");
    assert_eq!(files(&reimported)[0]["source_id"], source_id);
    assert!(files(&reimported)[0]["generation"].as_u64().unwrap() > 1);
    let after_reimport = envelope_data(
        fixture
            .command()
            .arg("--db-path")
            .arg(&restored_db)
            .args(["notes", "list", "--limit", "100", "--format", "json"]),
        "notes.list",
        0,
    );
    let notes = after_reimport["notes"].as_array().unwrap();
    assert_eq!(notes.len(), 2);
    assert!(notes
        .iter()
        .any(|note| note["id"] == manual_id && note["content"] == manual_content));
    assert!(notes
        .iter()
        .any(|note| note["content"].as_str().unwrap().contains("originalsource")));
}
