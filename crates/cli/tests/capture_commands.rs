//! Real subprocesses, private temporary drafts/corpora, blocking editor doubles,
//! and deterministic HTTP providers. No live models or user state are touched.
use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::thread;
use std::time::Duration;

struct Provider {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    failure: Arc<Mutex<String>>,
    cleanup_failure: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    server: Option<thread::JoinHandle<()>>,
}

fn request(stream: &mut TcpStream) -> (String, String) {
    stream.set_nonblocking(false).unwrap();
    stream
        .set_read_timeout(Some(Duration::from_secs(3)))
        .unwrap();
    let mut bytes = Vec::new();
    let mut buffer = [0; 2048];
    let end = loop {
        let count = stream.read(&mut buffer).unwrap();
        assert!(count > 0);
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..end]);
    let line = headers.lines().next().unwrap().to_string();
    let length = headers
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    let received = bytes.len();
    bytes.resize(end + length, 0);
    if received < bytes.len() {
        stream.read_exact(&mut bytes[received..]).unwrap();
    }
    (line, String::from_utf8(bytes[end..].to_vec()).unwrap())
}

impl Provider {
    fn new(drafts: PathBuf) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = requests.clone();
        let failure = Arc::new(Mutex::new(String::new()));
        let failures = failure.clone();
        let cleanup_failure = Arc::new(AtomicBool::new(false));
        let cleanup = cleanup_failure.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let server = thread::spawn(move || {
            let mut vector = vec![0.0_f32; 1024];
            vector[0] = 1.0;
            while !stopping.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("fixture accept: {error}"),
                };
                let (line, body) = request(&mut stream);
                recorded.lock().unwrap().push(format!("{line} {body}"));
                let embedding = line.starts_with("POST /api/embed ")
                    || line.starts_with("POST /api/embeddings ");
                if embedding && cleanup.swap(false, Ordering::Relaxed) {
                    let path = fs::read_dir(&drafts)
                        .unwrap()
                        .next()
                        .unwrap()
                        .unwrap()
                        .path();
                    fs::remove_file(&path).unwrap();
                    fs::create_dir(&path).unwrap();
                }
                let failure = failures.lock().unwrap().clone();
                let fail = (failure == "embedding" && embedding)
                    || (failure == "extraction" && line.starts_with("POST /api/chat "))
                    || failure == "health";
                let (status, payload) = if fail {
                    (
                        "503 Service Unavailable",
                        json!({"error":"isolated provider failure"}),
                    )
                } else if line.starts_with("GET /api/tags ") {
                    (
                        "200 OK",
                        json!({"models":[{"name":"bge-m3:latest"},{"name":"phi4-mini:latest"}]}),
                    )
                } else if line.starts_with("POST /api/embed ") {
                    let payload: Value = serde_json::from_str(&body).unwrap();
                    let inputs = payload["input"].as_array().map_or(1, Vec::len);
                    ("200 OK", json!({"embeddings":vec![&vector; inputs]}))
                } else if line.starts_with("POST /api/embeddings ") {
                    ("200 OK", json!({"embedding": &vector}))
                } else if line.starts_with("POST /api/chat ") {
                    (
                        "200 OK",
                        json!({"message":{"role":"assistant","content":"{\"entities\":[],\"relationships\":[]}"},"done":true}),
                    )
                } else {
                    ("200 OK", json!({}))
                };
                let payload = payload.to_string();
                let response = format!("HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}", payload.len());
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        Self {
            endpoint,
            requests,
            failure,
            cleanup_failure,
            stop,
            server: Some(server),
        }
    }
    fn fail(&self, stage: &str) {
        *self.failure.lock().unwrap() = stage.into();
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.server.take().unwrap().join().unwrap();
    }
}

struct Fixture {
    temp: tempfile::TempDir,
    config: PathBuf,
    db: PathBuf,
    drafts: PathBuf,
    editor: PathBuf,
    provider: Provider,
}
impl Fixture {
    fn new() -> Self {
        let temp = tempfile::tempdir().unwrap();
        let config = temp.path().join("config café.toml");
        let db = temp.path().join("knowledge.surreal");
        let drafts = temp.path().join("knowledge.surreal.drafts");
        let provider = Provider::new(drafts.clone());
        let mut settings = RuntimeConfig::default();
        settings.database.path = db.clone();
        settings.inference.embedding_provider = "ollama".into();
        settings.inference.extraction_provider = "ollama".into();
        settings.inference.embedding_url = provider.endpoint.clone();
        settings.inference.extraction_url = provider.endpoint.clone();
        settings.inference.embedding_model = "bge-m3".into();
        settings.inference.extraction_model = "phi4-mini".into();
        settings.inference.timeout_secs = 2;
        settings.inference.ollama_timeout_secs = 2;
        settings.inference.retry_attempts = 1;
        settings.inference.cache_enabled = false;
        settings.logging.level = "error".into();
        fs::write(&config, settings.redacted_toml().unwrap()).unwrap();
        let editor = temp.path().join("blocking editor café.sh");
        fs::write(&editor, "#!/bin/sh\nmode=$1\npayload=$2\ndraft=$3\nprintf 'EDITOR_DIAGNOSTIC\\n'\ncase \"$mode\" in\n unchanged) exit 0 ;;\n unchanged-cleanup-failure) chmod 500 \"${draft%/*}\"; exit 0 ;;\n cancel) printf '%s' \"$payload\" > \"$draft\"; exit 7 ;;\n binary) printf '\\377' > \"$draft\"; exit 0 ;;\n symlink) rm \"$draft\"; ln -s \"$payload\" \"$draft\" ;;\n directory) rm \"$draft\"; mkdir \"$draft\" ;;\n missing) rm \"$draft\" ;;\n replace) printf '%s' \"$payload\" > \"$draft.replacement\"; mv \"$draft.replacement\" \"$draft\" ;;\nesac\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&editor, fs::Permissions::from_mode(0o700)).unwrap();
        }
        Self {
            temp,
            config,
            db,
            drafts,
            editor,
            provider,
        }
    }
    fn cmd(&self) -> Command {
        let mut command = Command::cargo_bin("graphrag").unwrap();
        command
            .env_clear()
            .env("HOME", self.temp.path().join("home"))
            .env("XDG_CONFIG_HOME", self.temp.path().join("xdg"))
            .current_dir(self.temp.path())
            .args(["--config"])
            .arg(&self.config)
            .timeout(Duration::from_secs(20));
        command
    }
    fn editor(&self, command: &mut Command, mode: &str, payload: &str) {
        command
            .arg("--editor")
            .arg("--editor-command")
            .arg(&self.editor)
            .arg("--editor-arg")
            .arg(mode)
            .arg("--editor-arg")
            .arg(payload);
    }
    fn capture(&self, content: &str) -> String {
        let output = self
            .cmd()
            .args(["capture", "--stdin", "--format", "json"])
            .write_stdin(content)
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        data(&output)["id"].as_str().unwrap().into()
    }
    fn show(&self, id: &str) -> Value {
        data(
            &self
                .cmd()
                .args(["notes", "show", id, "--format", "json"])
                .assert()
                .success()
                .get_output()
                .stdout,
        )
    }
    fn notes(&self) -> Value {
        data(
            &self
                .cmd()
                .args(["notes", "list", "--format", "json"])
                .assert()
                .success()
                .get_output()
                .stdout,
        )
    }
    fn draft_paths(&self) -> Vec<PathBuf> {
        fs::read_dir(&self.drafts)
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default()
    }
    fn replay(&self, recovery: &str) -> Command {
        let mut command = Command::new("/bin/sh");
        let bin = Path::new(env!("CARGO_BIN_EXE_graphrag")).parent().unwrap();
        command
            .env_clear()
            .env("HOME", self.temp.path().join("home"))
            .env("PATH", format!("{}:/usr/bin:/bin", bin.display()))
            .current_dir(std::env::temp_dir())
            .args(["-c", recovery])
            .timeout(Duration::from_secs(20));
        command
    }
}
fn data(stdout: &[u8]) -> Value {
    let envelope: Value = serde_json::from_slice(stdout).unwrap();
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["success"], true);
    envelope["data"].clone()
}
fn recovery(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .find_map(|line| line.strip_prefix("Recover: "))
        .unwrap()
        .into()
}

#[test]
fn multiline_capture_json_and_jsonl_preserve_content_and_actionable_ids() {
    let f = Fixture::new();
    let content = "Planning café\nSecond line with `quotes`, # and 100%.\n";
    let id = f.capture(content);
    assert_eq!(f.show(&id)["content"], content);
    assert!(f.show(&id)["source_id"].is_null());
    assert!(f.draft_paths().is_empty());
    let output = f
        .cmd()
        .args(["capture", "next multiline\nentry", "--format", "jsonl"])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    assert_eq!(String::from_utf8_lossy(&output).lines().count(), 1);
    assert_eq!(data(&output)["status"], "created");
}

#[test]
fn blocking_editor_saves_by_rename_with_literal_arguments_and_clean_machine_stdout() {
    let f = Fixture::new();
    let sentinel = f.temp.path().join("must-not-exist");
    let content = format!("Editor café\n$(touch {})\n", sentinel.display());
    let mut command = f.cmd();
    command.args(["capture", "--format", "json"]);
    f.editor(&mut command, "replace", &content);
    let output = command.assert().success().get_output().clone();
    assert!(String::from_utf8_lossy(&output.stderr).contains("EDITOR_DIAGNOSTIC"));
    let id = data(&output.stdout)["id"].as_str().unwrap().to_string();
    assert_eq!(f.show(&id)["content"], content);
    assert!(!sentinel.exists());
    assert!(f.draft_paths().is_empty());
}

#[test]
fn unchanged_capture_editor_is_offline_and_does_not_open_a_database() {
    let f = Fixture::new();
    let mut command = f.cmd();
    command.args(["capture", "--title", "Ignored", "--format", "json"]);
    f.editor(&mut command, "unchanged", "");
    let output = command.assert().success().get_output().clone();
    assert_eq!(data(&output.stdout)["status"], "unchanged");
    assert!(!f.db.exists());
    assert_eq!(f.provider.count(), 0);
    assert!(f.draft_paths().is_empty());
}

#[cfg(unix)]
#[test]
fn unchanged_session_machine_output_reports_cleanup_success_and_retained_draft_warnings() {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    struct RestorePermissions(PathBuf);
    impl Drop for RestorePermissions {
        fn drop(&mut self) {
            let _ = fs::set_permissions(&self.0, fs::Permissions::from_mode(0o700));
        }
    }
    for format in ["json", "jsonl"] {
        for editing in [false, true] {
            for cleanup_failure in [false, true] {
                let f = Fixture::new();
                // Privileged root bypasses directory permissions and cannot
                // exercise this POSIX permission-denied cleanup scenario.
                if cleanup_failure && fs::metadata(f.temp.path()).unwrap().uid() == 0 {
                    continue;
                }
                let _restore_permissions = RestorePermissions(f.drafts.clone());
                let body = "Unchanged body retained exactly";
                let id = editing.then(|| f.capture(body));
                let before = id.as_deref().map(|id| f.show(id));
                let count = f.provider.count();
                let mut command = f.cmd();
                if let Some(id) = &id {
                    command.args(["notes", "edit", id, "--detach", "--title", "Ignored"]);
                } else {
                    command.args(["capture", body]);
                }
                command.args(["--format", format]);
                f.editor(
                    &mut command,
                    if cleanup_failure {
                        "unchanged-cleanup-failure"
                    } else {
                        "unchanged"
                    },
                    "",
                );
                let output = command.assert().success().get_output().clone();
                let result = data(&output.stdout);
                assert_eq!(result["status"], "unchanged");
                assert!(result["recovery_command"].is_null());
                let stderr = String::from_utf8_lossy(&output.stderr);
                assert!(!stderr.contains("Recover:"));
                if cleanup_failure {
                    let path = PathBuf::from(result["draft_path"].as_str().unwrap());
                    assert_eq!(path.parent(), Some(f.drafts.as_path()));
                    assert_eq!(fs::read_to_string(path).unwrap(), body);
                    assert!(stderr.contains("cleanup failed"));
                } else {
                    assert!(result["draft_path"].is_null());
                    assert!(f.draft_paths().is_empty());
                    assert!(!stderr.contains("cleanup failed"));
                }
                if let Some(id) = &id {
                    assert_eq!(f.show(id), before.unwrap());
                    assert_eq!(f.notes()["notes"].as_array().unwrap().len(), 1);
                    assert_eq!(result["id"], *id);
                    assert_eq!(result["detached"], false);
                } else {
                    assert!(result["id"].is_null());
                    assert!(!f.db.exists());
                }
                assert_eq!(f.provider.count(), count);
                if format == "jsonl" {
                    assert_eq!(String::from_utf8_lossy(&output.stdout).lines().count(), 1);
                }
            }
        }
    }
}

#[test]
fn cancelled_capture_retains_private_draft_and_replays_option_like_metadata_from_another_cwd() {
    let f = Fixture::new();
    let mut command = f.cmd();
    command.args([
        "capture",
        "--title=-Title",
        "--tags=-urgent",
        "--format",
        "json",
    ]);
    f.editor(
        &mut command,
        "cancel",
        "Cancelled café\nRecover this multiline draft.",
    );
    let output = command.assert().success().get_output().clone();
    let result = data(&output.stdout);
    assert_eq!(result["status"], "cancelled");
    assert!(result["id"].is_null());
    assert!(!f.db.exists());
    assert_eq!(f.provider.count(), 0);
    let path = PathBuf::from(result["draft_path"].as_str().unwrap());
    assert_eq!(path.parent(), Some(f.drafts.as_path()));
    assert!(fs::read_to_string(&path)
        .unwrap()
        .contains("Cancelled café"));
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(
            fs::metadata(&f.drafts).unwrap().permissions().mode() & 0o777,
            0o700
        );
    }
    let recovered = f
        .replay(result["recovery_command"].as_str().unwrap())
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let id = data(&recovered)["id"].as_str().unwrap().to_string();
    let note = f.show(&id);
    assert_eq!(note["title"], "-Title");
    assert_eq!(note["tags"], json!(["-urgent"]));
    assert!(
        path.exists(),
        "retry must not consume the retained input file"
    );
}

#[test]
fn invalid_and_non_utf8_input_are_preserved_before_any_provider_request() {
    for bytes in [b" \n\t".to_vec(), vec![0xff, 0xfe]] {
        let f = Fixture::new();
        let output = f
            .cmd()
            .args(["capture", "--stdin", "--format", "json"])
            .write_stdin(bytes.clone())
            .assert()
            .code(2)
            .get_output()
            .clone();
        assert!(output.stdout.is_empty());
        let paths = f.draft_paths();
        assert_eq!(paths.len(), 1);
        assert_eq!(fs::read(&paths[0]).unwrap(), bytes);
        assert_eq!(f.provider.count(), 0);
        assert!(!f.db.exists());
        assert!(recovery(&output.stderr).contains("--content-file"));
    }
}

#[cfg(unix)]
#[test]
fn non_utf8_recovery_directory_is_rejected_before_editor_or_provider_work() {
    use std::os::unix::ffi::OsStringExt;
    let f = Fixture::new();
    let invalid_directory = f
        .temp
        .path()
        .join(std::ffi::OsString::from_vec(b"drafts-\xff".to_vec()));
    let mut command = f.cmd();
    command
        .args(["capture", "--draft-dir"])
        .arg(&invalid_directory);
    f.editor(&mut command, "replace", "Must not launch");
    let output = command.assert().code(2).get_output().clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("paths must be valid UTF-8"));
    assert!(!stderr.contains("EDITOR_DIAGNOSTIC"));
    assert!(!stderr.contains("Recover:"));
    assert!(!f.db.exists());
    assert!(!invalid_directory.exists());
    assert_eq!(f.provider.count(), 0);
}

#[test]
fn recoverable_commands_reject_memory_before_input_editor_or_provider_work() {
    let f = Fixture::new();
    for args in [
        vec!["--memory", "capture", "--stdin"],
        vec!["--memory", "capture", "--editor"],
        vec!["--memory", "notes", "edit", "note:missing", "--editor"],
        vec!["--memory", "notes", "edit", "note:missing", "--stdin"],
        vec![
            "--memory",
            "notes",
            "edit",
            "note:missing",
            "--content-file",
            "missing-input.md",
        ],
    ] {
        let output = f.cmd().args(args).assert().code(2).get_output().clone();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("require a persistent database"));
        assert!(!stderr.contains("Recover:"));
        assert!(!stderr.contains("Editing draft:"));
        assert!(output.stdout.is_empty());
        assert!(!f.db.exists());
        assert!(f.draft_paths().is_empty());
        assert_eq!(f.provider.count(), 0);
    }
    f.cmd()
        .args(["--memory", "notes", "list", "--format", "json"])
        .assert()
        .success();
    f.cmd()
        .args([
            "--memory",
            "notes",
            "edit",
            "note:missing",
            "--title",
            "Legacy metadata",
        ])
        .assert()
        .code(3);
    assert!(!f.db.exists());
    assert_eq!(f.provider.count(), 0);
}

#[cfg(unix)]
#[test]
fn rejected_editor_paths_never_advertise_recovery_or_expose_symlink_targets() {
    for mode in ["symlink", "directory", "missing"] {
        let f = Fixture::new();
        let id = f.capture("Previous note remains intact");
        let before = f.show(&id);
        let count = f.provider.count();
        let secret = f.temp.path().join("unrelated-secret.txt");
        fs::write(&secret, "Unrelated target must never reach inference").unwrap();
        for editing in [false, true] {
            let mut command = f.cmd();
            if editing {
                command.args(["notes", "edit", &id]);
            } else {
                command.arg("capture");
            }
            f.editor(&mut command, mode, secret.to_str().unwrap());
            let output = command.assert().failure().get_output().clone();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(!stderr.contains("Recover:"), "{stderr}");
            assert!(stderr.contains("Recovery command withheld:"));
            assert!(stderr.contains("Inspect the path"));
            assert!(output.stdout.is_empty());
            assert_eq!(f.show(&id), before);
            assert_eq!(f.provider.count(), count);
            assert_eq!(
                fs::read_to_string(&secret).unwrap(),
                "Unrelated target must never reach inference"
            );
        }
    }
}

#[test]
fn editor_launch_and_invalid_editor_output_preserve_recovery_drafts() {
    let f = Fixture::new();
    let missing = f.temp.path().join("missing editor");
    let output = f
        .cmd()
        .args(["capture", "seeded content", "--editor", "--editor-command"])
        .arg(missing)
        .assert()
        .code(1)
        .get_output()
        .clone();
    assert_eq!(
        fs::read_to_string(&f.draft_paths()[0]).unwrap(),
        "seeded content"
    );
    assert!(recovery(&output.stderr).contains("capture"));
    assert_eq!(f.provider.count(), 0);
    let f = Fixture::new();
    let mut command = f.cmd();
    command.arg("capture");
    f.editor(&mut command, "binary", "");
    command.assert().code(2);
    assert_eq!(fs::read(&f.draft_paths()[0]).unwrap(), vec![0xff]);
    assert_eq!(f.provider.count(), 0);
}

#[test]
fn capture_provider_failures_leave_no_note_or_source_and_preserve_drafts() {
    for stage in ["health", "embedding", "extraction"] {
        let f = Fixture::new();
        f.provider.fail(stage);
        let output = f
            .cmd()
            .args(["capture", "failed capture body", "--format", "json"])
            .assert()
            .code(1)
            .get_output()
            .clone();
        assert!(output.stdout.is_empty());
        assert_eq!(
            fs::read_to_string(&f.draft_paths()[0]).unwrap(),
            "failed capture body"
        );
        assert!(f.notes()["notes"].as_array().unwrap().is_empty());
        let sources = f
            .cmd()
            .args(["sources", "list", "--format", "json"])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        assert!(serde_json::from_slice::<Value>(&sources)
            .unwrap()
            .as_array()
            .unwrap()
            .is_empty());
        assert!(recovery(&output.stderr).contains("capture"));
    }
}

#[test]
fn unchanged_and_cancelled_note_edit_do_not_apply_metadata_or_detach_flags() {
    let f = Fixture::new();
    let id = f.capture("Original editor note");
    let before = f.show(&id);
    let count = f.provider.count();
    for mode in ["unchanged", "cancel"] {
        let mut command = f.cmd();
        command.args([
            "notes",
            "edit",
            &id,
            "--title",
            "Ignored metadata",
            "--tags",
            "ignored",
            "--detach",
            "--format",
            "json",
        ]);
        f.editor(&mut command, mode, "Unsaved changed draft");
        let output = command.assert().success().get_output().clone();
        assert_eq!(
            data(&output.stdout)["status"],
            if mode == "cancel" {
                "cancelled"
            } else {
                "unchanged"
            }
        );
        assert_eq!(f.show(&id), before);
        assert_eq!(f.notes()["notes"].as_array().unwrap().len(), 1);
        assert_eq!(f.provider.count(), count);
    }
}

#[test]
fn changed_editor_edit_saves_and_failed_refresh_retains_previous_note_and_retry() {
    let f = Fixture::new();
    let id = f.capture("Original manual note");
    let mut command = f.cmd();
    command.args(["notes", "edit", &id, "--format", "json"]);
    f.editor(&mut command, "replace", "Revised multiline\nmanual note");
    command.assert().success();
    let before = f.show(&id);
    assert_eq!(before["content"], "Revised multiline\nmanual note");
    f.provider.fail("extraction");
    let mut command = f.cmd();
    command.args([
        "notes",
        "edit",
        &id,
        "--title=-RetryTitle",
        "--tags=-retry",
        "--format",
        "json",
    ]);
    f.editor(&mut command, "replace", "Failed replacement café");
    let output = command.assert().code(1).get_output().clone();
    assert_eq!(f.show(&id), before);
    assert_eq!(
        fs::read_to_string(&f.draft_paths()[0]).unwrap(),
        "Failed replacement café"
    );
    f.provider.fail("");
    f.replay(&recovery(&output.stderr)).assert().success();
    assert_eq!(f.show(&id)["title"], "-RetryTitle");
    assert_eq!(f.show(&id)["content"], "Failed replacement café");
}

#[test]
fn imported_editor_edits_offer_source_and_detach_without_editor_or_provider_calls() {
    let f = Fixture::new();
    let source = f.temp.path().join("source.md");
    fs::write(&source, "# Imported\nImported chunk content.").unwrap();
    f.cmd().arg("import").arg(&source).assert().success();
    let notes = f.notes();
    let hit: graphrag_db::repository::SearchResult =
        serde_json::from_value(notes["notes"][0].clone()).unwrap();
    let id = graphrag_core::record_id_to_string(&hit.id);
    let count = f.provider.count();
    let mut command = f.cmd();
    command.args(["notes", "edit", &id, "--format", "json"]);
    f.editor(&mut command, "replace", "Must not be applied");
    let output = command.assert().code(2).get_output().clone();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(" open "));
    assert!(stderr.contains("--detach --editor"));
    assert!(!stderr.contains("EDITOR_DIAGNOSTIC"));
    assert_eq!(f.provider.count(), count);
    assert!(f.draft_paths().is_empty());
    let mut command = f.cmd();
    command.args(["notes", "edit", &id, "--detach", "--format", "json"]);
    f.editor(&mut command, "replace", "Detached manual content");
    command.assert().success();
    assert_eq!(f.notes()["notes"].as_array().unwrap().len(), 2);
}

#[test]
fn imported_chat_notes_require_detach_and_detached_copies_remain_editable() {
    for mode in ["message", "qa", "fallback"] {
        let f = Fixture::new();
        let export = f.temp.path().join("chat.json");
        let messages = if mode == "fallback" {
            json!([])
        } else {
            json!([
                {"uuid":"question", "sender":"human", "text":"How should Atlas planning support daily use?"},
                {"uuid":"answer", "sender":"assistant", "text":"Atlas planning should keep an explicit daily record and reusable project context."}
            ])
        };
        fs::write(
            &export,
            json!([{
                "uuid": format!("ownership-{mode}"), "name":"Atlas ownership chat",
                "summary":"Atlas chat summary preserves imported conversation context.",
                "created_at":"2026-01-01T00:00:00Z", "updated_at":"2026-01-01T00:00:00Z",
                "chat_messages":messages
            }])
            .to_string(),
        )
        .unwrap();
        f.cmd()
            .arg("import-chats")
            .arg(&export)
            .args([
                "--mode",
                if mode == "fallback" { "message" } else { mode },
                "--skip-extraction",
            ])
            .assert()
            .success();
        let notes = f.notes();
        let originals = notes["notes"].as_array().unwrap();
        assert!(originals.len() >= 2, "{mode}: {notes}");
        for hit in originals {
            let hit: graphrag_db::repository::SearchResult =
                serde_json::from_value(hit.clone()).unwrap();
            let id = graphrag_core::record_id_to_string(&hit.id);
            let before = f.show(&id);
            assert!(before.get("source_generation").is_none());
            let count = f.provider.count();
            let mut command = f.cmd();
            command.args(["notes", "edit", &id, "--format", "json"]);
            f.editor(&mut command, "replace", "Must not overwrite imported chat");
            let output = command.assert().code(2).get_output().clone();
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(stderr.contains(" inspect "), "{stderr}");
            assert!(stderr.contains("--detach --editor"), "{stderr}");
            assert!(!stderr.contains("EDITOR_DIAGNOSTIC"));
            assert!(f.draft_paths().is_empty());
            f.cmd()
                .args(["notes", "edit", &id, "--title", "Must not change"])
                .assert()
                .code(2);
            f.cmd()
                .args(["notes", "edit", &id, "--stdin"])
                .write_stdin("Must not overwrite imported chat")
                .assert()
                .code(2);
            assert_eq!(f.provider.count(), count);
            assert_eq!(f.show(&id), before);

            let mut command = f.cmd();
            command.args(["notes", "edit", &id, "--detach", "--format", "json"]);
            f.editor(&mut command, "replace", "Detached chat manual copy");
            let output = command.assert().success().get_output().stdout.clone();
            let result = data(&output);
            assert_eq!(result["detached"], true);
            let persisted = f.notes();
            let detached = persisted["notes"]
                .as_array()
                .unwrap()
                .iter()
                .find(|note| note["id"] == result["note"]["id"])
                .unwrap();
            let detached: graphrag_db::repository::SearchResult =
                serde_json::from_value(detached.clone()).unwrap();
            let detached_id = graphrag_core::record_id_to_string(&detached.id);
            assert_ne!(detached_id, id);
            assert_eq!(result["note"]["source_id"], before["source_id"]);
            assert_eq!(result["note"]["tags"], before["tags"]);
            let inspected = data(
                &f.cmd()
                    .args(["inspect", &detached_id, "--format", "json"])
                    .assert()
                    .success()
                    .get_output()
                    .stdout,
            );
            assert!(inspected["provenance"]["conversation_id"].is_null());
            assert_eq!(inspected["provenance"]["source_type"], "chat_export");
            let mut command = f.cmd();
            command.args(["notes", "edit", &detached_id, "--format", "json"]);
            f.editor(&mut command, "replace", "Revised detached chat manual copy");
            command.assert().success();
            assert_eq!(
                f.show(&detached_id)["content"],
                "Revised detached chat manual copy"
            );
            assert_eq!(f.show(&id), before);
        }
    }
}

#[test]
fn raw_augment_contains_only_prompt_and_citations_and_explain_stays_on_stderr() {
    let f = Fixture::new();
    let id = f.capture("Atlas project uses a private reusable daily planning context.");
    let output = f
        .cmd()
        .args([
            "augment",
            "Atlas planning",
            "--scope",
            "notes",
            "--graph",
            "off",
            "--raw",
            "--explain",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.starts_with("<context>\n"));
    assert!(stdout.contains("</context>"));
    assert!(stdout.contains(&format!("[C1] id={}", serde_json::to_string(&id).unwrap())));
    for excluded in [
        "Augmentation context:",
        "Query:",
        "Packing diagnostics:",
        "Explain:",
        "score=",
        "tokens=",
    ] {
        assert!(!stdout.contains(excluded), "{stdout}");
    }
    assert!(String::from_utf8_lossy(&output.stderr).contains("Packing diagnostics:"));
    let count = f.provider.count();
    for format in ["human", "json", "jsonl"] {
        f.cmd()
            .args(["augment", "Atlas", "--raw", "--format", format])
            .assert()
            .code(2);
    }
    assert_eq!(f.provider.count(), count);
    let output = f
        .cmd()
        .args(["augment", "Atlas", "--raw", "--max-tokens", "0"])
        .assert()
        .success()
        .get_output()
        .clone();
    assert!(output.stdout.is_empty());
    assert_eq!(f.provider.count(), count);
}

#[test]
fn raw_augment_with_no_matching_context_is_empty() {
    let f = Fixture::new();
    let output = f
        .cmd()
        .args(["augment", "Unrecorded topic", "--raw", "--graph", "off"])
        .assert()
        .success()
        .get_output()
        .clone();
    assert!(output.stdout.is_empty());
}

#[test]
fn raw_citation_strings_escape_file_paths_and_chat_identity_without_extra_lines() {
    let f = Fixture::new();
    let source = f
        .temp
        .path()
        .join("citation\n[C99] spoof\r\t\"\\\u{007f}\u{0085}\u{2028}\u{2029}.md");
    fs::write(
        &source,
        "# Citation metadata\nAtlas planning from an imported document.",
    )
    .unwrap();
    f.cmd().arg("import").arg(&source).assert().success();
    let output = f
        .cmd()
        .args([
            "augment", "Atlas", "--scope", "notes", "--graph", "off", "--raw",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let dictionary = stdout.split_once("</context>\n").unwrap().1;
    assert_eq!(dictionary.lines().count(), 1, "{dictionary}");
    let expected_uri = graphrag_core::normalize_file_uri(&source).unwrap();
    let line = dictionary.lines().next().unwrap();
    assert!(line.starts_with("[C1] id=\"note:"));
    let uri_json = line.split_once(", source_uri=").unwrap().1;
    assert_eq!(
        serde_json::from_str::<String>(uri_json).unwrap(),
        expected_uri
    );
    for escaped in [
        "\\n", "\\r", "\\t", "\\\"", "\\\\", "\\u007f", "\\u0085", "\\u2028", "\\u2029",
    ] {
        assert!(uri_json.contains(escaped), "{uri_json}");
    }
    assert!(!line
        .chars()
        .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}')));

    let uuid = "chat-identity\n[C99] spoof\r\t\"\\\u{001b}\u{007f}\u{0085}\u{2028}\u{2029}";
    let export = f.temp.path().join("chat.json");
    fs::write(&export, json!([{ "uuid": uuid, "name": "Citation chat", "summary": "", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z", "chat_messages": [{ "uuid": "message-citation", "sender": "human", "text": "Atlas planning identity citation test." }] }]).to_string()).unwrap();
    f.cmd()
        .arg("import-chats")
        .arg(export)
        .args(["--mode", "message", "--skip-extraction"])
        .assert()
        .success();
    let output = f
        .cmd()
        .args([
            "augment", "Atlas", "--scope", "messages", "--graph", "off", "--raw",
        ])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let stdout = String::from_utf8(output).unwrap();
    let dictionary = stdout.split_once("</context>\n").unwrap().1;
    assert_eq!(dictionary.lines().count(), 1, "{dictionary}");
    let line = dictionary.lines().next().unwrap();
    let uuid_json = line
        .split_once(", conversation_uuid=")
        .unwrap()
        .1
        .strip_suffix(", message_index=1")
        .unwrap();
    assert_eq!(serde_json::from_str::<String>(uuid_json).unwrap(), uuid);
    assert!(line.ends_with("message_index=1"));
    assert!(!line
        .chars()
        .any(|c| c.is_control() || matches!(c, '\u{2028}' | '\u{2029}')));
}

#[test]
fn invalid_note_file_edit_retains_bytes_and_previous_note_without_inference() {
    let f = Fixture::new();
    let id = f.capture("Previous note stays intact");
    let previous = f.show(&id);
    let count = f.provider.count();
    let content_file = f.temp.path().join("invalid-note.md");
    fs::write(&content_file, [0xff, 0xfe]).unwrap();
    let output = f
        .cmd()
        .args(["notes", "edit", &id, "--content-file"])
        .arg(content_file)
        .assert()
        .code(2)
        .get_output()
        .clone();
    assert!(output.stdout.is_empty());
    assert_eq!(f.show(&id), previous);
    assert_eq!(f.provider.count(), count);
    assert_eq!(fs::read(&f.draft_paths()[0]).unwrap(), [0xff, 0xfe]);
    assert!(recovery(&output.stderr).contains("notes edit"));
}

#[test]
fn draft_cleanup_failure_after_commit_keeps_success_and_note_id_without_retry_guidance() {
    let f = Fixture::new();
    f.provider.cleanup_failure.store(true, Ordering::Relaxed);
    let output = f
        .cmd()
        .args([
            "capture",
            "Successfully committed despite cleanup failure",
            "--format",
            "json",
        ])
        .assert()
        .success()
        .get_output()
        .clone();
    let result = data(&output.stdout);
    let id = result["id"].as_str().unwrap();
    assert!(result["draft_path"].as_str().is_some());
    assert!(result["recovery_command"].is_null());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains(id));
    assert!(stderr.contains("cleanup failed"));
    assert!(!stderr.contains("Recover:"));
    assert_eq!(
        f.show(id)["content"],
        "Successfully committed despite cleanup failure"
    );
    assert_eq!(f.notes()["notes"].as_array().unwrap().len(), 1);
}
