//! Real persistent workspace sessions with private corpora and deterministic
//! loopback providers. Fixture processes exit before the workspace takes the
//! embedded database lock; no installed model, editor, or user corpus is used.
use assert_cmd::Command;
use graphrag_config::RuntimeConfig;
use graphrag_core::{
    normalize_file_uri, normalized_content_hash, record_id_to_string, Note, SourceType,
};
use graphrag_db::{init_persistent, Repository};
use serde_json::{json, Value};
use std::fs;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::process::{Child, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    mpsc, Arc, Mutex,
};
use std::thread;
use std::time::{Duration, Instant};

#[test]
fn workspace_fixture_process() {
    let Some(path) = std::env::var_os("GRAPHRAG_WORKSPACE_FIXTURE") else {
        return;
    };
    let directory = PathBuf::from(path);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let seed = runtime.block_on(async {
        let repo = Repository::new(init_persistent(directory.join("corpus.surreal")).await.unwrap());
        let source_path = directory.join("source $(touch injected-marker) `id` #100% café 日本語 \u{1b}[2J.md");
        let content = "# Workspace source\n\nsourcetoken cafelexeme original source body 🦀.\n";
        fs::write(&source_path, content).unwrap();
        let source_path = source_path.canonicalize().unwrap();
        let mut source = repo.begin_file_import(SourceType::Markdown, "Workspace source".into(), normalize_file_uri(&source_path).unwrap(), content.into(), normalized_content_hash(content), false).await.unwrap().source;
        let imported = repo.create_note(Note::new(content).with_title("Source café 日本語").with_source(source.id.clone().unwrap()).with_source_generation(source.generation)).await.unwrap();
        repo.complete_file_import(&mut source).await.unwrap();
        let manual = repo.create_note(Note::new("manualtoken cafelexeme manual workspace context.").with_title("Manual workspace note")).await.unwrap();
        let proposal = repo.upsert_gardener_proposal(imported.id.as_ref().unwrap(), manual.id.as_ref().unwrap(), 0.9, "Shared workspace evidence".into(), None, None).await.unwrap();
        json!({"source_note":record_id_to_string(imported.id.as_ref().unwrap()),"manual_note":record_id_to_string(manual.id.as_ref().unwrap()),"proposal":record_id_to_string(proposal.id.as_ref().unwrap()),"source_path":source_path,"source_content":content})
    });
    fs::write(
        directory.join("seed.json"),
        serde_json::to_vec(&seed).unwrap(),
    )
    .unwrap();
}

fn read_request(stream: &mut TcpStream) -> Option<(String, String)> {
    let mut bytes = Vec::new();
    let mut buffer = [0; 2048];
    let end = loop {
        // A cancelled request can close or reset before all headers/body bytes
        // arrive. Bounded socket reads also let fixture cleanup finish safely.
        let count = stream.read(&mut buffer).ok()?;
        if count == 0 {
            return None;
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(end) = bytes.windows(4).position(|part| part == b"\r\n\r\n") {
            break end + 4;
        }
    };
    let headers = String::from_utf8_lossy(&bytes[..end]);
    let line = headers.lines().next()?.to_owned();
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
        stream.read_exact(&mut bytes[received..]).ok()?;
    }
    Some((line, String::from_utf8(bytes[end..].to_vec()).ok()?))
}

struct Provider {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    mode: Arc<Mutex<String>>,
    entered: Arc<AtomicBool>,
    release: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    server: Option<thread::JoinHandle<()>>,
}
impl Provider {
    fn new() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let mode = Arc::new(Mutex::new("ok".to_owned()));
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let (recorded, modes, blocked, released, stopping) = (
            requests.clone(),
            mode.clone(),
            entered.clone(),
            release.clone(),
            stop.clone(),
        );
        let server = thread::spawn(move || {
            let vector = vec![1.0_f32; 1024];
            while !stopping.load(Ordering::Acquire) {
                let (mut stream, _) = match listener.accept() {
                    Ok(pair) => pair,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                    Err(error) => panic!("provider accept failed: {error}"),
                };
                // On macOS an accepted socket inherits O_NONBLOCK from the
                // listener. Normalize it before reading a real HTTP request.
                stream.set_nonblocking(false).unwrap();
                stream
                    .set_read_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(3)))
                    .unwrap();
                let Some((line, body)) = read_request(&mut stream) else {
                    continue;
                };
                recorded.lock().unwrap().push(format!("{line} {body}"));
                let mode = modes.lock().unwrap().clone();
                if mode == "block" && line.starts_with("POST ") {
                    blocked.store(true, Ordering::Release);
                    let deadline = Instant::now() + Duration::from_secs(15);
                    while !released.load(Ordering::Acquire)
                        && !stopping.load(Ordering::Acquire)
                        && Instant::now() < deadline
                    {
                        thread::sleep(Duration::from_millis(2));
                    }
                }
                let (status, payload) = if mode == "fail" {
                    (
                        "503 Service Unavailable",
                        json!({"error":"isolated workspace provider failure"}),
                    )
                } else if line.starts_with("GET /api/tags ") {
                    (
                        "200 OK",
                        json!({"models":[{"name":"bge-m3:latest"},{"name":"phi4-mini:latest"}]}),
                    )
                } else if line.starts_with("POST /api/embed ") {
                    let input: Value = serde_json::from_str(&body).unwrap();
                    (
                        "200 OK",
                        json!({"embeddings":vec![&vector;input["input"].as_array().map_or(1,Vec::len)]}),
                    )
                } else if line.starts_with("POST /api/embeddings ") {
                    ("200 OK", json!({"embedding":vector}))
                } else if line.starts_with("POST /api/chat ") {
                    (
                        "200 OK",
                        json!({"message":{"role":"assistant","content":"{\"entities\":[],\"relationships\":[]}"},"done":true}),
                    )
                } else {
                    (
                        "404 Not Found",
                        json!({"error":"unexpected workspace provider route"}),
                    )
                };
                let payload = payload.to_string();
                // Dropped read requests are a normal cancellation outcome.
                let _ = write!(stream,"HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",payload.len());
            }
        });
        Self {
            endpoint,
            requests,
            mode,
            entered,
            release,
            stop,
            server: Some(server),
        }
    }
    fn mode(&self, mode: &str) {
        *self.mode.lock().unwrap() = mode.into();
    }
    fn count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }
    fn wait_blocked(&self) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !self.entered.load(Ordering::Acquire) {
            assert!(
                Instant::now() < deadline,
                "workspace never reached blocked provider"
            );
            thread::sleep(Duration::from_millis(5));
        }
    }
}
impl Drop for Provider {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.release.store(true, Ordering::Release);
        let result = self.server.take().unwrap().join();
        if !thread::panicking() {
            result.expect("workspace provider fixture failed");
        }
    }
}

struct Fixture {
    directory: tempfile::TempDir,
    config: PathBuf,
    provider: Provider,
    seed: Value,
}
impl Fixture {
    fn new(seed: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let config = directory.path().join("workspace config café.toml");
        let provider = Provider::new();
        let mut settings = RuntimeConfig::default();
        settings.database.path = directory.path().join("corpus.surreal");
        settings.inference.embedding_provider = "ollama".into();
        settings.inference.extraction_provider = "ollama".into();
        settings.inference.embedding_url = provider.endpoint.clone();
        settings.inference.extraction_url = provider.endpoint.clone();
        settings.inference.embedding_model = "bge-m3".into();
        settings.inference.extraction_model = "phi4-mini".into();
        settings.inference.timeout_secs = 2;
        settings.inference.ollama_timeout_secs = 10;
        settings.inference.retry_attempts = 1;
        settings.inference.cache_enabled = false;
        settings.logging.level = "error".into();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let opener = directory.path().join("literal opener café.sh");
            fs::write(&opener, "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$1\"\n").unwrap();
            fs::set_permissions(&opener, fs::Permissions::from_mode(0o700)).unwrap();
            settings.navigation.opener = vec![
                opener.to_str().unwrap().into(),
                directory.path().join("opened.txt").to_str().unwrap().into(),
                "literal $(touch injected-marker) `id`".into(),
            ];
        }
        fs::write(&config, settings.redacted_toml().unwrap()).unwrap();
        let seed = if seed {
            let output = std::process::Command::new(std::env::current_exe().unwrap())
                .env_clear()
                .env("HOME", directory.path().join("home"))
                .env("GRAPHRAG_WORKSPACE_FIXTURE", directory.path())
                .current_dir(directory.path())
                .args(["--exact", "workspace_fixture_process", "--nocapture"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "fixture seed failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            serde_json::from_slice(&fs::read(directory.path().join("seed.json")).unwrap()).unwrap()
        } else {
            Value::Null
        };
        Self {
            directory,
            config,
            provider,
            seed,
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::cargo_bin("graphrag").unwrap();
        command
            .env_clear()
            .env("HOME", self.directory.path().join("home"))
            .env("XDG_CONFIG_HOME", self.directory.path().join("xdg"))
            .env("COLUMNS", "24")
            .env("LINES", "8")
            .env("TERM", "dumb")
            .current_dir(self.directory.path())
            .arg("--config")
            .arg(&self.config)
            .timeout(Duration::from_secs(30));
        command
    }
    fn session(&self, script: &str) -> std::process::Output {
        self.command()
            .arg("workspace")
            .write_stdin(script)
            .assert()
            .success()
            .get_output()
            .clone()
    }
    fn notes(&self) -> Value {
        let output: Value = serde_json::from_slice(
            &self
                .command()
                .args(["notes", "list", "--format", "json"])
                .assert()
                .success()
                .get_output()
                .stdout,
        )
        .unwrap();
        assert_eq!(output["schema_version"], 1);
        assert_eq!(output["command"], "notes.list");
        output
    }
    fn proposal(&self) -> Value {
        let output = self
            .command()
            .args([
                "garden",
                "review",
                "--id",
                self.seed["proposal"].as_str().unwrap(),
                "--format",
                "json",
            ])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        envelope["data"]["proposals"][0].clone()
    }
    fn drafts(&self) -> Vec<PathBuf> {
        fs::read_dir(self.directory.path().join("corpus.surreal.drafts"))
            .map(|entries| entries.map(|entry| entry.unwrap().path()).collect())
            .unwrap_or_default()
    }
}

struct LiveSession {
    child: Child,
    output: mpsc::Receiver<(bool, Vec<u8>)>,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    readers: Vec<thread::JoinHandle<()>>,
}
impl LiveSession {
    fn start(fixture: &Fixture) -> Self {
        let mut command = std::process::Command::new(assert_cmd::cargo::cargo_bin("graphrag"));
        command
            .env_clear()
            .env("HOME", fixture.directory.path().join("home"))
            .env("XDG_CONFIG_HOME", fixture.directory.path().join("xdg"))
            .current_dir(fixture.directory.path())
            .arg("--config")
            .arg(&fixture.config)
            .arg("workspace")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        let mut child = command.spawn().unwrap();
        let (sender, output) = mpsc::channel();
        let mut readers = Vec::new();
        for (is_stderr, mut stream) in [
            (
                false,
                Box::new(child.stdout.take().unwrap()) as Box<dyn Read + Send>,
            ),
            (
                true,
                Box::new(child.stderr.take().unwrap()) as Box<dyn Read + Send>,
            ),
        ] {
            let sender = sender.clone();
            readers.push(thread::spawn(move || {
                let mut buffer = [0; 4096];
                while let Ok(count) = stream.read(&mut buffer) {
                    if count == 0 {
                        break;
                    }
                    if sender.send((is_stderr, buffer[..count].to_vec())).is_err() {
                        break;
                    }
                }
            }));
        }
        Self {
            child,
            output,
            stdout: Vec::new(),
            stderr: Vec::new(),
            readers,
        }
    }
    fn send(&mut self, line: &str) {
        writeln!(self.child.stdin.as_mut().unwrap(), "{line}").unwrap();
        self.child.stdin.as_mut().unwrap().flush().unwrap();
    }
    fn text(&self) -> String {
        format!(
            "{}\n{}",
            String::from_utf8_lossy(&self.stdout),
            String::from_utf8_lossy(&self.stderr)
        )
    }
    fn wait_for(&mut self, needle: &str) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while !self.text().contains(needle) {
            assert!(
                Instant::now() < deadline,
                "workspace missing {needle:?}: {}",
                self.text()
            );
            match self.output.recv_timeout(Duration::from_millis(20)) {
                Ok((is_stderr, bytes)) => {
                    if is_stderr {
                        self.stderr.extend(bytes);
                    } else {
                        self.stdout.extend(bytes);
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("workspace exited before {needle:?}: {}", self.text())
                }
            }
        }
    }
    #[cfg(unix)]
    fn interrupt(&self) {
        assert!(std::process::Command::new("/bin/kill")
            .args(["-INT", &self.child.id().to_string()])
            .status()
            .unwrap()
            .success());
    }
    fn finish(&mut self) {
        self.send("quit");
        self.child.stdin.take();
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            while let Ok((is_stderr, bytes)) = self.output.try_recv() {
                if is_stderr {
                    self.stderr.extend(bytes);
                } else {
                    self.stdout.extend(bytes);
                }
            }
            if let Some(status) = self.child.try_wait().unwrap() {
                assert!(status.success(), "{}", self.text());
                break;
            }
            assert!(
                Instant::now() < deadline,
                "workspace failed to quit: {}",
                self.text()
            );
            thread::sleep(Duration::from_millis(5));
        }
        for reader in self.readers.drain(..) {
            reader.join().unwrap();
        }
        while let Ok((is_stderr, bytes)) = self.output.try_recv() {
            if is_stderr {
                self.stderr.extend(bytes);
            } else {
                self.stdout.extend(bytes);
            }
        }
    }
}
impl Drop for LiveSession {
    fn drop(&mut self) {
        if self.child.try_wait().unwrap().is_none() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[test]
fn offline_workspace_browses_selects_and_copies_without_provider_requests() {
    let fixture = Fixture::new(true);
    fixture.provider.mode("fail");
    let output = fixture.session(
        "recent\nkeyword sourcetoken\nselect 1\ninspect\ncopy\nstatus\nproposals\nstats\nquit\n",
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(
        stdout.contains(fixture.seed["source_note"].as_str().unwrap()),
        "{stdout}"
    );
    assert!(stdout.contains("Revision:"), "{stdout}");
    assert!(
        stdout.contains(fixture.seed["source_content"].as_str().unwrap()),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!(
            "graphrag> {}\ngraphrag>",
            fixture.seed["source_content"].as_str().unwrap()
        )),
        "copy must preserve exact selected bytes: {stdout}"
    );
    assert!(
        stdout.contains(fixture.seed["proposal"].as_str().unwrap()),
        "{stdout}"
    );
    assert!(
        stderr.contains(fixture.seed["source_note"].as_str().unwrap()),
        "copy citation: {stderr}"
    );
    assert!(!stdout.contains("Error:"), "{stdout}\n{stderr}");
    assert_eq!(fixture.provider.count(), 0);
}

#[test]
fn empty_narrow_workspace_unknown_input_and_eof_remain_usable() {
    let fixture = Fixture::new(false);
    let output =
        fixture.session("recent\nkeyword absentlexeme\nselect 99\nunknown-command\nhelp\nstats\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        text.contains("No notes") || text.contains("No results"),
        "{text}"
    );
    assert!(text.contains("Unknown command"), "{text}");
    assert!(text.contains("Goodbye"), "{text}");
    assert!(
        !text.contains('\u{1b}'),
        "piped narrow output contains ANSI: {text}"
    );
    assert_eq!(fixture.provider.count(), 0);
}

#[test]
fn provider_failure_retains_capture_and_returns_to_offline_actions() {
    let fixture = Fixture::new(true);
    fixture.provider.mode("fail");
    let output=fixture.session("search cafelexeme\ncapture recoverable workspace failure café\nkeyword manualtoken\nselect 1\ncopy\nstats\nquit\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("Error:"), "{text}");
    assert!(text.contains("Recover:"), "{text}");
    assert!(
        text.contains(fixture.seed["manual_note"].as_str().unwrap()),
        "{text}"
    );
    let drafts = fixture.drafts();
    assert_eq!(drafts.len(), 1, "{text}");
    assert_eq!(
        fs::read_to_string(&drafts[0]).unwrap(),
        "recoverable workspace failure café"
    );
    assert!(fixture.provider.count() > 0);
    let notes = fixture.notes();
    assert_eq!(notes["data"]["notes"].as_array().unwrap().len(), 2);
}

#[cfg(unix)]
#[test]
fn selected_source_opens_with_literal_arguments_and_new_results_clear_selection() {
    let fixture = Fixture::new(true);
    let output = fixture
        .session("keyword sourcetoken\nselect 1\nopen\ncopy\nkeyword manualtoken\ncopy\nquit\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let opened = fs::read_to_string(fixture.directory.path().join("opened.txt")).unwrap();
    let arguments: Vec<_> = opened.lines().collect();
    assert_eq!(arguments.len(), 3, "{opened}");
    assert_eq!(arguments[1], "literal $(touch injected-marker) `id`");
    assert_eq!(arguments[2], fixture.seed["source_path"].as_str().unwrap());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("\\u{1b}[2J.md"),
        "opened path should escape terminal controls: {stdout}"
    );
    assert!(
        !stdout.contains('\u{1b}'),
        "human output printed an active terminal escape: {stdout}"
    );
    assert!(!fixture.directory.path().join("injected-marker").exists());
    assert!(
        text.contains("Error:"),
        "copy after new results requires a new selection: {text}"
    );
    assert_eq!(fixture.provider.count(), 0);
}

#[test]
fn competing_cli_reports_owner_and_succeeds_after_workspace_quits() {
    let fixture = Fixture::new(true);
    let mut session = LiveSession::start(&fixture);
    session.send("keyword manualtoken");
    session.wait_for(fixture.seed["manual_note"].as_str().unwrap());
    let output = fixture
        .command()
        .arg("stats")
        .assert()
        .failure()
        .get_output()
        .clone();
    let error = String::from_utf8(output.stderr).unwrap();
    assert!(
        error.to_ascii_lowercase().contains("lock")
            || error.to_ascii_lowercase().contains("already in use"),
        "{error}"
    );
    assert!(error.to_ascii_lowercase().contains("workspace"), "{error}");
    assert!(error.to_ascii_lowercase().contains("retry"), "{error}");
    assert!(error.contains("corpus.surreal"), "{error}");
    session.finish();
    fixture.command().arg("stats").assert().success();
    assert_eq!(fixture.provider.count(), 0);
}

#[cfg(unix)]
#[test]
fn cancelled_search_does_not_cancel_the_next_action() {
    let fixture = Fixture::new(true);
    fixture.provider.mode("block");
    let mut session = LiveSession::start(&fixture);
    session.send("search cafelexeme");
    fixture.provider.wait_blocked();
    session.interrupt();
    session.wait_for("Action cancelled");
    fixture.provider.release.store(true, Ordering::Release);
    fixture.provider.mode("ok");
    session.send("keyword manualtoken");
    session.wait_for(fixture.seed["manual_note"].as_str().unwrap());
    session.send("capture fresh token successful note");
    session.wait_for("Captured note");
    session.finish();
    assert_eq!(
        fixture.notes()["data"]["notes"].as_array().unwrap().len(),
        3,
        "{}",
        session.text()
    );
}

#[cfg(unix)]
#[test]
fn cancelled_capture_retains_draft_and_next_capture_uses_fresh_token() {
    let fixture = Fixture::new(false);
    fixture.provider.mode("block");
    let mut session = LiveSession::start(&fixture);
    session.send("capture cancelled pending workspace body");
    fixture.provider.wait_blocked();
    session.interrupt();
    session.wait_for("Interrupt requested");
    fixture.provider.release.store(true, Ordering::Release);
    session.wait_for("Recover:");
    fixture.provider.mode("ok");
    session.send("capture fresh workspace body after cancellation");
    session.wait_for("Captured note");
    session.finish();
    let drafts = fixture.drafts();
    assert_eq!(drafts.len(), 1, "{}", session.text());
    assert_eq!(
        fs::read_to_string(&drafts[0]).unwrap(),
        "cancelled pending workspace body"
    );
    assert_eq!(
        fixture.notes()["data"]["notes"].as_array().unwrap().len(),
        1,
        "{}",
        session.text()
    );
}

#[test]
fn proposal_review_skip_returns_to_the_same_offline_workspace() {
    let fixture = Fixture::new(true);
    let before = fixture.proposal();
    let output = fixture.session("proposals\nreview 1\ns\nstats\nquit\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("Review skipped; no changes saved."), "{text}");
    assert!(text.contains("Notes:"), "{text}");
    assert!(
        !text.contains("Unknown command"),
        "nested review must consume its own input: {text}"
    );
    assert_eq!(fixture.provider.count(), 0);
    assert_eq!(fixture.proposal(), before);
}

#[test]
fn proposal_confirmation_cancellation_and_eof_leave_the_audit_unchanged() {
    for script in [
        "proposals\nreview 1\na\n\nno\nstats\nquit\n",
        "proposals\nreview 1\na\n",
    ] {
        let fixture = Fixture::new(true);
        let before = fixture.proposal();
        let output = fixture.session(script);
        let text = format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(text.contains("Goodbye"), "{text}");
        assert_eq!(fixture.proposal(), before, "{text}");
        assert_eq!(fixture.provider.count(), 0);
    }
}

#[test]
fn workspace_confirmed_accept_and_undo_use_shared_manual_audit() {
    let fixture = Fixture::new(true);
    let accepted = fixture
        .session("proposals\nreview 1\na\nworkspace acceptance evidence\nyes\nstats\nquit\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&accepted.stdout),
        String::from_utf8_lossy(&accepted.stderr)
    );
    let before_undo = fixture.proposal();
    assert_eq!(before_undo["status"], "accepted", "{text}");
    assert_eq!(before_undo["acceptance_is_manual"], true, "{text}");
    assert_eq!(
        before_undo["action_reason"], "workspace acceptance evidence",
        "{text}"
    );
    assert!(before_undo["reviewer"].is_string(), "{text}");
    let undone =
        fixture.session("proposals all\nreview 1\nu\nworkspace undo evidence\nyes\nquit\n");
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&undone.stdout),
        String::from_utf8_lossy(&undone.stderr)
    );
    let after_undo = fixture.proposal();
    assert_eq!(after_undo["status"], "superseded", "{text}");
    assert_eq!(after_undo["reviewer"], before_undo["reviewer"]);
    assert_eq!(after_undo["reviewed_at"], before_undo["reviewed_at"]);
    assert!(after_undo["resulting_edge_id"].is_null(), "{text}");
    assert_eq!(fixture.provider.count(), 0);
}

#[test]
fn interactive_alias_keeps_memory_browsing_and_refuses_recoverable_capture() {
    let fixture = Fixture::new(false);
    let output = fixture
        .command()
        .args(["--memory", "interactive"])
        .write_stdin("recent\ncapture private transient text\nstats\nquit\n")
        .assert()
        .success()
        .get_output()
        .clone();
    let text = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(text.contains("persistent storage"), "{text}");
    assert!(text.contains("restart without --memory"), "{text}");
    assert!(text.contains("Notes:"), "{text}");
    assert!(fixture.drafts().is_empty());
    assert!(!fixture.directory.path().join("corpus.surreal").exists());
    assert_eq!(fixture.provider.count(), 0);
}
