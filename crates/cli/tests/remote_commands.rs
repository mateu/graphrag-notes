//! Remote failures must not bootstrap a client database or lose capture recovery.
use assert_cmd::Command;
use predicates::str::contains;
use std::fs;

fn graphrag() -> Command {
    let mut command = Command::cargo_bin("graphrag").unwrap();
    command.env_remove("GRAPHRAG_SERVER");
    command.env_remove("GRAPHRAG_CONFIG");
    command
}

#[test]
fn remote_dispatch_bypasses_invalid_local_configuration_and_database() {
    let temp = tempfile::tempdir().unwrap();
    let config = temp.path().join("invalid.toml");
    let database = temp.path().join("must-not-exist");
    fs::write(&config, "this is invalid TOML [").unwrap();
    let token = "synthetic-client-token-do-not-print-1234";
    let result = graphrag()
        .env("GRAPHRAG_CONFIG", &config)
        .env("GRAPHRAG_DB_PATH", &database)
        .env("GRAPHRAG_TOKEN", token)
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "search",
            "synthetic",
            "--mode",
            "keyword",
        ])
        .assert()
        .failure()
        .stderr(contains("service_unreachable"));
    let error = String::from_utf8_lossy(&result.get_output().stderr);
    assert!(!error.contains(token));
    assert!(!database.exists());
}

#[test]
fn project_dotenv_cannot_redirect_an_inherited_bearer_credential() {
    use std::io::Read;
    use std::sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    };
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("private-home");
    fs::create_dir(&home).unwrap();
    let database = temp.path().join("local-database");
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    fs::write(
        temp.path().join(".env"),
        format!(
            "GRAPHRAG_SERVER=http://{}/mcp\nGRAPHRAG_DB_PATH=\"{}\"\n",
            listener.local_addr().unwrap(),
            database.display()
        ),
    )
    .unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let connections = Arc::new(AtomicUsize::new(0));
    let thread_stop = Arc::clone(&stop);
    let thread_connections = Arc::clone(&connections);
    let attacker = std::thread::spawn(move || {
        while !thread_stop.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, _)) => {
                    thread_connections.fetch_add(1, Ordering::Relaxed);
                    stream.set_nonblocking(false).unwrap();
                    stream
                        .set_read_timeout(Some(std::time::Duration::from_millis(200)))
                        .unwrap();
                    let mut request = [0; 4096];
                    let _ = stream.read(&mut request);
                }
                Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                Err(error) => panic!("synthetic endpoint accept failed: {error}"),
            }
        }
    });
    let result = graphrag()
        .current_dir(temp.path())
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", &home)
        .env_remove("GRAPHRAG_DB_PATH")
        .env("GRAPHRAG_TOKEN", "synthetic-inherited-bearer-do-not-send")
        .args([
            "search",
            "synthetic",
            "--mode",
            "keyword",
            "--graph",
            "off",
            "--format",
            "json",
        ])
        .timeout(std::time::Duration::from_secs(5))
        .assert();
    stop.store(true, Ordering::Relaxed);
    attacker.join().unwrap();
    assert_eq!(
        connections.load(Ordering::Relaxed),
        0,
        "a project-controlled endpoint must receive no credential-bearing request"
    );
    result.success();
    assert!(
        database.exists(),
        "the command should use its isolated local database"
    );
}

#[test]
fn inherited_process_endpoint_still_dispatches_before_local_bootstrap() {
    let temp = tempfile::tempdir().unwrap();
    let database = temp.path().join("must-not-exist");
    let config = temp.path().join("invalid.toml");
    fs::write(&config, "this is invalid TOML [").unwrap();
    graphrag()
        .current_dir(temp.path())
        .env("GRAPHRAG_SERVER", "http://127.0.0.1:0/mcp")
        .env("GRAPHRAG_CONFIG", &config)
        .env("GRAPHRAG_DB_PATH", &database)
        .env("GRAPHRAG_TOKEN", "synthetic-inherited-bearer-12345678")
        .args(["search", "synthetic", "--mode", "keyword"])
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure()
        .stderr(contains("service_unreachable"));
    assert!(!database.exists());
}

#[test]
fn ambient_proxy_cannot_receive_private_mcp_bearer_requests() {
    for inherited_proxy in [false, true] {
        use std::io::Read;
        use std::sync::{
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        };
        let temp = tempfile::tempdir().unwrap();
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        listener.set_nonblocking(true).unwrap();
        let proxy = format!("http://{}", listener.local_addr().unwrap());
        fs::write(temp.path().join(".env"), format!("HTTP_PROXY={proxy}\nhttp_proxy={proxy}\nALL_PROXY={proxy}\nall_proxy={proxy}\nNO_PROXY=''\nno_proxy=''\n")).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let connections = Arc::new(AtomicUsize::new(0));
        let thread_stop = Arc::clone(&stop);
        let thread_connections = Arc::clone(&connections);
        let attacker = std::thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        thread_connections.fetch_add(1, Ordering::Relaxed);
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(std::time::Duration::from_millis(200)))
                            .unwrap();
                        let mut request = [0; 4096];
                        let _ = stream.read(&mut request);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        std::thread::sleep(std::time::Duration::from_millis(5));
                    }
                    Err(error) => panic!("synthetic proxy accept failed: {error}"),
                }
            }
        });
        let mut command = graphrag();
        for name in [
            "HTTP_PROXY",
            "http_proxy",
            "HTTPS_PROXY",
            "https_proxy",
            "ALL_PROXY",
            "all_proxy",
            "NO_PROXY",
            "no_proxy",
        ] {
            command.env_remove(name);
        }
        if inherited_proxy {
            command
                .env("HTTP_PROXY", &proxy)
                .env("ALL_PROXY", &proxy)
                .env("NO_PROXY", "");
        }
        let result = command
            .current_dir(temp.path())
            .env("HOME", temp.path())
            .env(
                "GRAPHRAG_TOKEN",
                "synthetic-private-mcp-bearer-do-not-proxy",
            )
            .args([
                "--server",
                "http://127.0.0.1:0/mcp",
                "search",
                "synthetic",
                "--mode",
                "keyword",
            ])
            .timeout(std::time::Duration::from_secs(5))
            .assert();
        stop.store(true, Ordering::Relaxed);
        attacker.join().unwrap();
        assert_eq!(
            connections.load(Ordering::Relaxed),
            0,
            "a project-controlled proxy must never receive the bearer request"
        );
        result.failure().stderr(contains("service_unreachable"));
    }
}

#[test]
fn remote_endpoint_does_not_load_a_project_supplied_bearer_credential() {
    let temp = tempfile::tempdir().unwrap();
    let token = "synthetic-project-controlled-credential-12345678";
    fs::write(
        temp.path().join(".env"),
        format!("GRAPHRAG_TOKEN={token}\n"),
    )
    .unwrap();
    let result = graphrag()
        .current_dir(temp.path())
        .env("HOME", temp.path())
        .env_remove("GRAPHRAG_TOKEN")
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "search",
            "synthetic",
            "--mode",
            "keyword",
        ])
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure()
        .stderr(contains(
            "remote credential environment variable is missing",
        ));
    assert!(!String::from_utf8_lossy(&result.get_output().stderr).contains(token));
}

#[test]
fn failed_capture_keeps_private_draft_and_original_request_identity() {
    let temp = tempfile::tempdir().unwrap();
    let drafts = temp.path().join("client drafts");
    graphrag()
        .env("GRAPHRAG_TOKEN", "synthetic-capture-token-12345678901234")
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id",
            "capture-replay-001",
            "capture",
            "synthetic saved draft",
            "--title",
            "Synthetic",
            "--tags",
            "test",
            "--draft-dir",
        ])
        .arg(&drafts)
        .assert()
        .failure()
        .stderr(contains("--request-id='capture-replay-001'"))
        .stderr(contains("--content-file"))
        .stderr(contains("--recover-draft"))
        .stderr(contains(format!("--draft-dir '{}'", drafts.display())));
    let paths = fs::read_dir(&drafts)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    assert_eq!(
        fs::read_to_string(&paths[0]).unwrap(),
        "synthetic saved draft"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&paths[0]).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[cfg(unix)]
#[test]
fn option_like_capture_request_identity_survives_the_printed_recovery_command() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("private-home");
    fs::create_dir(&home).unwrap();
    let drafts = temp.path().join("client drafts");
    let token = "synthetic-replay-token-12345678901234";
    let result = graphrag()
        .current_dir(temp.path())
        .env("HOME", &home)
        .env("GRAPHRAG_TOKEN", token)
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id=-custom",
            "capture",
            "synthetic replay draft",
            "--draft-dir",
        ])
        .arg(&drafts)
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&result.get_output().stderr);
    let recovery = stderr
        .lines()
        .find_map(|line| line.strip_prefix("Recover: "))
        .expect("failed capture must print its retry command");
    assert!(recovery.contains("--request-id='-custom'"));
    let original = fs::read_dir(&drafts)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    // Execute the actual printed shell recovery command against only the
    // synthetic fixture and private binary PATH, exercising shell + Clap.
    let binary_directory = std::path::Path::new(env!("CARGO_BIN_EXE_graphrag"))
        .parent()
        .unwrap();
    let replay = Command::new("/bin/sh")
        .current_dir(temp.path())
        .env_remove("GRAPHRAG_SERVER")
        .env_remove("GRAPHRAG_CONFIG")
        .env("HOME", &home)
        .env("PATH", binary_directory)
        .env("GRAPHRAG_TOKEN", token)
        .args(["-c", recovery])
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure()
        .stderr(contains("Capture request ID: -custom"))
        .stderr(contains("service_unreachable"));
    assert!(!String::from_utf8_lossy(&replay.get_output().stderr).contains(token));
    assert_eq!(
        fs::read_to_string(original).unwrap(),
        "synthetic replay draft"
    );
}

#[cfg(unix)]
#[test]
fn relative_capture_draft_recovery_works_from_another_working_directory() {
    let temp = tempfile::tempdir().unwrap();
    let original_directory = temp.path().join("original working directory");
    let retry_directory = temp.path().join("different working directory");
    let home = temp.path().join("private-home");
    for directory in [&original_directory, &retry_directory, &home] {
        fs::create_dir(directory).unwrap();
    }
    // macOS getcwd resolves the /var -> /private/var temporary-directory alias.
    let drafts = fs::canonicalize(&original_directory)
        .unwrap()
        .join("private drafts");
    let token = "synthetic-cross-directory-token-12345678";
    let failure = graphrag()
        .current_dir(&original_directory)
        .env("HOME", &home)
        .env("GRAPHRAG_TOKEN", token)
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id=cross-directory-001",
            "capture",
            "synthetic retained cross-directory draft",
            "--draft-dir",
            "private drafts",
        ])
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&failure.get_output().stderr);
    let recovery = stderr
        .lines()
        .find_map(|line| line.strip_prefix("Recover: "))
        .expect("failed capture must print its retry command");
    assert!(recovery.contains(&format!("--draft-dir '{}'", drafts.display())));
    let original = fs::read_dir(&drafts)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let binary_directory = std::path::Path::new(env!("CARGO_BIN_EXE_graphrag"))
        .parent()
        .unwrap();
    Command::new("/bin/sh")
        .current_dir(&retry_directory)
        .env_remove("GRAPHRAG_SERVER")
        .env_remove("GRAPHRAG_CONFIG")
        .env("HOME", &home)
        .env("PATH", binary_directory)
        .env("GRAPHRAG_TOKEN", token)
        .args(["-c", recovery])
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure()
        .stderr(contains("Capture request ID: cross-directory-001"))
        .stderr(contains("service_unreachable"));
    assert_eq!(
        fs::read_to_string(original).unwrap(),
        "synthetic retained cross-directory draft"
    );
    assert_eq!(fs::read_dir(&retry_directory).unwrap().count(), 0);
}

#[test]
fn recovery_requires_remote_file_write_and_explicit_original_request_identity() {
    for args in [
        vec!["--recover-draft", "capture", "text"],
        vec![
            "--recover-draft",
            "--request-id",
            "retry-1",
            "capture",
            "text",
        ],
        vec![
            "--server",
            "http://127.0.0.1:0/mcp",
            "--recover-draft",
            "--request-id",
            "retry-1",
            "search",
            "text",
        ],
        vec![
            "--server",
            "http://127.0.0.1:0/mcp",
            "--recover-draft",
            "--request-id",
            "retry-1",
            "capture",
            "text",
        ],
    ] {
        graphrag()
            .args(args)
            .assert()
            .failure()
            .stderr(contains("--recover-draft"));
    }
}

#[test]
fn invalid_service_security_policy_is_rejected_before_database_creation() {
    let temp = tempfile::tempdir().unwrap();
    let database = temp.path().join("must-not-exist");
    graphrag()
        .arg("--db-path")
        .arg(&database)
        .args(["serve", "--listen", "0.0.0.0:3000", "--credentials-file"])
        .arg(temp.path().join("missing-policy.json"))
        .assert()
        .failure()
        .stderr(contains("Non-loopback binding requires"));
    assert!(!database.exists());
}

#[test]
fn in_memory_service_is_rejected_before_configuration_or_database_creation() {
    let temp = tempfile::tempdir().unwrap();
    let home = temp.path().join("private-home");
    fs::create_dir(&home).unwrap();
    let config = temp.path().join("invalid.toml");
    let database = temp.path().join("must-not-exist");
    fs::write(&config, "this is invalid TOML [").unwrap();
    graphrag()
        .env("HOME", &home)
        .env("XDG_CONFIG_HOME", &home)
        .arg("--config")
        .arg(&config)
        .arg("--db-path")
        .arg(&database)
        .args(["--memory", "serve", "--credentials-file"])
        .arg(temp.path().join("missing-policy.json"))
        .timeout(std::time::Duration::from_secs(5))
        .assert()
        .failure()
        .stderr(contains("serve requires a persistent database"))
        .stderr(contains("remove --memory"))
        .stderr(contains("--db-path"));
    assert!(!database.exists());
    assert_eq!(
        fs::read_to_string(&config).unwrap(),
        "this is invalid TOML ["
    );
    assert_eq!(fs::read_dir(&home).unwrap().count(), 0);
}

#[cfg(unix)]
#[test]
fn cancelled_editor_preserves_changed_remote_draft_without_sending() {
    use std::os::unix::fs::PermissionsExt;
    let temp = tempfile::tempdir().unwrap();
    let editor = temp.path().join("cancel-editor");
    let drafts = temp.path().join("drafts");
    fs::write(
        &editor,
        "#!/bin/sh\nprintf 'valuable edited draft' > \"$1\"\nexit 1\n",
    )
    .unwrap();
    fs::set_permissions(&editor, fs::Permissions::from_mode(0o700)).unwrap();
    graphrag()
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id",
            "cancel-recovery-001",
            "capture",
            "--editor",
            "--editor-command",
        ])
        .arg(&editor)
        .arg("--draft-dir")
        .arg(&drafts)
        .assert()
        .failure()
        .stderr(contains("editor cancelled"))
        .stderr(contains("--request-id='cancel-recovery-001'"));
    let paths = fs::read_dir(&drafts)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    assert_eq!(
        fs::read_to_string(&paths[0]).unwrap(),
        "valuable edited draft"
    );
}

#[test]
fn remote_edit_disconnect_retains_content_revision_and_request_identity() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("edit.md");
    let drafts = temp.path().join("private drafts");
    fs::write(&input, "valuable replacement text").unwrap();
    let revision = "a".repeat(64);
    graphrag()
        .env("GRAPHRAG_TOKEN", "synthetic-client-token-12345678901234")
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id",
            "edit-retry-001",
            "--expected-revision",
            &revision,
            "notes",
            "edit",
            "note:synthetic",
            "--content-file",
        ])
        .arg(&input)
        .arg("--draft-dir")
        .arg(&drafts)
        .assert()
        .failure()
        .stderr(contains("service_unreachable"))
        .stderr(contains("--recover-draft"))
        .stderr(contains("--request-id='edit-retry-001'"))
        .stderr(contains(format!("--expected-revision '{revision}'")))
        .stderr(contains(format!("--draft-dir '{}'", drafts.display())));
    let paths = fs::read_dir(&drafts)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    assert_eq!(
        fs::read_to_string(&paths[0]).unwrap(),
        "valuable replacement text"
    );
}

#[test]
fn remote_decisions_require_review_revision_and_confirmation_before_connecting() {
    for args in [
        vec!["notes", "delete", "note:synthetic"],
        vec!["garden", "proposals", "accept", "proposed_edge:synthetic"],
        vec!["garden", "proposals", "reject", "proposed_edge:synthetic"],
        vec!["garden", "proposals", "undo", "proposed_edge:synthetic"],
    ] {
        graphrag()
            .args(["--server", "http://127.0.0.1:0/mcp"])
            .args(&args)
            .assert()
            .failure()
            .stderr(contains("--yes"));
        graphrag()
            .args(["--server", "http://127.0.0.1:0/mcp"])
            .args(&args)
            .arg("--yes")
            .assert()
            .failure()
            .stderr(contains("--expected-revision"));
    }
}

#[test]
fn failed_upload_preserves_private_input_and_logical_identity_without_local_corpus() {
    let temp = tempfile::tempdir().unwrap();
    let input = temp.path().join("client markdown.md");
    let drafts = temp.path().join("private drafts");
    let database = temp.path().join("must-not-exist");
    fs::write(&input, "# Uploaded synthetic content\n\nrecoverylexeme").unwrap();
    let result = graphrag()
        .env("HOME", temp.path())
        .env("GRAPHRAG_DB_PATH", &database)
        .env("GRAPHRAG_TOKEN", "synthetic-upload-token-12345678901234")
        .args([
            "--server",
            "http://127.0.0.1:0/mcp",
            "--request-id=-upload-recovery-001",
            "upload",
            "--document-key=-logical-key",
            "--title=-title",
            "--content-file",
        ])
        .arg(&input)
        .arg("--draft-dir")
        .arg(&drafts)
        .arg("--extract-entities")
        .args(["--format", "json"])
        .assert()
        .failure()
        .stderr(contains("--request-id='-upload-recovery-001'"))
        .stderr(contains("--document-key='-logical-key'"))
        .stderr(contains("--title='-title'"))
        .stderr(contains("--draft-dir"))
        .stderr(contains("--extract-entities"));
    let error = String::from_utf8_lossy(&result.get_output().stderr);
    assert!(!error.contains("synthetic-upload-token"));
    assert!(!database.exists());
    let paths = fs::read_dir(&drafts)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect::<Vec<_>>();
    assert_eq!(paths.len(), 1);
    assert_eq!(
        fs::read_to_string(&paths[0]).unwrap(),
        fs::read_to_string(&input).unwrap()
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            fs::metadata(&paths[0]).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn upload_without_server_fails_before_creating_corpus_or_reading_client_input() {
    let temp = tempfile::tempdir().unwrap();
    let database = temp.path().join("must-not-exist");
    graphrag()
        .env("HOME", temp.path())
        .arg("--db-path")
        .arg(&database)
        .args([
            "upload",
            "--document-key",
            "logical",
            "--content-file",
            "/missing/synthetic-file.md",
        ])
        .assert()
        .failure()
        .stderr(contains("upload requires --server"));
    assert!(!database.exists());
}
