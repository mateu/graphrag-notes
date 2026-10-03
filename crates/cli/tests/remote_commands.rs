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
        .stderr(contains("--request-id 'capture-replay-001'"))
        .stderr(contains("--content-file"))
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
        .stderr(contains("--request-id 'cancel-recovery-001'"));
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
