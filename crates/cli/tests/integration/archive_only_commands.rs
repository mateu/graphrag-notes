use assert_cmd::Command;
use std::fs;

fn graphrag() -> Command {
    let mut command = Command::cargo_bin("graphrag").unwrap();
    command.env_remove("GRAPHRAG_CONFIG");
    command
}

#[test]
fn archive_verification_and_dry_runs_bypass_malformed_runtime_config() {
    let temp = tempfile::tempdir().unwrap();
    let valid_config = temp.path().join("valid.toml");
    fs::write(&valid_config, "").unwrap();
    let malformed_config = temp.path().join("malformed.toml");
    fs::write(&malformed_config, "[inference\ntimeout_secs = nope").unwrap();
    let archive = temp.path().join("backup");
    let jsonl = temp.path().join("export.jsonl");

    graphrag()
        .arg("--config")
        .arg(&valid_config)
        .arg("--memory")
        .args(["backup", "create"])
        .arg(&archive)
        .assert()
        .success();
    graphrag()
        .arg("--config")
        .arg(&valid_config)
        .arg("--memory")
        .arg("export")
        .arg(&jsonl)
        .assert()
        .success();

    graphrag()
        .env("GRAPHRAG_CONFIG", &malformed_config)
        .args(["backup", "verify"])
        .arg(&archive)
        .args(["--format", "json"])
        .assert()
        .success();
    graphrag()
        .env("GRAPHRAG_CONFIG", &malformed_config)
        .arg("--db-path")
        .arg(temp.path().join("restore-target"))
        .args(["backup", "restore"])
        .arg(&archive)
        .args(["--dry-run", "--format", "json"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\"dry_run\":true"));
    graphrag()
        .env("GRAPHRAG_CONFIG", &malformed_config)
        .arg("--db-path")
        .arg(temp.path().join("import-target"))
        .arg("import-data")
        .arg(&jsonl)
        .args(["--dry-run", "--format", "json"])
        .assert()
        .success()
        .stdout(predicates::str::contains("\"dry_run\":true"));
}
