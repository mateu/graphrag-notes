use super::*;

fn fixture() -> (tempfile::TempDir, PathBuf, FolderConfig) {
    let directory = tempfile::tempdir().unwrap();
    let config = directory.path().join("private config.toml");
    let root = directory.path().join("notes café 日本語");
    std::fs::create_dir(&root).unwrap();
    std::fs::write(&config, "# Keep this comment\n[inference]\nembedding_url = 'http://user:fake-secret@localhost:8081/?token=literal-credential' # do not redact\n[logging]\nlevel = 'warn' # custom setting\n").unwrap();
    (
        directory,
        config,
        FolderConfig {
            path: root,
            ..Default::default()
        },
    )
}

#[test]
fn registration_preserves_comments_credentials_other_settings_and_permissions() {
    let (_directory, path, folder) = fixture();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640)).unwrap();
    }
    let previous = std::fs::read_to_string(&path).unwrap();
    let registration = register_folder(&path, "work", folder).unwrap();
    let registered = registration.folder;
    assert_eq!(
        std::fs::read(&registration.config_backup).unwrap(),
        previous.as_bytes()
    );
    let saved = std::fs::read_to_string(&path).unwrap();
    assert!(saved.starts_with(&previous));
    assert!(saved.contains("fake-secret"));
    assert!(saved.contains("literal-credential"));
    assert!(saved.contains("# custom setting"));
    assert_eq!(
        RuntimeConfig::from_file(&path).unwrap().folders["work"],
        registered
    );
    assert!(registered.path.is_absolute());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o640
        );
        assert_eq!(
            std::fs::metadata(registration.config_backup)
                .unwrap()
                .permissions()
                .mode()
                & 0o777,
            0o640
        );
    }
}

#[test]
fn invalid_names_rules_duplicate_names_and_overlapping_roots_leave_config_untouched() {
    let (_directory, path, folder) = fixture();
    let before = std::fs::read(&path).unwrap();
    for invalid in ["", "two words", "a/b", "日本語", "--all", "-work", "_work"] {
        assert!(register_folder(&path, invalid, folder.clone()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), before);
    }
    let mut invalid = folder.clone();
    invalid.include = vec!["[".into()];
    assert!(register_folder(&path, "invalid", invalid).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    register_folder(&path, "work", folder.clone()).unwrap();
    let registered = std::fs::read(&path).unwrap();
    assert!(register_folder(&path, "work", folder.clone()).is_err());
    assert!(register_folder(&path, "duplicate", folder.clone()).is_err());
    let nested = folder.path.join("nested");
    std::fs::create_dir(&nested).unwrap();
    assert!(register_folder(
        &path,
        "nested",
        FolderConfig {
            path: nested,
            ..Default::default()
        }
    )
    .is_err());
    assert_eq!(std::fs::read(&path).unwrap(), registered);
}

#[test]
fn registration_supports_empty_and_existing_inline_folder_maps_without_losing_settings() {
    for existing in [false, true] {
        let (directory, path, folder) = fixture();
        let other = directory.path().join("other");
        std::fs::create_dir(&other).unwrap();
        let folders = if existing {
            format!("{{ old = {{ path = {:?}, recursive = false, include = ['*.md'], exclude = [] }} }}", std::fs::canonicalize(&other).unwrap().to_str().unwrap())
        } else {
            "{}".into()
        };
        let original = format!("# Keep inline style\nfolders = {folders} # folder comment\n[inference]\nembedding_url = 'http://user:fake-secret@localhost:8081/?token=literal-credential' # preserve credentials\n");
        std::fs::write(&path, original).unwrap();
        let added = register_folder(&path, "work", folder).unwrap().folder;
        let saved = std::fs::read_to_string(&path).unwrap();
        assert!(saved.contains("# folder comment"));
        assert!(saved.contains("# preserve credentials"));
        assert!(saved.contains("fake-secret"));
        let parsed = RuntimeConfig::from_file(&path).unwrap();
        assert_eq!(parsed.folders["work"], added);
        assert_eq!(parsed.folders.len(), if existing { 2 } else { 1 });
        if existing {
            assert!(!parsed.folders["old"].recursive);
        }
    }
}

#[cfg(unix)]
#[test]
fn symlink_config_is_rejected_and_symlink_root_alias_is_canonicalized() {
    let (directory, path, folder) = fixture();
    let original = std::fs::read(&path).unwrap();
    let link = directory.path().join("linked.toml");
    std::os::unix::fs::symlink(&path, &link).unwrap();
    assert!(register_folder(&link, "work", folder.clone()).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), original);
    let alias = directory.path().join("alias");
    std::os::unix::fs::symlink(&folder.path, &alias).unwrap();
    let registered = register_folder(
        &path,
        "work",
        FolderConfig {
            path: alias,
            ..Default::default()
        },
    )
    .unwrap()
    .folder;
    assert_eq!(
        registered.path,
        std::fs::canonicalize(&folder.path).unwrap()
    );
    assert!(register_folder(&path, "duplicate", folder).is_err());
}

#[test]
fn concurrent_registration_lock_refuses_write_without_removing_other_owners_lock() {
    let (_directory, path, folder) = fixture();
    let before = std::fs::read(&path).unwrap();
    let lock = path.with_extension("toml.folders.lock");
    std::fs::write(&lock, "another registrar owns this").unwrap();
    assert!(register_folder(&path, "work", folder).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(
        std::fs::read_to_string(lock).unwrap(),
        "another registrar owns this"
    );
}

#[test]
fn concurrent_external_edit_is_preserved_when_atomic_commit_checks_snapshot() {
    let (_directory, path, _) = fixture();
    let original = std::fs::read(&path).unwrap();
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    temporary.write_all(b"replacement").unwrap();
    std::fs::write(&path, "# New editor content\n[logging]\nlevel = 'debug'\n").unwrap();
    let edited = std::fs::read(&path).unwrap();
    assert!(persist_config_if_unchanged(&path, &original, temporary).is_err());
    assert_eq!(std::fs::read(&path).unwrap(), edited);
    assert!(!path.with_extension("toml.folders.lock").exists());
}

#[test]
fn an_editor_save_after_the_snapshot_check_is_captured_and_restored() {
    let (_directory, path, _) = fixture();
    let original = std::fs::read(&path).unwrap();
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    temporary.write_all(b"staged registration").unwrap();
    // Simulate the previously unprotected window between the snapshot check
    // and commit. Capture must inspect the file actually moved, not old bytes.
    let edited = b"# Saved after snapshot check\n[logging]\nlevel = 'debug'\n";
    std::fs::write(&path, edited).unwrap();
    let error = capture_and_replace_config(&path, &original, temporary).unwrap_err();
    assert!(error.to_string().contains("config changed"));
    assert_eq!(std::fs::read(&path).unwrap(), edited);
    let backup = std::fs::read_dir(path.parent().unwrap())
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .find(|candidate| {
            candidate
                .file_name()
                .unwrap()
                .to_string_lossy()
                .contains(".folders-backup-")
        })
        .unwrap();
    assert_eq!(std::fs::read(&backup).unwrap(), edited);
    assert!(error.to_string().contains(backup.to_str().unwrap()));
}

#[test]
fn an_editor_recreating_the_config_during_commit_wins_without_losing_either_version() {
    let (_directory, path, _) = fixture();
    let original = std::fs::read(&path).unwrap();
    let backup = path.with_extension("captured-config");
    std::fs::rename(&path, &backup).unwrap();
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    temporary.write_all(b"staged registration").unwrap();
    let edited = b"# Editor created a new config during commit\n";
    std::fs::write(&path, edited).unwrap();
    let error = persist_captured_config(&path, &original, temporary, backup.clone()).unwrap_err();
    assert_eq!(std::fs::read(&path).unwrap(), edited);
    assert_eq!(std::fs::read(&backup).unwrap(), original);
    assert!(error.to_string().contains(backup.to_str().unwrap()));
}

#[test]
fn an_editor_holding_the_old_file_can_save_to_the_retained_backup_after_commit() {
    let (_directory, path, _) = fixture();
    let original = std::fs::read(&path).unwrap();
    let mut editor = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    let mut temporary = tempfile::NamedTempFile::new_in(path.parent().unwrap()).unwrap();
    temporary.write_all(b"staged registration").unwrap();
    let backup = persist_config_if_unchanged(&path, &original, temporary).unwrap();
    let edited = b"# Saved through an already-open editor descriptor\n";
    editor.set_len(0).unwrap();
    editor.write_all(edited).unwrap();
    editor.sync_all().unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), b"staged registration");
    assert_eq!(std::fs::read(backup).unwrap(), edited);
}
