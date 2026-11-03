//! Deterministic wait for an embedded RocksDB store to be fully closed.
//!
//! Dropping the last [`DbConnection`](crate::DbConnection) only closes
//! SurrealDB's session channel. A detached engine task then flushes the WAL and
//! memtables, drains compactions and finally drops RocksDB, which releases
//! `<path>/LOCK` as its last write-side step. Until that happens the store is
//! still being written.
//!
//! The lock does not protect a store against a second engine *in the same
//! process* once its directory is renamed: RocksDB's same-process check is
//! keyed by the lock file's path string, and POSIX `fcntl` locks never conflict
//! within one process. Renaming a staged store into place (or reopening it
//! under another path) while the old engine is still shutting down therefore
//! runs two engines on one directory. They allocate the same file numbers, and
//! the shared SST files fail block checksums on the next open.
//!
//! Callers that rename or hand over a store must wait for the release
//! explicitly instead of sleeping for a guessed grace period.

use std::path::Path;
use std::time::{Duration, Instant};

use crate::{DbError, Result};

const POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Wait until this process no longer holds the embedded RocksDB store at
/// `path` open, or fail after `timeout`.
///
/// Returns immediately when the store has no lock file. The check observes
/// whether any file descriptor of this process still refers to the lock file;
/// it never opens the lock file itself, because closing any descriptor for it
/// would silently drop the owning engine's POSIX lock.
pub async fn wait_for_persistent_release(path: impl AsRef<Path>, timeout: Duration) -> Result<()> {
    let path = path.as_ref();
    let lock = match std::fs::canonicalize(path.join("LOCK")) {
        Ok(lock) => lock,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => {
            return Err(DbError::Connection(format!(
                "inspect embedded database lock {}: {error}",
                path.display()
            )))
        }
    };
    let deadline = Instant::now() + timeout;
    loop {
        let held = process_holds_open(&lock).map_err(|error| {
            DbError::Connection(format!(
                "inspect open descriptors for {}: {error}",
                lock.display()
            ))
        })?;
        if !held {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(DbError::Connection(format!(
                "embedded database {} is still held by a closing engine after {timeout:?}",
                path.display()
            )));
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
}

/// Whether any open descriptor of this process refers to `target`.
#[cfg(target_os = "linux")]
fn process_holds_open(target: &Path) -> std::io::Result<bool> {
    for entry in std::fs::read_dir("/proc/self/fd")? {
        let Ok(entry) = entry else { continue };
        // A descriptor may close between listing and inspection.
        if std::fs::read_link(entry.path()).is_ok_and(|link| link == target) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether any open descriptor of this process refers to `target`.
#[cfg(target_os = "macos")]
fn process_holds_open(target: &Path) -> std::io::Result<bool> {
    use std::ffi::{CStr, OsStr};
    use std::os::unix::ffi::OsStrExt;

    for entry in std::fs::read_dir("/dev/fd")? {
        let Ok(entry) = entry else { continue };
        let Some(fd) = entry.file_name().to_str().and_then(|n| n.parse().ok()) else {
            continue;
        };
        let mut buffer = [0 as libc::c_char; libc::PATH_MAX as usize];
        // SAFETY: F_GETPATH writes a NUL-terminated path of at most PATH_MAX
        // bytes into the buffer and does not otherwise affect the descriptor.
        // A descriptor that closed after listing just returns an error.
        if unsafe { libc::fcntl(fd, libc::F_GETPATH, buffer.as_mut_ptr()) } == -1 {
            continue;
        }
        // SAFETY: on success the buffer holds a NUL-terminated string.
        let found = unsafe { CStr::from_ptr(buffer.as_ptr()) };
        if Path::new(OsStr::from_bytes(found.to_bytes())) == target {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Platforms without a descriptor listing fall back to a conservative grace
/// period; supported release targets (Linux, macOS) use the exact check.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn process_holds_open(_target: &Path) -> std::io::Result<bool> {
    std::thread::sleep(Duration::from_millis(500));
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn absent_store_is_released() {
        let temp = tempfile::tempdir().unwrap();
        wait_for_persistent_release(temp.path().join("never-created"), Duration::ZERO)
            .await
            .unwrap();
    }

    #[cfg(any(target_os = "linux", target_os = "macos"))]
    #[test]
    fn an_open_descriptor_is_observed_and_its_close_is_not_missed() {
        let temp = tempfile::tempdir().unwrap();
        let lock = temp.path().join("LOCK");
        std::fs::write(&lock, b"").unwrap();
        let lock = std::fs::canonicalize(lock).unwrap();
        let held = std::fs::File::open(&lock).unwrap();
        assert!(process_holds_open(&lock).unwrap());
        drop(held);
        assert!(!process_holds_open(&lock).unwrap());
    }

    #[cfg(feature = "rocksdb")]
    #[tokio::test]
    async fn dropped_engine_releases_its_store_before_wait_returns() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("store");
        let db = crate::init_persistent(&path).await.unwrap();
        // The live engine holds the lock: a zero timeout must report it.
        assert!(wait_for_persistent_release(&path, Duration::ZERO)
            .await
            .is_err());
        drop(db);
        wait_for_persistent_release(&path, Duration::from_secs(30))
            .await
            .unwrap();
        // The same-process RocksDB lock is free, so reopening cannot be busy.
        let reopened = crate::init_persistent(&path).await.unwrap();
        drop(reopened);
        wait_for_persistent_release(&path, Duration::from_secs(30))
            .await
            .unwrap();
    }
}
