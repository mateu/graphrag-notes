//! Per-request private credential loading and constant-time bearer verification.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs::OpenOptions, io::Read, path::Path};
use subtle::ConstantTimeEq;
use thiserror::Error;

const MAX_CREDENTIAL_FILE_BYTES: u64 = 64 * 1024;
const MAX_CREDENTIALS: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Capability {
    Read,
    Capture,
    /// Create a pending reviewed endpoint proposal; it grants neither acceptance nor rejection.
    Propose,
    Edit,
    Delete,
    Accept,
    Reject,
    Undo,
    Upload,
    Jobs,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Credential {
    pub instance_id: String,
    pub token_sha256: String,
    pub capabilities: Vec<Capability>,
}

impl std::fmt::Debug for Credential {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Credential")
            .field("instance_id", &self.instance_id)
            .field("token_sha256", &"[redacted]")
            .field("capabilities", &self.capabilities)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CredentialFile {
    pub schema_version: u32,
    pub credentials: Vec<Credential>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Principal {
    pub instance_id: String,
    pub capabilities: Vec<Capability>,
}

impl Principal {
    pub fn allows(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }
}

#[derive(Debug, Error)]
pub enum CredentialError {
    #[error("Cannot read the credential file; check its location and permissions.")]
    Unavailable,
    #[error(
        "Credentials must be a regular file owned by the service user with permissions 0600; symlinks are refused."
    )]
    UnsafeFile,
    #[error(
        "Invalid credential file; expected schema version 1, unique instance IDs, SHA-256 token hashes and supported capabilities."
    )]
    InvalidFile,
}

impl CredentialFile {
    pub fn load_private(path: &Path) -> Result<Self, CredentialError> {
        let before = std::fs::symlink_metadata(path).map_err(|_| CredentialError::Unavailable)?;
        if !before.is_file() || before.file_type().is_symlink() {
            return Err(CredentialError::UnsafeFile);
        }
        let mut options = OpenOptions::new();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
        }
        let file = options
            .open(path)
            .map_err(|_| CredentialError::Unavailable)?;
        let metadata = file.metadata().map_err(|_| CredentialError::Unavailable)?;
        if !metadata.is_file() || metadata.len() > MAX_CREDENTIAL_FILE_BYTES {
            return Err(CredentialError::UnsafeFile);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            // SAFETY: geteuid has no arguments, side effects, or pointer access.
            let uid = unsafe { libc::geteuid() };
            if metadata.mode() & 0o777 != 0o600
                || metadata.uid() != uid
                || before.dev() != metadata.dev()
                || before.ino() != metadata.ino()
            {
                return Err(CredentialError::UnsafeFile);
            }
        }
        let mut bytes = Vec::new();
        file.take(MAX_CREDENTIAL_FILE_BYTES + 1)
            .read_to_end(&mut bytes)
            .map_err(|_| CredentialError::Unavailable)?;
        if bytes.len() as u64 > MAX_CREDENTIAL_FILE_BYTES {
            return Err(CredentialError::UnsafeFile);
        }
        let document: Self =
            serde_json::from_slice(&bytes).map_err(|_| CredentialError::InvalidFile)?;
        document.validate()?;
        Ok(document)
    }

    fn validate(&self) -> Result<(), CredentialError> {
        if self.schema_version != 1
            || self.credentials.is_empty()
            || self.credentials.len() > MAX_CREDENTIALS
        {
            return Err(CredentialError::InvalidFile);
        }
        let mut names = BTreeSet::new();
        let mut hashes = BTreeSet::new();
        for credential in &self.credentials {
            let name = &credential.instance_id;
            if name.is_empty()
                || name.len() > 64
                || !name
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
                || !names.insert(name)
                || decode_hash(&credential.token_sha256).is_none()
                || !hashes.insert(&credential.token_sha256)
                || credential.capabilities.is_empty()
                || (credential
                    .capabilities
                    .iter()
                    .copied()
                    .collect::<BTreeSet<_>>()
                    .len()
                    != credential.capabilities.len())
            {
                return Err(CredentialError::InvalidFile);
            }
        }
        Ok(())
    }

    /// No raw bearer token is retained in the credential store or principal.
    pub fn authenticate(&self, token: &str) -> Option<Principal> {
        if !(32..=512).contains(&token.len()) || !token.bytes().all(|byte| byte.is_ascii_graphic())
        {
            return None;
        }
        let supplied: [u8; 32] = Sha256::digest(token.as_bytes()).into();
        let mut principal = None;
        // Check every digest instead of exposing the matching credential's index.
        for credential in &self.credentials {
            let expected = decode_hash(&credential.token_sha256)?;
            if bool::from(supplied.ct_eq(&expected)) {
                principal = Some(Principal {
                    instance_id: credential.instance_id.clone(),
                    capabilities: credential.capabilities.clone(),
                });
            }
        }
        principal
    }
}

fn decode_hash(hash: &str) -> Option<[u8; 32]> {
    if hash.len() != 64
        || !hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return None;
    }
    let mut digest = [0u8; 32];
    for (index, part) in hash.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(part).ok()?;
        digest[index] = u8::from_str_radix(text, 16).ok()?;
    }
    Some(digest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> CredentialFile {
        CredentialFile {
            schema_version: 1,
            credentials: vec![Credential {
                instance_id: "openclaw.office".into(),
                token_sha256: format!(
                    "{:x}",
                    Sha256::digest(b"private-instance-token-at-least-32-bytes")
                ),
                capabilities: vec![Capability::Read],
            }],
        }
    }

    #[test]
    fn digests_authenticate_without_retaining_or_displaying_secrets() {
        let document = document();
        document.validate().unwrap();
        let principal = document
            .authenticate("private-instance-token-at-least-32-bytes")
            .unwrap();
        assert_eq!(principal.instance_id, "openclaw.office");
        assert!(principal.allows(Capability::Read));
        assert!(!principal.allows(Capability::Capture));
        assert!(document
            .authenticate("another-private-instance-token-long-enough")
            .is_none());
        assert!(document.authenticate("short").is_none());
        assert!(document
            .authenticate("private-instance-token-at-least-32-bytes ")
            .is_none());
        assert!(!format!("{document:?}").contains(&document.credentials[0].token_sha256));
        assert!(!serde_json::to_string(&document)
            .unwrap()
            .contains("private-instance-token"));
    }

    #[test]
    fn ambiguous_identities_and_malformed_credentials_are_rejected() {
        let original = document();
        let mut duplicate = original.clone();
        duplicate.credentials.push(original.credentials[0].clone());
        assert!(duplicate.validate().is_err());
        duplicate.credentials[1].instance_id = "second".into();
        assert!(
            duplicate.validate().is_err(),
            "tokens cannot map to multiple principals"
        );
        let mut invalid = original.clone();
        invalid.credentials[0].token_sha256 = "not-a-hash".into();
        assert!(invalid.validate().is_err());
        let mut invalid = original.clone();
        invalid.credentials[0].capabilities = vec![Capability::Read, Capability::Read];
        assert!(invalid.validate().is_err());
        let mut invalid = original;
        invalid.credentials[0].instance_id = "unsafe\nidentity".into();
        assert!(invalid.validate().is_err());
    }

    #[cfg(unix)]
    #[test]
    fn credential_loading_refuses_symlinks_and_exposed_file_permissions() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("credentials.json");
        std::fs::write(&path, serde_json::to_vec(&document()).unwrap()).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(matches!(
            CredentialFile::load_private(&path),
            Err(CredentialError::UnsafeFile)
        ));
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        CredentialFile::load_private(&path).unwrap();
        let link = temp.path().join("link.json");
        symlink(&path, &link).unwrap();
        assert!(matches!(
            CredentialFile::load_private(&link),
            Err(CredentialError::UnsafeFile)
        ));
    }
}
