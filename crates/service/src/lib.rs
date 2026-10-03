//! Authenticated, bounded Streamable HTTP MCP access to shared application operations.
//! Transport identity and request authorization are resolved for each HTTP request.

pub mod credentials;
mod server;
mod tools;

pub use credentials::{Capability, Credential, CredentialFile, Principal};
pub use server::{run, serve, ServiceError, ServiceOptions};
