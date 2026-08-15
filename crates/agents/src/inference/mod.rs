//! Provider-facing inference boundaries.
//!
//! `traits` is the provider-neutral contract used by agents. `factory`
//! selects configured implementations, and `tei`, `tgi`, and `ollama` expose
//! provider-specific compatibility surfaces. The initial v0.2 move keeps the
//! existing request/response implementation together in `providers` so this
//! topology change cannot alter request semantics; subsequent cleanup can
//! move those private DTOs without changing callers.

pub mod providers;

/// Provider-neutral embedding and entity-extraction contracts.
pub mod traits {
    pub use super::providers::{Embedder, EntityExtractor, SharedEmbedder, SharedEntityExtractor};
}

/// Construction and configuration of supported inference providers.
pub mod factory {
    pub use super::providers::{
        InferenceCapabilities, InferenceProviderConfig, InferenceProviders,
    };
}

/// Text Embeddings Inference compatibility surface.
pub mod tei {
    pub use super::providers::TeiClient;
}

/// Text Generation Inference compatibility surface.
pub mod tgi {
    pub use super::providers::TgiClient;
}

/// Ollama behavior is selected through [`factory::InferenceProviders`].
pub mod ollama {}

/// Shared agent error compatibility surface.
pub mod errors {
    pub use crate::{AgentError, Result};
}

pub use providers::*;
