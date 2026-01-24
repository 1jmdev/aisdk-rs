//! Embedding model
//! TODO: add more doc

mod request;

use async_trait::async_trait;

/// The options for embedding requests.
pub type EmbeddingModelOptions = Vec<String>;

/// The core trait abstracting the capabilities of an embedding model.
#[async_trait]
pub trait EmbeddingModel: Clone + Send + Sync + std::fmt::Debug + 'static {
    /// Embeds a text input into a vector of floats.
    async fn embed(&self) -> EmbeddingModelResponse;
}

/// The response type for embedding requests.
pub type EmbeddingModelResponse = Vec<Vec<f32>>;
