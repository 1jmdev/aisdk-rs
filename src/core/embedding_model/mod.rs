//! Embedding model
//! TODO: add more doc

mod request;

/// The core trait abstracting the capabilities of an embedding model.
pub trait EmbeddingModel: Clone + Send + Sync + std::fmt::Debug + 'static {
    /// Embeds a text input into a vector of floats.
    fn embed(&self) -> EmbeddingModelResponse {
        todo!()
    }
}

/// The response type for embedding requests.
pub type EmbeddingModelResponse = Vec<Vec<f32>>;
