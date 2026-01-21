use crate::core::embedding_model::{EmbeddingModel, EmbeddingModelResponse};
use derive_builder::Builder;

/// OpenAI Embeddings
#[derive(Builder, Debug, Clone)]
#[allow(dead_code)]
pub struct EmbeddingModelRequest<M: EmbeddingModel> {
    /// Specific OpenAI model to use
    pub model: M,
    /// The input text to generate embeddings for
    pub input: Vec<String>,
}

#[allow(dead_code)]
impl<M: EmbeddingModel> EmbeddingModelRequest<M> {
    /// Returns the OpenAI Embeddings builder.
    pub fn builder() -> EmbeddingModelRequestBuilder<M> {
        EmbeddingModelRequestBuilder::default()
    }

    pub fn embed(&self) -> EmbeddingModelResponse {
        todo!()
    }
}
