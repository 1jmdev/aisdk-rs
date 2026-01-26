use crate::core::embedding_model::{EmbeddingModel, EmbeddingModelOptions, EmbeddingModelResponse};
use crate::error::Result;
use derive_builder::Builder;

/// Options for embedding generation requests to be used by `embed`.
#[derive(Builder, Debug, Clone)]
#[allow(dead_code)]
pub struct EmbeddingModelRequest<M: EmbeddingModel> {
    /// Specific OpenAI model to use
    pub model: M,
    /// The input text to generate embeddings for
    pub input: EmbeddingModelOptions,
}

#[allow(dead_code)]
impl<M: EmbeddingModel> EmbeddingModelRequest<M> {
    /// Returns the OpenAI Embeddings builder.
    pub fn builder() -> EmbeddingModelRequestBuilder<M> {
        EmbeddingModelRequestBuilder::default()
    }

    /// Generates embeddings for the input text.
    ///
    /// # Returns
    ///
    /// A Result containing a vector of embedding vectors, where each embedding is a vector of floats,
    /// or an error if the embedding request fails.
    pub async fn embed(&self) -> Result<EmbeddingModelResponse> {
        self.model.embed(self.input.clone()).await
    }
}
