//! Embedding model implementation for the OpenAI provider.

use crate::{
    core::embedding_model::{EmbeddingModel, EmbeddingModelResponse},
    providers::openai::settings::OpenAIProviderSettings,
};

#[derive(Debug, Clone)]
/// Settings for OpenAI that are specific to embedding models.
pub struct OpenAIEmbeddingModelSettings {}

#[derive(Debug, Clone)]
/// OpenAI Embedding Model
pub struct OpenAIEmbeddingModel {
    settings: OpenAIProviderSettings,
    options: OpenAIEmbeddingModelSettings,
}

impl EmbeddingModel for OpenAIEmbeddingModel {
    fn embed(&self) -> EmbeddingModelResponse {
        todo!()
    }
}
