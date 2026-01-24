//! Embedding model implementation for the OpenAI provider.

use crate::{
    core::{
        capabilities::ModelName,
        client::EmbeddingClient,
        embedding_model::{EmbeddingModel, EmbeddingModelResponse},
    },
    providers::openai::OpenAI,
};
use async_trait::async_trait;

#[derive(Debug, Clone)]
/// Settings for OpenAI that are specific to embedding models.
pub struct OpenAIEmbeddingModelOptions {}

#[async_trait]
impl<M: ModelName> EmbeddingModel for OpenAI<M> {
    async fn embed(&self) -> EmbeddingModelResponse {
        let response = self.send(&self.settings.base_url).await.unwrap();

        let data = response.data.clone();
        let data: Vec<Vec<f32>> = data.into_iter().map(|e| e.embedding).collect();

        data
    }
}
