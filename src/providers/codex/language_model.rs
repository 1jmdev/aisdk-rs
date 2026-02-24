//! Language model implementation for the Codex provider.

use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream, Usage,
};
use crate::core::messages::AssistantMessage;
use crate::providers::codex::{Codex, client};
use crate::providers::openai::client::OpenAILanguageModelOptions;
use crate::providers::openai::client::types;
use crate::{
    core::{language_model::LanguageModel, tools::ToolCallInfo},
    error::{Error, Result},
};
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for Codex<M> {
    /// Returns the name of the model.
    fn name(&self) -> String {
        self.lm_options.model.clone()
    }

    /// Generates text using the Codex provider.
    async fn generate_text(
        &mut self,
        _options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        Err(Error::Other(
            "Codex provider supports streaming only; use stream_text()".to_string(),
        ))
    }

    /// Streams text using the Codex provider.
    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let mut options: OpenAILanguageModelOptions = options.into();

        options.model = self.lm_options.model.to_string();
        options.stream = Some(true);

        self.lm_options = options;

        let max_retries = 5;
        let mut retry_count = 0;
        let mut wait_time = std::time::Duration::from_secs(1);

        let codex_stream = loop {
            match self.send_and_stream(&self.settings.base_url).await {
                Ok(stream) => break stream,
                Err(crate::error::Error::ApiError {
                    status_code: Some(status),
                    ..
                }) if status == reqwest::StatusCode::TOO_MANY_REQUESTS
                    && retry_count < max_retries =>
                {
                    retry_count += 1;
                    tokio::time::sleep(wait_time).await;
                    wait_time *= 2;
                    continue;
                }
                Err(e) => return Err(e),
            }
        };

        let stream = codex_stream.map(|evt_res| match evt_res {
            Ok(client::OpenAiStreamEvent::ResponseOutputTextDelta { delta, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Text(delta),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseReasoningSummaryTextDelta { delta, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Reasoning(delta),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseCompleted { response, .. }) => {
                let mut result: Vec<LanguageModelStreamChunk> = Vec::new();

                let usage: Usage = response.usage.unwrap_or_default().into();
                let output = response.output.unwrap_or_default();

                for msg in output {
                    match &msg {
                        types::MessageItem::OutputMessage { content, .. } => {
                            if let Some(types::OutputContent::OutputText { text, .. }) =
                                content.first()
                            {
                                result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::new(text.clone()),
                                    usage: Some(usage.clone()),
                                }));
                            }
                        }
                        types::MessageItem::Reasoning { summary, .. } => {
                            if let Some(types::ReasoningSummary { text, .. }) = summary.first() {
                                result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: LanguageModelResponseContentType::Reasoning {
                                        content: text.to_owned(),
                                        extensions: crate::extensions::Extensions::default(),
                                    },
                                    usage: Some(usage.clone()),
                                }));
                            }
                        }
                        types::MessageItem::FunctionCall {
                            call_id,
                            name,
                            arguments,
                            ..
                        } => {
                            let mut tool_info = ToolCallInfo::new(name.clone());
                            tool_info.id(call_id.clone());
                            tool_info.input(serde_json::from_str(arguments).unwrap_or_default());

                            result.push(LanguageModelStreamChunk::Done(AssistantMessage {
                                content: LanguageModelResponseContentType::ToolCall(tool_info),
                                usage: Some(usage.clone()),
                            }));
                        }
                        _ => {}
                    }
                }

                Ok(result)
            }
            Ok(client::OpenAiStreamEvent::ResponseIncomplete { response, .. }) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Incomplete(
                        response
                            .incomplete_details
                            .map(|d| d.reason)
                            .unwrap_or("Unknown".to_string()),
                    ),
                )])
            }
            Ok(client::OpenAiStreamEvent::ResponseError { code, message, .. }) => {
                let reason = format!("{}: {}", code.unwrap_or("unknown".to_string()), message);
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::Failed(reason),
                )])
            }
            Ok(evt) => Ok(vec![LanguageModelStreamChunk::Delta(
                LanguageModelStreamChunkType::NotSupported(format!("{evt:?}")),
            )]),
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}
