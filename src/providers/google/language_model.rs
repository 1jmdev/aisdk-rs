//! Language model implementation for the Google provider.
use crate::core::capabilities::ModelName;
use crate::core::client::Client;
use crate::core::language_model::{
    LanguageModelOptions, LanguageModelResponse, LanguageModelResponseContentType,
    LanguageModelStreamChunk, LanguageModelStreamChunkType, ProviderStream, Usage,
};
use crate::core::messages::AssistantMessage;
use crate::providers::google::{Google, client::types};
use crate::{
    core::{language_model::LanguageModel, tools::ToolCallInfo},
    error::Result,
};
use async_trait::async_trait;
use futures::StreamExt;

#[async_trait]
impl<M: ModelName> LanguageModel for Google<M> {
    fn name(&self) -> String {
        self.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let request: types::GenerateContentRequest = options.into();
        self.options.request = Some(request);
        self.options.streaming = false;

        let response: types::GenerateContentResponse = self.send(&self.settings.base_url).await?;

        let mut collected = Vec::new();
        let usage = response.usage_metadata.map(|u| u.into());

        for candidate in response.candidates {
            for part in candidate.content.parts {
                if let Some(t) = part.text {
                    collected.push(LanguageModelResponseContentType::Text(t));
                }
                if let Some(fc) = part.function_call {
                    let mut tool_info = ToolCallInfo::new(fc.name);
                    tool_info.input(fc.args);
                    collected.push(LanguageModelResponseContentType::ToolCall(tool_info));
                }
            }
        }

        Ok(LanguageModelResponse {
            contents: collected,
            usage,
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let request: types::GenerateContentRequest = options.into();
        self.options.request = Some(request);
        self.options.streaming = true;

        let google_stream = self.send_and_stream(&self.settings.base_url).await?;

        let stream = google_stream.map(|evt_res| match evt_res {
            Ok(types::GoogleStreamEvent::Response(response)) => {
                let mut chunks = Vec::new();
                let usage = response.usage_metadata.clone().map(Usage::from);

                for candidate in &response.candidates {
                    for part in &candidate.content.parts {
                        if let Some(t) = &part.text {
                            chunks.push(LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::Text(t.clone()),
                            ));
                        }
                        if let Some(fc) = &part.function_call {
                            chunks.push(LanguageModelStreamChunk::Delta(
                                LanguageModelStreamChunkType::ToolCall(
                                    serde_json::to_string(&fc).unwrap_or_default(),
                                ),
                            ));
                        }
                    }

                    if candidate.finish_reason.is_some() {
                        let content = if let Some(fc) = candidate
                            .content
                            .parts
                            .iter()
                            .find_map(|p| p.function_call.as_ref())
                        {
                            let mut tool_info = ToolCallInfo::new(fc.name.clone());
                            tool_info.input(fc.args.clone());
                            LanguageModelResponseContentType::ToolCall(tool_info)
                        } else {
                            let text = candidate
                                .content
                                .parts
                                .iter()
                                .filter_map(|p| p.text.clone())
                                .collect::<Vec<_>>()
                                .join("");
                            LanguageModelResponseContentType::Text(text)
                        };

                        chunks.push(LanguageModelStreamChunk::Done(AssistantMessage {
                            content,
                            usage: usage.clone(),
                        }));
                    }
                }
                Ok(chunks)
            }
            Ok(types::GoogleStreamEvent::NotSupported(msg)) => {
                Ok(vec![LanguageModelStreamChunk::Delta(
                    LanguageModelStreamChunkType::NotSupported(msg),
                )])
            }
            Err(e) => Err(e),
        });

        Ok(Box::pin(stream))
    }
}
