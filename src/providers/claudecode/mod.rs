//! This module provides the ClaudeCode provider, which is an Anthropic-compatible provider
//! that authenticates via OAuth 2.0 (`Authorization: Bearer <token>`) instead of an x-api-key
//! header, and includes the `anthropic-beta: oauth-2025-04-20` header required by Claude Code.
//!
//! All model types, conversions, and streaming logic are reused from the `anthropic` module.

use crate::core::DynamicModel;
use crate::core::capabilities::ModelName;
use crate::core::client::LanguageModelClient;
use crate::core::utils::validate_base_url;
use crate::error::Error;
use crate::providers::anthropic::{
    ANTHROPIC_API_VERSION, client::AnthropicOptions, settings::AnthropicProviderSettings,
};
use reqwest::header::CONTENT_TYPE;
use serde::Serialize;

// Re-export all Anthropic model capability types so users can do
// `ClaudeCode::<ClaudeSonnet40>::default()` etc.
pub use crate::providers::anthropic::capabilities::*;

// Forward Anthropic model capability markers to ClaudeCode so it has the exact
// same compile-time model capabilities without duplicating model lists.
impl<M> crate::core::capabilities::ToolCallSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::ToolCallSupport,
{
}

impl<M> crate::core::capabilities::ReasoningSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::ReasoningSupport,
{
}

impl<M> crate::core::capabilities::StructuredOutputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::StructuredOutputSupport,
{
}

impl<M> crate::core::capabilities::TextInputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::TextInputSupport,
{
}

impl<M> crate::core::capabilities::VideoInputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::VideoInputSupport,
{
}

impl<M> crate::core::capabilities::AudioInputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::AudioInputSupport,
{
}

impl<M> crate::core::capabilities::ImageInputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::ImageInputSupport,
{
}

impl<M> crate::core::capabilities::TextOutputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::TextOutputSupport,
{
}

impl<M> crate::core::capabilities::VideoOutputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::VideoOutputSupport,
{
}

impl<M> crate::core::capabilities::AudioOutputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::AudioOutputSupport,
{
}

impl<M> crate::core::capabilities::ImageOutputSupport for ClaudeCode<M>
where
    M: ModelName,
    crate::providers::anthropic::Anthropic<M>: crate::core::capabilities::ImageOutputSupport,
{
}

/// The ClaudeCode provider.
///
/// Behaves identically to `Anthropic` but authenticates using OAuth 2.0:
/// - `Authorization: Bearer <token>` instead of `x-api-key`
/// - Adds `anthropic-beta: oauth-2025-04-20`
///
/// The token is read from the `CLAUDE_CODE_API_KEY` environment variable by
/// default, or can be set explicitly via the builder.
#[derive(Debug, Serialize, Clone)]
pub struct ClaudeCode<M: ModelName> {
    /// Configuration settings (base URL, token, etc.).
    pub settings: AnthropicProviderSettings,
    options: AnthropicOptions,
    _phantom: std::marker::PhantomData<M>,
}

// ---------------------------------------------------------------------------
// LanguageModelClient — only the headers() impl differs from Anthropic
// ---------------------------------------------------------------------------

impl<M: ModelName> LanguageModelClient for ClaudeCode<M> {
    type Response = <crate::providers::anthropic::Anthropic<M> as LanguageModelClient>::Response;
    type StreamEvent =
        <crate::providers::anthropic::Anthropic<M> as LanguageModelClient>::StreamEvent;

    fn path(&self) -> String {
        self.settings
            .path
            .clone()
            .unwrap_or_else(|| "/messages".to_string())
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert(
            reqwest::header::AUTHORIZATION,
            format!("Bearer {}", self.settings.api_key).parse().unwrap(),
        );
        headers.insert("anthropic-version", ANTHROPIC_API_VERSION.parse().unwrap());
        headers.insert("anthropic-beta", "oauth-2025-04-20".parse().unwrap());
        headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        let body = serde_json::to_string(&self.options).unwrap();
        reqwest::Body::from(body)
    }

    fn parse_stream_sse(
        event: std::result::Result<reqwest_eventsource::Event, reqwest_eventsource::Error>,
    ) -> crate::error::Result<Self::StreamEvent> {
        crate::providers::anthropic::Anthropic::<M>::parse_stream_sse(event)
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        crate::providers::anthropic::Anthropic::<M>::end_stream(event)
    }
}

// ---------------------------------------------------------------------------
// LanguageModel — delegate to the generated Anthropic impl by sharing the
// same AnthropicOptions / base_url plumbing.
// ---------------------------------------------------------------------------

use crate::core::language_model::{
    LanguageModel, LanguageModelOptions, LanguageModelResponse, ProviderStream,
};
use crate::error::Result;
use async_trait::async_trait;

#[async_trait]
impl<M: ModelName> LanguageModel for ClaudeCode<M> {
    fn name(&self) -> String {
        self.options.model.clone()
    }

    async fn generate_text(
        &mut self,
        options: LanguageModelOptions,
    ) -> Result<LanguageModelResponse> {
        let mut opts: AnthropicOptions = options.into();
        opts.model = self.options.model.clone();
        self.options = opts;
        self.send(self.settings.base_url.clone()).await.map(|resp| {
            // Reuse Anthropic's response-to-LanguageModelResponse mapping by
            // converting through the same fields.
            use crate::core::ToolCallInfo;
            use crate::core::language_model::LanguageModelResponseContentType;
            use crate::core::tools::ToolDetails;
            use crate::extensions::Extensions;
            use crate::providers::anthropic::client::AnthropicContentBlock;
            use crate::providers::anthropic::extensions;

            let mut collected: Vec<LanguageModelResponseContentType> = Vec::new();
            for block in resp.content {
                match block {
                    AnthropicContentBlock::Text { text, .. } => {
                        collected.push(LanguageModelResponseContentType::new(text));
                    }
                    AnthropicContentBlock::Thinking {
                        signature,
                        thinking,
                    } => {
                        let exts = Extensions::default();
                        exts.get_mut::<extensions::AnthropicThinkingMetadata>()
                            .signature = Some(signature);
                        collected.push(LanguageModelResponseContentType::Reasoning {
                            content: thinking,
                            extensions: exts,
                        });
                    }
                    AnthropicContentBlock::RedactedThinking { data } => {
                        collected.push(LanguageModelResponseContentType::Reasoning {
                            content: data,
                            extensions: Extensions::default(),
                        });
                    }
                    AnthropicContentBlock::ToolUse { id, input, name } => {
                        collected.push(LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                            input,
                            tool: ToolDetails {
                                id: id.to_string(),
                                name: name.to_string(),
                            },
                            extensions: Extensions::default(),
                        }));
                    }
                }
            }
            LanguageModelResponse {
                contents: collected,
                usage: Some(resp.usage.into()),
            }
        })
    }

    async fn stream_text(&mut self, options: LanguageModelOptions) -> Result<ProviderStream> {
        let mut opts: AnthropicOptions = options.into();
        opts.stream = Some(true);
        opts.model = self.options.model.clone();
        self.options = opts;

        let max_retries = 5;
        let mut retry_count = 0;
        let mut wait_time = std::time::Duration::from_secs(1);

        let response = loop {
            match self.send_and_stream(self.settings.base_url.clone()).await {
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

        // Delegate stream parsing to the Anthropic language model by temporarily
        // constructing an Anthropic instance with the same settings and streaming
        // from the already-open stream.
        use crate::core::ToolCallInfo;
        use crate::core::language_model::{
            LanguageModelResponseContentType, LanguageModelStreamChunk,
            LanguageModelStreamChunkType,
        };
        use crate::core::messages::AssistantMessage;
        use crate::core::tools::ToolDetails;
        use crate::extensions::Extensions;
        use crate::providers::anthropic::client::{
            AnthropicContentBlock, AnthropicDelta, AnthropicMessageDeltaUsage, AnthropicStreamEvent,
        };
        use crate::providers::anthropic::extensions;
        use futures::StreamExt;
        use std::collections::HashMap;

        #[derive(Default)]
        struct StreamState {
            content_blocks: HashMap<usize, AccumulatedBlock>,
            usage: Option<AnthropicMessageDeltaUsage>,
        }

        #[derive(Debug)]
        enum AccumulatedBlock {
            Text(String),
            Thinking {
                thinking: String,
                signature: Option<String>,
            },
            RedactedThinking(String),
            ToolUse {
                id: String,
                name: String,
                accumulated_json: String,
            },
        }

        let stream = response.scan::<_, Result<Vec<LanguageModelStreamChunk>>, _, _>(
            StreamState::default(),
            |state, evt_res| {
                let unsupported = |event: &str| {
                    vec![LanguageModelStreamChunk::Delta(
                        LanguageModelStreamChunkType::NotSupported(format!("AnthropicStreamEvent::{event}")),
                    )]
                };
                futures::future::ready(match evt_res {
                    Ok(event) => match event {
                        AnthropicStreamEvent::MessageStart { .. } => Some(Ok(vec![
                            LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::Start),
                        ])),
                        AnthropicStreamEvent::ContentBlockStart { index, content_block } => {
                            match content_block {
                                AnthropicContentBlock::Text { .. } => {
                                    state.content_blocks.insert(index, AccumulatedBlock::Text(String::new()));
                                    Some(Ok(unsupported("ContentBlockStart::Text")))
                                }
                                AnthropicContentBlock::Thinking { .. } => {
                                    state.content_blocks.insert(index, AccumulatedBlock::Thinking {
                                        thinking: String::new(),
                                        signature: None,
                                    });
                                    Some(Ok(unsupported("ContentBlockStart::Thinking")))
                                }
                                AnthropicContentBlock::RedactedThinking { data } => {
                                    state.content_blocks.insert(index, AccumulatedBlock::RedactedThinking(data));
                                    Some(Ok(unsupported("ContentBlockStart::RedactedThinking")))
                                }
                                AnthropicContentBlock::ToolUse { id, name, .. } => {
                                    state.content_blocks.insert(index, AccumulatedBlock::ToolUse {
                                        id,
                                        name,
                                        accumulated_json: String::new(),
                                    });
                                    Some(Ok(unsupported("ContentBlockStart::ToolUse")))
                                }
                            }
                        }
                        AnthropicStreamEvent::ContentBlockDelta { index, delta } => {
                            if let Some(block) = state.content_blocks.get_mut(&index) {
                                match (block, delta) {
                                    (AccumulatedBlock::Text(text), AnthropicDelta::TextDelta { text: dt }) => {
                                        text.push_str(&dt);
                                        Some(Ok(vec![LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::Text(dt))]))
                                    }
                                    (AccumulatedBlock::Thinking { thinking, .. }, AnthropicDelta::ThinkingDelta { thinking: dt }) => {
                                        thinking.push_str(&dt);
                                        Some(Ok(vec![LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::Text(dt))]))
                                    }
                                    (AccumulatedBlock::Thinking { signature, .. }, AnthropicDelta::SignatureDelta { signature: ds }) => {
                                        *signature = Some(ds);
                                        Some(Ok(unsupported("SignatureDelta")))
                                    }
                                    (AccumulatedBlock::ToolUse { accumulated_json, .. }, AnthropicDelta::ToolUseDelta { partial_json }) => {
                                        accumulated_json.push_str(&partial_json);
                                        Some(Ok(vec![LanguageModelStreamChunk::Delta(LanguageModelStreamChunkType::ToolCall(partial_json))]))
                                    }
                                    _ => Some(Ok(unsupported("ContentBlockDelta"))),
                                }
                            } else {
                                unreachable!("ClaudeCode accumulator must be initialized on ContentBlockStart")
                            }
                        }
                        AnthropicStreamEvent::ContentBlockStop { .. } => Some(Ok(unsupported("ContentBlockStop"))),
                        AnthropicStreamEvent::MessageDelta { usage, .. } => {
                            state.usage = Some(usage);
                            Some(Ok(unsupported("MessageDelta")))
                        }
                        AnthropicStreamEvent::MessageStop => {
                            let mut collected = vec![];
                            for block in state.content_blocks.values() {
                                match block {
                                    AccumulatedBlock::Text(text) => {
                                        collected.push(LanguageModelResponseContentType::new(text.clone()));
                                    }
                                    AccumulatedBlock::Thinking { thinking, signature } => {
                                        let exts = Extensions::default();
                                        if let Some(sig) = signature {
                                            exts.get_mut::<extensions::AnthropicThinkingMetadata>().signature = Some(sig.clone());
                                        }
                                        collected.push(LanguageModelResponseContentType::Reasoning {
                                            content: thinking.clone(),
                                            extensions: exts,
                                        });
                                    }
                                    AccumulatedBlock::RedactedThinking(data) => {
                                        collected.push(LanguageModelResponseContentType::Reasoning {
                                            content: data.clone(),
                                            extensions: Extensions::default(),
                                        });
                                    }
                                    AccumulatedBlock::ToolUse { id, name, accumulated_json } => {
                                        let json_str = if accumulated_json.trim().is_empty() { "{}" } else { accumulated_json };
                                        if let Ok(input) = serde_json::from_str(json_str) {
                                            collected.push(LanguageModelResponseContentType::ToolCall(ToolCallInfo {
                                                input,
                                                tool: ToolDetails { id: id.clone(), name: name.clone() },
                                                extensions: Extensions::default(),
                                            }));
                                        } else {
                                            collected.push(LanguageModelResponseContentType::NotSupported(
                                                format!("Invalid tool json: {accumulated_json}"),
                                            ));
                                        }
                                    }
                                }
                            }
                            Some(Ok(collected.into_iter().map(|ref c| {
                                LanguageModelStreamChunk::Done(AssistantMessage {
                                    content: c.clone(),
                                    usage: state.usage.clone().map(|u| u.into()),
                                })
                            }).collect()))
                        }
                        AnthropicStreamEvent::Error { error } => Some(Ok(vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::Failed(format!("{}: {}", error.type_, error.message)),
                        )])),
                        AnthropicStreamEvent::NotSupported(txt) => Some(Ok(vec![LanguageModelStreamChunk::Delta(
                            LanguageModelStreamChunkType::NotSupported(txt),
                        )])),
                    },
                    Err(e) => Some(Err(e)),
                })
            },
        );

        Ok(Box::pin(stream))
    }
}

// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

impl<M: ModelName> ClaudeCode<M> {
    /// ClaudeCode provider setting builder.
    pub fn builder() -> ClaudeCodeBuilder<M> {
        ClaudeCodeBuilder::default()
    }
}

impl ClaudeCode<DynamicModel> {
    /// Creates a ClaudeCode provider with a dynamic model name using default settings.
    ///
    /// The OAuth token is read from the `CLAUDE_CODE_API_KEY` environment variable.
    pub fn model_name(name: impl Into<String>) -> Self {
        let settings = default_settings();
        let options = AnthropicOptions::builder()
            .model(name.into())
            .build()
            .unwrap();
        ClaudeCode {
            settings,
            options,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<M: ModelName> Default for ClaudeCode<M> {
    fn default() -> Self {
        let settings = default_settings();
        let options = AnthropicOptions::builder()
            .model(M::MODEL_NAME.to_string())
            .build()
            .unwrap();
        Self {
            settings,
            options,
            _phantom: std::marker::PhantomData,
        }
    }
}

fn default_settings() -> AnthropicProviderSettings {
    AnthropicProviderSettings {
        provider_name: "claudecode".to_string(),
        base_url: "https://api.anthropic.com/v1/".to_string(),
        api_key: std::env::var("CLAUDE_CODE_API_KEY")
            .or_else(|_| std::env::var("ANTHROPIC_API_KEY"))
            .unwrap_or_default(),
        path: None,
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

/// Builder for `ClaudeCode`.
pub struct ClaudeCodeBuilder<M: ModelName> {
    settings: AnthropicProviderSettings,
    options: AnthropicOptions,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelName> Default for ClaudeCodeBuilder<M> {
    fn default() -> Self {
        let settings = default_settings();
        let options = AnthropicOptions::builder()
            .model(M::MODEL_NAME.to_string())
            .build()
            .unwrap();
        Self {
            settings,
            options,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl ClaudeCodeBuilder<DynamicModel> {
    /// Sets the model name from a string.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.options.model = model_name.into();
        self
    }
}

impl<M: ModelName> ClaudeCodeBuilder<M> {
    /// Sets the base URL for the Anthropic API.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Sets the OAuth token used in the `Authorization: Bearer` header.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into();
        self
    }

    /// Sets the name of the provider. Defaults to `"claudecode"`.
    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.settings.provider_name = provider_name.into();
        self
    }

    /// Sets a custom API path, overriding the default `"/messages"`.
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.settings.path = Some(path.into());
        self
    }

    /// Builds the ClaudeCode provider.
    pub fn build(self) -> Result<ClaudeCode<M>> {
        let base_url = validate_base_url(&self.settings.base_url)?;

        if self.settings.api_key.is_empty() {
            return Err(Error::MissingField("api_key".to_string()));
        }

        Ok(ClaudeCode {
            settings: AnthropicProviderSettings {
                base_url,
                ..self.settings
            },
            options: self.options,
            _phantom: std::marker::PhantomData,
        })
    }
}
