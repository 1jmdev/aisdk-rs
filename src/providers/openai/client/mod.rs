//! This module provides the OpenAI client, an HTTP client for interacting with the OpenAI API.
//! It is a thin wrapper around the `reqwest` crate.
//! HTTP requests have this parts:

pub mod types;

pub use types::*;

use crate::error::Error;
use crate::{core::client::Client, providers::openai::OpenAI};
use derive_builder::Builder;
use reqwest::{self, header::CONTENT_TYPE};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into), build_fn(error = "Error"))]
pub struct OpenAIOptions {
    pub model: String,
    pub input: Input,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<types::TextConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<types::ReasoningConfig>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_output_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<ToolParams>>,
}

impl OpenAIOptions {
    pub fn builder() -> OpenAIOptionsBuilder {
        OpenAIOptionsBuilder::default()
    }
}

impl Client for OpenAI {
    type Response = types::OpenAiResponse;
    type StreamEvent = types::OpenAiStreamEvent;

    fn path(&self) -> &str {
        "/v1/responses"
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        // Default headers
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        // Authorization
        default_headers.insert(
            "Authorization",
            format!("Bearer {}", self.settings.api_key.clone())
                .parse()
                .unwrap(),
        );

        default_headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        // prettified json
        println!(
            "OpenAi Request Body: \n---\n{}\n---",
            serde_json::to_string_pretty(&self.options).unwrap()
        );
        let body = serde_json::to_string(&self.options).unwrap();
        reqwest::Body::from(body)
    }
}
