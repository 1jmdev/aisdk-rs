//! Client implementation for the Google provider.
use crate::core::client::Client;
use crate::error::{Error, Result};
use crate::providers::google::{Google, ModelName};
use derive_builder::Builder;
use futures::Stream;
use futures::StreamExt;
use reqwest::header::CONTENT_TYPE;
use reqwest_eventsource::{Event, RequestBuilderExt};
use serde::{Deserialize, Serialize};
use std::pin::Pin;

pub(crate) mod types;

#[derive(Debug, Default, Clone, Serialize, Deserialize, Builder)]
#[builder(setter(into), build_fn(error = "Error"))]
pub(crate) struct GoogleOptions {
    pub(crate) model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub(crate) request: Option<types::GenerateContentRequest>,
    #[serde(skip)]
    #[builder(default)]
    pub(crate) streaming: bool,
}

impl GoogleOptions {
    pub(crate) fn builder() -> GoogleOptionsBuilder {
        GoogleOptionsBuilder::default()
    }
}

impl<M: ModelName> Client for Google<M> {
    type Response = types::GenerateContentResponse;
    type StreamEvent = types::GoogleStreamEvent;

    fn path(&self) -> &str {
        // Path is handled dynamically in overridden send/send_and_stream
        ""
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut headers = reqwest::header::HeaderMap::new();
        headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        headers.insert("x-goog-api-key", self.settings.api_key.parse().unwrap());
        headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        if let Some(request) = &self.options.request {
            let body = serde_json::to_string(request).unwrap();
            reqwest::Body::from(body)
        } else {
            reqwest::Body::from("{}")
        }
    }

    async fn send(&self, base_url: impl reqwest::IntoUrl) -> Result<Self::Response> {
        let client = reqwest::Client::new();
        let base_url = base_url
            .into_url()
            .map_err(|_| Error::InvalidInput("Invalid base URL".into()))?;

        let path = format!("models/{}:generateContent", self.options.model);
        let url = base_url
            .join(&path)
            .map_err(|_| Error::InvalidInput("Failed to join base URL and path".into()))?;

        let resp = client
            .request(self.method(), url)
            .headers(self.headers())
            .query(&self.query_params())
            .body(self.body())
            .send()
            .await
            .map_err(|e| Error::ApiError(e.to_string()))?;

        let status = resp.status();
        let body = resp
            .text()
            .await
            .map_err(|e| Error::ApiError(e.to_string()))?;

        if !status.is_success() {
            println!("DEBUG: Google API Error ({}): {}", status, body);
            return Err(Error::ApiError(format!(
                "Status: {}, Body: {}",
                status, body
            )));
        }

        serde_json::from_str::<Self::Response>(&body).map_err(|e| {
            println!("DEBUG: Google Decoding Error: {}, Body: {}", e, body);
            Error::ApiError(format!("Decoding error: {}, Body: {}", e, body))
        })
    }

    async fn send_and_stream(
        &self,
        base_url: impl reqwest::IntoUrl,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<Self::StreamEvent>> + Send>>>
    where
        Self::StreamEvent: Send + 'static,
        Self: Sync,
    {
        let client = reqwest::Client::new();
        let base_url = base_url
            .into_url()
            .map_err(|_| Error::InvalidInput("Invalid base URL".into()))?;

        let path = format!("models/{}:streamGenerateContent", self.options.model);
        let mut url = base_url
            .join(&path)
            .map_err(|_| Error::InvalidInput("Failed to join base URL and path".into()))?;

        url.set_query(Some("alt=sse"));

        let events_stream = client
            .request(self.method(), url)
            .headers(self.headers())
            .query(&self.query_params())
            .body(self.body())
            .eventsource()
            .map_err(|e| Error::ApiError(format!("SSE stream error: {}", e)))?;

        let mapped_stream = events_stream.map(|event_result| Self::parse_stream_sse(event_result));
        let ended = std::sync::Arc::new(std::sync::Mutex::new(false));

        let stream = mapped_stream.scan(ended, |ended, res| {
            let mut ended = ended.lock().unwrap();
            if *ended {
                return futures::future::ready(None);
            }
            *ended = res.as_ref().map_or(true, |evt| Self::end_stream(evt));
            futures::future::ready(Some(res))
        });

        Ok(Box::pin(stream))
    }

    fn parse_stream_sse(
        event: std::result::Result<Event, reqwest_eventsource::Error>,
    ) -> Result<Self::StreamEvent> {
        match event {
            Ok(event) => match event {
                Event::Open => Ok(types::GoogleStreamEvent::NotSupported("{}".to_string())),
                Event::Message(msg) => {
                    let value: serde_json::Value = serde_json::from_str(&msg.data)
                        .map_err(|e| Error::ApiError(format!("Invalid JSON in SSE data: {}", e)))?;

                    Ok(
                        serde_json::from_value::<types::GenerateContentResponse>(value)
                            .map(types::GoogleStreamEvent::Response)
                            .unwrap_or(types::GoogleStreamEvent::NotSupported(msg.data)),
                    )
                }
            },
            Err(e) => Err(Error::ApiError(e.to_string())),
        }
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        match event {
            types::GoogleStreamEvent::Response(resp) => {
                resp.candidates.iter().any(|c| c.finish_reason.is_some())
            }
            _ => false,
        }
    }
}
