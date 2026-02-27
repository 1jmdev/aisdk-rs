//! This module provides the Codex client, an HTTP client for interacting with the Codex API.

pub(crate) use crate::providers::openai::client::types::*;

use crate::core::client::LanguageModelClient;
use crate::core::utils::join_url;
use crate::error::Error;
use crate::providers::codex::Codex;
use crate::providers::codex::ModelName;
use futures::{Stream, StreamExt, stream};
use reqwest::IntoUrl;
use reqwest::header::{ACCEPT, CONTENT_TYPE};
use reqwest_eventsource::Event;
use serde_json::json;
use std::pin::Pin;
use tokio::sync::mpsc;

impl<M: ModelName> LanguageModelClient for Codex<M> {
    type Response = OpenAIResponse;
    type StreamEvent = OpenAiStreamEvent;

    fn path(&self) -> String {
        self.settings
            .path
            .clone()
            .unwrap_or_else(|| "/responses".to_string())
    }

    fn method(&self) -> reqwest::Method {
        reqwest::Method::POST
    }

    fn headers(&self) -> reqwest::header::HeaderMap {
        let mut default_headers = reqwest::header::HeaderMap::new();
        default_headers.insert(CONTENT_TYPE, "application/json".parse().unwrap());
        default_headers.insert(ACCEPT, "text/event-stream".parse().unwrap());
        let api_key = self.settings.api_key.trim();
        default_headers.insert(
            "Authorization",
            format!("Bearer {}", api_key).parse().unwrap(),
        );

        default_headers
    }

    fn query_params(&self) -> Vec<(&str, &str)> {
        Vec::new()
    }

    fn body(&self) -> reqwest::Body {
        let mut body = serde_json::to_value(&self.lm_options).unwrap_or_else(|_| json!({}));

        if let Some(obj) = body.as_object_mut() {
            obj.insert(
                "instructions".to_string(),
                json!(self.settings.instructions.clone()),
            );
            obj.insert("store".to_string(), json!(false));
        }

        reqwest::Body::from(serde_json::to_vec(&body).unwrap_or_default())
    }

    fn parse_stream_sse(
        event: std::result::Result<Event, reqwest_eventsource::Error>,
    ) -> crate::error::Result<Self::StreamEvent> {
        match event {
            Ok(event) => match event {
                Event::Open => Ok(OpenAiStreamEvent::NotSupported("{}".to_string())),
                Event::Message(msg) => {
                    if msg.data.trim() == "[DONE]" || msg.data.is_empty() {
                        return Ok(OpenAiStreamEvent::NotSupported("[END]".to_string()));
                    }

                    let value: serde_json::Value =
                        serde_json::from_str(&msg.data).map_err(|e| Error::ApiError {
                            status_code: None,
                            details: format!("Invalid JSON in SSE data: {e}"),
                        })?;

                    Ok(serde_json::from_value::<OpenAiStreamEvent>(value)
                        .unwrap_or(OpenAiStreamEvent::NotSupported(msg.data)))
                }
            },
            Err(e) => {
                let status_code = match &e {
                    reqwest_eventsource::Error::InvalidStatusCode(status, _) => Some(*status),
                    _ => None,
                };
                Err(Error::ApiError {
                    status_code,
                    details: e.to_string(),
                })
            }
        }
    }

    fn end_stream(event: &Self::StreamEvent) -> bool {
        matches!(event, OpenAiStreamEvent::ResponseCompleted { .. })
            || matches!(event, OpenAiStreamEvent::NotSupported(json) if json == "[END]")
            || matches!(event, OpenAiStreamEvent::ResponseError { .. })
    }

    async fn send_and_stream(
        &self,
        base_url: impl IntoUrl,
    ) -> crate::error::Result<
        Pin<Box<dyn Stream<Item = crate::error::Result<Self::StreamEvent>> + Send>>,
    >
    where
        Self::StreamEvent: Send + 'static,
        Self: Sync,
    {
        let client = reqwest::Client::new();
        let url = join_url(base_url, &self.path())?;
        let method = self.method();
        let headers = self.headers();
        let query_params = self.query_params();
        let body = self.body();
        let body_bytes = body.as_bytes().map_or_else(Vec::new, |b| b.to_vec());

        let response = client
            .request(method.clone(), url.clone())
            .headers(headers.clone())
            .query(&query_params)
            .body(reqwest::Body::from(body_bytes.clone()))
            .send()
            .await
            .map_err(|e| Error::ApiError {
                status_code: e.status(),
                details: format!("SSE stream request failed: {e}"),
            })?;

        let status = response.status();
        if !status.is_success() {
            let text = response
                .text()
                .await
                .unwrap_or_else(|err| format!("<failed to read body: {err}>"));
            return Err(Error::ApiError {
                status_code: Some(status),
                details: text,
            });
        }

        let (tx, rx) = mpsc::unbounded_channel::<crate::error::Result<OpenAiStreamEvent>>();
        let mut bytes = response.bytes_stream();

        tokio::spawn(async move {
            let mut buffer = String::new();
            loop {
                match bytes.next().await {
                    Some(Ok(chunk)) => {
                        let s = String::from_utf8_lossy(&chunk);
                        buffer.push_str(&s);

                        while let Some(idx) = buffer.find("\n\n") {
                            let raw_event = buffer[..idx].to_string();
                            buffer.drain(..idx + 2);

                            let mut data_lines: Vec<String> = Vec::new();
                            for line in raw_event.lines() {
                                let line = line.trim_end_matches('\r');
                                if let Some(rest) = line.strip_prefix("data:") {
                                    data_lines.push(rest.trim_start().to_string());
                                }
                            }

                            if data_lines.is_empty() {
                                continue;
                            }

                            let data = data_lines.join("\n");

                            let event = if data.trim() == "[DONE]" || data.trim().is_empty() {
                                OpenAiStreamEvent::NotSupported("[END]".to_string())
                            } else {
                                serde_json::from_str::<OpenAiStreamEvent>(&data)
                                    .unwrap_or(OpenAiStreamEvent::NotSupported(data))
                            };

                            if tx.send(Ok(event.clone())).is_err() {
                                return;
                            }
                            if matches!(
                                event,
                                OpenAiStreamEvent::ResponseCompleted { .. }
                                    | OpenAiStreamEvent::ResponseError { .. }
                            ) {
                                return;
                            }
                        }
                    }
                    Some(Err(e)) => {
                        let _ = tx.send(Err(Error::ApiError {
                            status_code: None,
                            details: format!("SSE body stream error: {e}"),
                        }));
                        return;
                    }
                    None => {
                        if !buffer.trim().is_empty() {
                            let trailing = buffer.trim().to_string();
                            let event = if trailing == "[DONE]" {
                                OpenAiStreamEvent::NotSupported("[END]".to_string())
                            } else {
                                serde_json::from_str::<OpenAiStreamEvent>(&trailing)
                                    .unwrap_or(OpenAiStreamEvent::NotSupported(trailing))
                            };
                            let _ = tx.send(Ok(event));
                        }
                        return;
                    }
                }
            }
        });

        let event_stream = stream::unfold(rx, |mut rx| async {
            rx.recv().await.map(|item| (item, rx))
        });

        Ok(Box::pin(event_stream))
    }
}
