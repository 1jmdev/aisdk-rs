//! Defines the settings for the Anthropic provider.

use reqwest::{IntoUrl, Url};
use serde::{Deserialize, Serialize};

use crate::{
    core::capabilities::ModelName,
    error::Error,
    providers::anthropic::{Anthropic, AnthropicOptions},
};

/// Settings for the Anthropic provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicProviderSettings {
    /// The API base URL for the Anthropic API.
    pub base_url: Url,

    /// The API key for the Anthropic API.
    pub api_key: String,

    /// The name of the provider.
    pub provider_name: String,
}

impl AnthropicProviderSettings {
    pub fn default() -> Self {
        Self {
            base_url: Url::parse("https://api.anthropic.com/v1/").unwrap(),
            api_key: std::env::var("ANTHROPIC_API_KEY").unwrap_or_default(),
            provider_name: "anthropic".to_string(),
        }
    }

    /// Creates a new builder for `AnthropicProviderSettings`.
    pub fn builder<M: ModelName>() -> AnthropicProviderSettingsBuilder<M> {
        AnthropicProviderSettingsBuilder::default()
    }
}

pub struct AnthropicProviderSettingsBuilder<M: ModelName> {
    base_url: Option<Url>,
    api_key: Option<String>,
    provider_name: Option<String>,
    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelName> AnthropicProviderSettingsBuilder<M> {
    pub fn base_url(mut self, base_url: impl IntoUrl) -> Self {
        self.base_url = Some(base_url.into_url().expect("Invalid base URL"));
        self
    }

    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = Some(provider_name.into());
        self
    }

    pub fn build(self) -> Result<Anthropic<M>, Error> {
        let settings = AnthropicProviderSettings {
            base_url: self.base_url.expect("Missing base URL"),
            api_key: self.api_key.unwrap_or_default(),
            provider_name: self
                .provider_name
                .unwrap_or_else(|| "anthropic".to_string()),
        };

        Ok(Anthropic {
            settings,
            options: AnthropicOptions::builder()
                .model(M::MODEL_NAME.to_string())
                .build()
                .unwrap(),
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<M: ModelName> Default for AnthropicProviderSettingsBuilder<M> {
    fn default() -> Self {
        let settings = AnthropicProviderSettings::default();
        Self {
            base_url: Some(settings.base_url),
            api_key: Some(settings.api_key),
            provider_name: Some(settings.provider_name),
            _phantom: std::marker::PhantomData,
        }
    }
}
