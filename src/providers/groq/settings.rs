//! Defines the settings for the Groq provider.

use crate::{
    core::capabilities::ModelName,
    error::Error,
    providers::{groq::Groq, openai::OpenAI},
};

/// Settings for the Groq provider (delegates to OpenAI).
#[derive(Debug, Clone)]
pub struct GroqProviderSettings;

impl GroqProviderSettings {
    /// Creates a new builder for GroqSettings.
    pub fn builder<M: ModelName>() -> GroqProviderSettingsBuilder<M> {
        GroqProviderSettingsBuilder::default()
    }
}

pub struct GroqProviderSettingsBuilder<M: ModelName> {
    /// The base URL for the Groq API.
    base_url: Option<String>,

    /// The API key for the Groq API.
    api_key: Option<String>,

    /// The name of the provider. Defaults to "groq".
    provider_name: Option<String>,

    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelName> GroqProviderSettingsBuilder<M> {
    /// Sets the base URL for the Groq API.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = Some(base_url.into());
        self
    }

    /// Sets the API key for the Groq API.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the name of the provider. Defaults to "groq".
    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = Some(provider_name.into());
        self
    }

    /// Builds the Groq provider settings.
    pub fn build(self) -> Result<Groq<M>, Error> {
        let openai = OpenAI::builder()
            .base_url(
                self.base_url
                    .unwrap_or_else(|| "https://api.groq.com/openai/v1".to_string()),
            )
            .api_key(
                self.api_key
                    .unwrap_or_else(|| std::env::var("GROQ_API_KEY").unwrap_or_default()),
            )
            .provider_name(self.provider_name.unwrap_or_else(|| "groq".to_string()))
            .build()?;

        Ok(Groq { inner: openai })
    }
}

impl<M: ModelName> Default for GroqProviderSettingsBuilder<M> {
    /// Returns the default settings for the Groq provider.
    fn default() -> Self {
        Self {
            base_url: Some("https://api.groq.com/openai/v1".to_string()),
            api_key: Some(std::env::var("GROQ_API_KEY").unwrap_or_default()),
            provider_name: Some("groq".to_string()),
            _phantom: std::marker::PhantomData,
        }
    }
}
