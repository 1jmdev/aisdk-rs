//! Defines the settings for the OpenAI provider.

use reqwest::{IntoUrl, Url};

use crate::{
    error::Error,
    providers::openai::{ModelName, OpenAI, client::OpenAIOptions},
};

/// Settings for the OpenAI provider.
#[derive(Debug, Clone)]
pub struct OpenAIProviderSettings {
    /// The API base URL for the OpenAI API.
    pub base_url: Url,

    /// The API key for the OpenAI API.
    pub api_key: String,

    /// The name of the provider. Defaults to "openai".
    pub provider_name: String,
}

impl Default for OpenAIProviderSettings {
    /// Returns the default settings for the OpenAI provider.
    fn default() -> Self {
        Self {
            base_url: Url::parse("https://api.openai.com/v1/").unwrap(),
            api_key: std::env::var("OPENAI_API_KEY").unwrap_or_default(),
            provider_name: "openai".to_string(),
        }
    }
}

impl OpenAIProviderSettings {
    /// Creates a new builder for `OpenAISettings`.
    pub fn builder<M: ModelName>() -> OpenAIProviderSettingsBuilder<M> {
        OpenAIProviderSettingsBuilder::default()
    }
}

pub struct OpenAIProviderSettingsBuilder<M: ModelName> {
    /// The base URL for the OpenAI API.
    base_url: Option<Url>,

    /// The API key for the OpenAI API.
    api_key: Option<String>,

    /// The name of the provider. Defaults to "openai".
    provider_name: Option<String>,

    _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelName> OpenAIProviderSettingsBuilder<M> {
    /// Sets the base URL for the OpenAI API.
    pub fn base_url(mut self, base_url: impl IntoUrl) -> Self {
        self.base_url = Some(base_url.into_url().expect("Invalid base URL"));
        self
    }

    /// Sets the API key for the OpenAI API.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    /// Sets the name of the provider. Defaults to "openai".
    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.provider_name = Some(provider_name.into());
        self
    }

    /// Builds the OpenAI provider settings.
    pub fn build(self) -> Result<OpenAI<M>, Error> {
        let settings = OpenAIProviderSettings {
            base_url: self.base_url.expect("Missing base URL"),
            api_key: self.api_key.unwrap_or_default(),
            provider_name: self.provider_name.unwrap_or_else(|| "openai".to_string()),
        };

        let mut options = OpenAIOptions::default();
        options.model = M::MODEL_NAME.to_string();

        Ok(super::OpenAI {
            settings,
            options,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<M: ModelName> Default for OpenAIProviderSettingsBuilder<M> {
    /// Returns the default settings for the OpenAI provider.
    fn default() -> Self {
        let settings = OpenAIProviderSettings::default();
        Self {
            base_url: Some(settings.base_url),
            api_key: Some(settings.api_key),
            provider_name: Some(settings.provider_name),
            _phantom: std::marker::PhantomData,
        }
    }
}
