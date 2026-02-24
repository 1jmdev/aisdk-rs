//! Codex provider implementation.

pub mod capabilities;
pub mod client;
pub mod language_model;
pub mod settings;

use crate::core::DynamicModel;
use crate::core::capabilities::ModelName;
use crate::core::utils::validate_base_url;
use crate::error::Error;
use crate::providers::codex::settings::CodexProviderSettings;
use crate::providers::openai::client::OpenAILanguageModelOptions;

/// The Codex provider.
#[derive(Debug, Clone)]
pub struct Codex<M: ModelName> {
    /// Configuration settings for the Codex provider.
    pub settings: CodexProviderSettings,
    /// Options for Language Model.
    pub(crate) lm_options: OpenAILanguageModelOptions,
    pub(crate) _phantom: std::marker::PhantomData<M>,
}

impl<M: ModelName> Codex<M> {
    /// Codex provider setting builder.
    pub fn builder() -> CodexBuilder<M> {
        CodexBuilder::default()
    }
}

impl<M: ModelName> Default for Codex<M> {
    /// Creates a new Codex provider with default settings.
    fn default() -> Self {
        let settings = CodexProviderSettings::default();
        let lm_options = OpenAILanguageModelOptions::builder()
            .model(M::MODEL_NAME.to_string())
            .build()
            .unwrap();

        Self {
            settings,
            lm_options,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl Codex<DynamicModel> {
    /// Creates a Codex provider with a dynamic model name using default settings.
    ///
    /// This allows you to specify the model name as a string rather than
    /// using typed constructor methods.
    ///
    /// **WARNING**: when using `DynamicModel`, model capabilities are not validated.
    pub fn model_name(name: impl Into<String>) -> Self {
        let settings = CodexProviderSettings::default();
        let lm_options = OpenAILanguageModelOptions::builder()
            .model(name.into())
            .build()
            .unwrap();

        Codex {
            settings,
            lm_options,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Codex Provider Builder.
pub struct CodexBuilder<M: ModelName> {
    settings: CodexProviderSettings,
    options: OpenAILanguageModelOptions,
    _phantom: std::marker::PhantomData<M>,
}

impl CodexBuilder<DynamicModel> {
    /// Sets the model name from a string. e.g., "gpt-5.3-codex".
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.options.model = model_name.into();
        self
    }
}

impl<M: ModelName> Default for CodexBuilder<M> {
    /// Creates a new Codex provider builder with default settings.
    fn default() -> Self {
        let settings = CodexProviderSettings::default();

        let options = OpenAILanguageModelOptions::builder()
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

impl<M: ModelName> CodexBuilder<M> {
    /// Sets the base URL for the Codex API.
    pub fn base_url(mut self, base_url: impl Into<String>) -> Self {
        self.settings.base_url = base_url.into();
        self
    }

    /// Sets the API key for the Codex API.
    pub fn api_key(mut self, api_key: impl Into<String>) -> Self {
        self.settings.api_key = api_key.into().trim().to_string();
        self
    }

    /// Sets the name of the provider. Defaults to "codex".
    pub fn provider_name(mut self, provider_name: impl Into<String>) -> Self {
        self.settings.provider_name = provider_name.into();
        self
    }

    /// Sets a custom API path, overriding the default ("/responses").
    pub fn path(mut self, path: impl Into<String>) -> Self {
        self.settings.path = Some(path.into());
        self
    }

    /// Sets the request `instructions` field sent to Codex.
    pub fn instructions(mut self, instructions: impl Into<String>) -> Self {
        self.settings.instructions = instructions.into();
        self
    }

    /// Builds the Codex provider.
    pub fn build(self) -> Result<Codex<M>, Error> {
        let base_url = validate_base_url(&self.settings.base_url)?;

        let api_key = self.settings.api_key.trim().to_string();
        if api_key.is_empty() {
            return Err(Error::MissingField("api_key".to_string()));
        }

        let lm_options = OpenAILanguageModelOptions::builder()
            .model(M::MODEL_NAME.to_string())
            .build()
            .unwrap();

        Ok(Codex {
            settings: CodexProviderSettings {
                base_url,
                api_key,
                ..self.settings
            },
            lm_options,
            _phantom: std::marker::PhantomData,
        })
    }
}

// Re-exports Models for convenience
pub use capabilities::*;
