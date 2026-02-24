//! Defines the settings for the Codex provider.

use derive_builder::Builder;

#[derive(Debug, Clone, Builder)]
#[builder(setter(into), default)]
/// Settings for the Codex provider.
pub struct CodexProviderSettings {
    /// The name of the provider. Defaults to "codex".
    pub provider_name: String,

    /// The API base URL for the Codex API.
    pub base_url: String,

    /// The API key for the Codex API.
    pub api_key: String,

    /// Custom API path override. When set, this path is used instead of the
    /// provider's default path ("/responses").
    pub path: Option<String>,

    /// Instructions field injected into each request body.
    pub instructions: String,
}

impl Default for CodexProviderSettings {
    /// Returns the default settings for the Codex provider.
    fn default() -> Self {
        Self {
            provider_name: "codex".to_string(),
            base_url: "https://chatgpt.com/backend-api/codex".to_string(),
            api_key: std::env::var("CODEX_API_KEY")
                .or_else(|_| std::env::var("OPENAI_API_KEY"))
                .map(|v| v.trim().to_string())
                .unwrap_or_default(),
            path: Some("/responses".to_string()),
            instructions: "".to_string(),
        }
    }
}

impl CodexProviderSettings {
    /// Creates a new builder for `CodexProviderSettings`.
    pub fn builder() -> CodexProviderSettingsBuilder {
        CodexProviderSettingsBuilder::default()
    }
}
