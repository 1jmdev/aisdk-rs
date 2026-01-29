//! This module provides the Amazon Bedrock provider, wrapping OpenAI Chat Completions for Bedrock requests.

pub mod capabilities;

// Generate the settings module
crate::openai_compatible_settings!(
    AmazonBedrockProviderSettings,
    AmazonBedrockProviderSettingsBuilder,
    "AmazonBedrock",
    "https://bedrock-runtime.us-east-1.amazonaws.com/openai/",
    "BEDROCK_API_KEY"
);

// Generate the provider struct and builder
crate::openai_compatible_provider!(
    AmazonBedrock,
    AmazonBedrockBuilder,
    AmazonBedrockProviderSettings,
    "Amazon Bedrock",
    "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic_claude_3_5_sonnet_v1_0()"
);

// Generate the language model implementation
crate::openai_compatible_language_model!(AmazonBedrock, "Amazon Bedrock");
