//! This module provides the `Provider` trait, which defines the interface for
//! interacting with different AI providers.

#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAI;

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "anthropic")]
pub use anthropic::Anthropic;

#[cfg(feature = "groq")]
pub mod groq;
#[cfg(feature = "groq")]
pub use groq::Groq;

#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "google")]
pub use google::Google;

#[cfg(feature = "vercel")]
pub mod vercel;
#[cfg(feature = "vercel")]
pub use vercel::Vercel;

#[cfg(feature = "openrouter")]
pub mod openrouter;
#[cfg(feature = "openrouter")]
pub use openrouter::OpenRouter;

// Internal module for OpenAI Chat Completions API compatible providers
#[cfg(feature = "openaichatcompletions")]
pub(crate) mod openai_chat_completions;
