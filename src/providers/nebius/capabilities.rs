//! Capabilities for nebius models.
//!
//! This module defines model types and their capabilities for nebius providers.
//! Users can implement additional traits on custom models.

use crate::core::capabilities::*;
use crate::model_capabilities;
use crate::providers::nebius::Nebius;

model_capabilities! {
    provider: Nebius,
    models: {
        NousresearchHermes4405b {
            model_name: "NousResearch/hermes-4-405b",
            constructor_name: nousresearch_hermes_4_405b,
            display_name: "Hermes-4 405B",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        NousresearchHermes470b {
            model_name: "NousResearch/hermes-4-70b",
            constructor_name: nousresearch_hermes_4_70b,
            display_name: "Hermes 4 70B",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        DeepseekAiDeepseekV3 {
            model_name: "deepseek-ai/deepseek-v3",
            constructor_name: deepseek_ai_deepseek_v3,
            display_name: "DeepSeek V3",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        MetaLlamaLlama3370bInstructBase {
            model_name: "meta-llama/llama-3.3-70b-instruct-base",
            constructor_name: meta_llama_llama_3_3_70b_instruct_base,
            display_name: "Llama-3.3-70B-Instruct (Base)",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        MetaLlamaLlama3370bInstructFast {
            model_name: "meta-llama/llama-3.3-70b-instruct-fast",
            constructor_name: meta_llama_llama_3_3_70b_instruct_fast,
            display_name: "Llama-3.3-70B-Instruct (Fast)",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        MetaLlamaLlama31405bInstruct {
            model_name: "meta-llama/llama-3_1-405b-instruct",
            constructor_name: meta_llama_llama_3_1_405b_instruct,
            display_name: "Llama 3.1 405B Instruct",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        MoonshotaiKimiK2Instruct {
            model_name: "moonshotai/kimi-k2-instruct",
            constructor_name: moonshotai_kimi_k2_instruct,
            display_name: "Kimi K2 Instruct",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        NvidiaLlama31NemotronUltra253bV1 {
            model_name: "nvidia/llama-3_1-nemotron-ultra-253b-v1",
            constructor_name: nvidia_llama_3_1_nemotron_ultra_253b_v1,
            display_name: "Llama 3.1 Nemotron Ultra 253B v1",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        OpenaiGptOss120b {
            model_name: "openai/gpt-oss-120b",
            constructor_name: openai_gpt_oss_120b,
            display_name: "GPT OSS 120B",
            capabilities: [ImageInputSupport, ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        OpenaiGptOss20b {
            model_name: "openai/gpt-oss-20b",
            constructor_name: openai_gpt_oss_20b,
            display_name: "GPT OSS 20B",
            capabilities: [ImageInputSupport, ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        QwenQwen3235bA22bInstruct2507 {
            model_name: "qwen/qwen3-235b-a22b-instruct-2507",
            constructor_name: qwen_qwen3_235b_a22b_instruct_2507,
            display_name: "Qwen3 235B A22B Instruct 2507",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        QwenQwen3235bA22bThinking2507 {
            model_name: "qwen/qwen3-235b-a22b-thinking-2507",
            constructor_name: qwen_qwen3_235b_a22b_thinking_2507,
            display_name: "Qwen3 235B A22B Thinking 2507",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        QwenQwen3Coder480bA35bInstruct {
            model_name: "qwen/qwen3-coder-480b-a35b-instruct",
            constructor_name: qwen_qwen3_coder_480b_a35b_instruct,
            display_name: "Qwen3 Coder 480B A35B Instruct",
            capabilities: [TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        ZaiOrgGlm45 {
            model_name: "zai-org/glm-4.5",
            constructor_name: zai_org_glm_4_5,
            display_name: "GLM 4.5",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
        ZaiOrgGlm45Air {
            model_name: "zai-org/glm-4.5-air",
            constructor_name: zai_org_glm_4_5_air,
            display_name: "GLM 4.5 Air",
            capabilities: [ReasoningSupport, TextInputSupport, TextOutputSupport, ToolCallSupport]
        },
    }
}
