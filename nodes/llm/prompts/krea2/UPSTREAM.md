# Krea-2 Prompting Docs -- Upstream Source

Bundled verbatim from krea-ai/krea-2 docs (Apache-style permissive).

| File                  | Origin                                                                                       |
| --------------------- | -------------------------------------------------------------------------------------------- |
| xpansion.txt       | https://github.com/krea-ai/krea-2/blob/main/docs/expansion.txt (system prompt for LLMs)      |
| prompting.md        | https://github.com/krea-ai/krea-2/blob/main/docs/prompting.md (user-facing T2I guide)         |

The Krea2PromptGenerator ComfyUI node (
odes/llm/krea2_prompt_generator.py) sends
xpansion.txt verbatim as the system prompt, wraps the user prompt in a small aspect-ratio
hint, and lets the configured LLMServiceConnector talk to any OpenAI-chat-compat backend.

Per upstream rule 6, the LLM emits **one cohesive paragraph** as its visible answer -- no
bullets, no JSON, no markdown. The Krea2PromptGenerator postprocess trims the raw response
and strips any leading <think>...</think> block left by reasoning models (e.g. M3, DeepSeek-R1)
so the final string is always the pure paragraph.
