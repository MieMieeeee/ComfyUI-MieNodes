# LLM Service Platform Selection Guide

> This file is an LLM-readable instruction document. It is intended to be
> loaded into the system prompt (or read on demand) by any assistant that
> helps users choose a `Set*LLMServiceConnector` node and fill in the
> matching API key. Keep it authoritative: when a connector is added,
> removed, or its endpoint changes, this file MUST be updated in lockstep
> with `services/llm.py`.

## 0. How to use this file

You (the assistant) are helping a ComfyUI-MieNodes user pick the right LLM
service connector and figure out where their API key goes. To do that, walk
through the decision tree below **before** recommending a node.

If the user already has an API key, the key prefix alone is enough to
disambiguate. If the user only knows the provider (e.g. "I have a 智谱
account"), use §3 to match provider to connector family.

---

## 1. The 30-second decision tree

```
User has an API key in hand?
├── YES → §2 (match by key prefix)
└── NO  → §3 (match by provider / use case / model preference)
```

After you pick the connector family, jump to §5 (recommended models) and
§6 (common pitfalls).

---

## 2. Pick by API key prefix

| API key starts with | Service tier | Connector node (ComfyUI name) | config_key |
|---|---|---|---|
| `sk-` (and provider = Alibaba Bailian) | Bailian PAYG / 按量计费 | `SetBailianLLMServiceConnector` | `bailian` |
| `sk-sp-` | Bailian Token Plan / 套餐 | `SetBailianTokenPlanLLMServiceConnector` | `bailian_token_plan` |
| `sk-cp-` | Bailian Coding Plan / 编程订阅 | `SetBailianCodingPlanLLMServiceConnector` | `bailian_coding` |
| `eyJ` (and provider = 智谱) | 智谱 BigModel Open Platform | `SetZhiPuLLMServiceConnector` | `zhipu` |
| any other (and provider = 智谱) | 智谱 GLM Coding / Token Plan | `SetZhiPuCodeLLMServiceConnector` | `zhipu_code` |
| `sk-` (and provider = MiniMax) | MiniMax Open Platform | `SetMiniMaxLLMServiceConnector` | `minimax_open_platform` |
| `sk-cp-` (and provider = MiniMax) | MiniMax Token Plan / Coding Plan | `SetMiniMaxTokenPlanLLMServiceConnector` | `minimax` |
| `sk-` (and provider = 小米 MiMo) | MiMo Open Platform | `SetMiMoLLMServiceConnector` | `mimo` |
| `tp-` | MiMo Token Plan / Coding Plan | `SetMiMoTokenPlanLLMServiceConnector` | `mimo_token_plan` |
| `sk-` (and provider = SiliconFlow) | 硅基流动 | `SetSiliconFlowLLMServiceConnector` | `siliconflow` |
| `sk-` (and provider = DeepSeek) | DeepSeek | `SetDeepSeekLLMServiceConnector` | `deepseek` |
| `sk-` (and provider = Kimi) | 月之暗面 Kimi | `SetKimiLLMServiceConnector` | `kimi` |
| `sk-` (and provider = GitHub Models) | GitHub Models | `SetGithubModelsLLMServiceConnector` | `github_models` |
| starts with `AIza` | Google Gemini | `SetGeminiLLMServiceConnector` | `gemini` |
| empty / not applicable | Ollama (local) | `SetOllamaLLMServiceConnector` | `ollama` |
| anything else (custom base URL) | Custom OpenAI-compatible | `SetGeneralLLMServiceConnector` | `openai_compatible` |

> **Note on `sk-` collisions.** The bare `sk-` prefix is shared by many
> vendors (Bailian PAYG, MiniMax Open, MiMo Open, DeepSeek, Kimi,
> SiliconFlow). You cannot distinguish them from the key alone — you must
> ask the user which provider the key was issued from. Bailian specifically
> documents that its PAYG `sk-` keys must NOT be used against Token Plan
> or Coding Plan endpoints and vice versa.

---

## 3. Pick by provider / use case

If the user does not have a key yet, route them by what they want to do:

| User says... | Recommend | Why |
|---|---|---|
| "I have an 阿里云百炼 / Bailian account and want to use Qwen" | `SetBailianLLMServiceConnector` (PAYG) if they have a metered key; otherwise ask whether they subscribed to Token Plan or Coding Plan | Bailian has three independent tiers — PAYG vs Token Plan vs Coding Plan. The keys are NOT interchangeable. |
| "I bought a 阿里云百炼 Token Plan / 套餐 / 千问套餐" | `SetBailianTokenPlanLLMServiceConnector` | Fixed-fee subscription, full multimodal lineup. |
| "I bought a 阿里云百炼 Coding Plan / 编程套餐 / Qwen-Coder 包" | `SetBailianCodingPlanLLMServiceConnector` | Qwen-Coder-only subscription; non-coder models will 400. |
| "I want to run a local model" / "no API key" / "Ollama" | `SetOllamaLLMServiceConnector` | Local; no key needed (key field can stay empty). |
| "I have a vendor not listed" / "I have a weird base URL" | `SetGeneralLLMServiceConnector` | OpenAI-compatible escape hatch — user fills base URL + model themselves. |
| Any other named provider | Use the table in §2 by inferring the key prefix once they have one. |

---

## 4. Endpoint reference (for diagnosis, not selection)

| Connector | Base URL | Notes |
|---|---|---|
| `SetBailianLLMServiceConnector` | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` | PAYG / 按量计费. OpenAI-compatible. Slim payload (no `response_format`). |
| `SetBailianTokenPlanLLMServiceConnector` | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions` | Token Plan / 套餐. MaaS workspace endpoint, NOT on the public `dashscope.aliyuncs.com` host. |
| `SetBailianCodingPlanLLMServiceConnector` | `https://coding.dashscope.aliyuncs.com/v1/chat/completions` | Coding Plan / 编程订阅. Different host again from Token Plan. |
| `SetZhiPuLLMServiceConnector` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | Standard BigModel. |
| `SetZhiPuCodeLLMServiceConnector` | `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` | Coding / Token Plan tier; same host, different path. |
| `SetMiniMaxLLMServiceConnector` | `https://api.minimaxi.com/v1/chat/completions` | Open Platform. |
| `SetMiniMaxTokenPlanLLMServiceConnector` | `https://api.minimaxi.com/v1/chat/completions` | Same URL as Open Platform; the **key prefix** (`sk-cp-`) is what tells the server to bill Token Plan. |
| `SetMiMoLLMServiceConnector` | `https://api.xiaomimimo.com/v1/chat/completions` | Standard. |
| `SetMiMoTokenPlanLLMServiceConnector` | `https://token-plan-cn.xiaomimimo.com/v1/chat/completions` | Different host. |
| `SetSiliconFlowLLMServiceConnector` | `https://api.siliconflow.cn/v1/chat/completions` | Many free-tier models. |
| `SetDeepSeekLLMServiceConnector` | `https://api.deepseek.com/chat/completions` | |
| `SetKimiLLMServiceConnector` | `https://api.moonshot.cn/v1/chat/completions` | |
| `SetGithubModelsLLMServiceConnector` | `https://models.github.ai/inference/chat/completions` | Use a fine-grained PAT. |
| `SetGeminiLLMServiceConnector` | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...` | Image inputs forwarded as `inline_data`. |
| `SetOllamaLLMServiceConnector` | `{host}/v1/chat/completions` | Default `host = http://127.0.0.1:11434`. No key required. |
| `SetGeneralLLMServiceConnector` | user-supplied | For any OpenAI-compatible service not listed. |

---

## 5. Recommended models per connector

These are the **stable family names** baked into the dropdowns. The user
can pick a specific date-suffixed version (`qwen3-max-2025-09-22`) by
switching to `Custom` and pasting the full model id from
https://bailian.console.aliyun.com/.

### Bailian PAYG (`SetBailianLLMServiceConnector`)

- `qwen3.7-max`, `qwen3.7-plus`, `qwen3.6-flash`, `qwen3.6-plus`,
  `qwen3.5-flash`, `qwen3.5-plus`, `qwen-plus`, `qwen-max`, `qwen-flash`,
  `qwen-turbo`, `qwen-long`
- Plus hosted third-party: `glm-5.2`, `glm-5.1`, `glm-5`, `kimi-k2.6`,
  `deepseek-v4-pro`

### Bailian Token Plan (`SetBailianTokenPlanLLMServiceConnector`)

Full multimodal lineup:
- Text: `qwen3-max`, `qwen3-plus`, `qwen3-flash`, `qwen3-turbo`, `qwen3-long`
- Vision: `qwen3-vl-plus`, `qwen3-vl-flash`
- Image gen: `qwen-image`
- Code: `qwen3-coder-plus`, `qwen3-coder-flash`

### Bailian Coding Plan (`SetBailianCodingPlanLLMServiceConnector`)

Qwen-Coder only:
- `qwen3-coder-plus`, `qwen3-coder-flash`, `qwen-coder`

---

## 6. Common pitfalls

### "I plugged my key into the wrong node and got a 401."

Most often this is **Bailian tier confusion**: PAYG `sk-` keys are scoped
to `dashscope.aliyuncs.com`, Token Plan `sk-sp-` keys are scoped to
`token-plan.cn-beijing.maas.aliyuncs.com`, and Coding Plan `sk-cp-` keys
are scoped to `coding.dashscope.aliyuncs.com`. Bailian's docs state the
three namespaces are completely isolated — there is no automatic
redirection. Ask the user which subscription they bought; the key prefix
is the most reliable hint.

### "I picked the right connector but the wrong model."

- Bailian Coding Plan will reject (`HTTP 400`) any non-coder model
  (`qwen3-max`, `qwen3-vl-plus`, etc.). Tell the user to either pick
  from the dropdown or switch to the Token Plan connector for non-code
  tasks.
- Bailian PAYG's `qwen3.7-max` is a placeholder name; the real model id
  may include a date suffix. If the user reports `model not found`, ask
  them to check the console and paste the full id into `custom_model`.

### "I want to use a free model."

Several providers (SiliconFlow, GitHub Models, 智谱 GLM Flash tier, Bailian
`qwen3-flash` at low-volume promo) offer free or near-free tiers. The free
lineup changes often — recommend the user check the provider's pricing
page. Enter the free model id into `custom_model`.

### "I don't want to paste my key into the node every time."

The connector nodes read from `mie_llm_keys.json` when their
`config_key` is set and the key field is left blank. The default
`config_key` for each connector matches its name suffix
(`SetBailianTokenPlanLLMServiceConnector` → `bailian_token_plan`, etc.).
Just paste the key into the matching slot in `mie_llm_keys.json` once,
and every workflow will pick it up.

### "My ZhiPu key isn't `eyJ...` but the connector still says 401."

The user may have a Coding Plan key. The Coding Plan key format is
opaque (not `eyJ...`); it's only usable against `/api/coding/paas/v4/...`.
Route them to `SetZhiPuCodeLLMServiceConnector`.

---

## 7. Verification workflow

Once the user has wired up a connector:

1. Recommend they attach a `CheckLLMServiceConnectivity` node to the
   connector's output. It sends `你是什么模型？` and surfaces HTTP status
   + first 200 chars of response. This is the fastest way to detect
   wrong-tier / wrong-model / wrong-key without running a full workflow.
2. If the connectivity check passes, attach it to a downstream node
   (`CallLLMService`, `Translator`, `PromptGenerator`, etc.) and queue the
   workflow.
3. If the connectivity check fails with `401` or `403`, re-check the
   key-vs-connector pairing per §2. If it fails with `400` and the model
   looks like a non-coder model on Coding Plan, switch connector or
   switch model.

---

## 8. Where to look if this file disagrees with the code

The source of truth is `services/llm.py` (the `api_url` class attributes
and the `INPUT_TYPES` dropdowns). If this guide and the code ever drift,
update the guide — the code is what ComfyUI loads at runtime, but the
guide is what an LLM reads to advise a user. Both must agree.

Tests that pin the most common regressions live in
`tests/test_llm_bailian_plans.py` (this PR) and `tests/test_llm_mimo.py`
(MiniMax / MiMo precedent).
