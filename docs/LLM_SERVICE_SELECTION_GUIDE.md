# LLM Service Platform Selection Guide

> This file is an LLM-readable instruction document. It is intended to be
> loaded into the system prompt (or read on demand) by any assistant that
> helps users choose a `Set*LLMServiceConnector` node and fill in the
> matching API key. Keep it authoritative: when a connector is added,
> removed, or its endpoint changes, this file MUST be updated in lockstep
> with `services/llm.py`.
>
> Last audited against live APIs + official docs on 2026-09-29.

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
| `sk-sp-` (Bailian, bought Token Plan / 套餐) | Bailian Token Plan | `SetBailianTokenPlanLLMServiceConnector` | `bailian_token_plan` |
| `sk-sp-` (Bailian, bought Coding Plan / 编程订阅) | Bailian Coding Plan | `SetBailianCodingPlanLLMServiceConnector` | `bailian_coding` |
| hex `{id}.{secret}` (and provider = 智谱, no subscription) | 智谱 BigModel Open Platform | `SetZhiPuLLMServiceConnector` | `zhipu` |
| hex `{id}.{secret}` (and provider = 智谱, bought GLM Coding / Token Plan) | 智谱 GLM Coding / Token Plan | `SetZhiPuCodeLLMServiceConnector` | `zhipu_code` |
| `sk-` (and provider = MiniMax) | MiniMax Open Platform | `SetMiniMaxLLMServiceConnector` | `minimax_open_platform` |
| `sk-cp-` (and provider = MiniMax) | MiniMax Token Plan / Coding Plan | `SetMiniMaxTokenPlanLLMServiceConnector` | `minimax` |
| `sk-` (and provider = 小米 MiMo) | MiMo Open Platform | `SetMiMoLLMServiceConnector` | `mimo` |
| `tp-` | MiMo Token Plan / Coding Plan | `SetMiMoTokenPlanLLMServiceConnector` | `mimo_token_plan` |
| `sk-` (and provider = SiliconFlow) | 硅基流动 | `SetSiliconFlowLLMServiceConnector` | `siliconflow` |
| `sk-sp-` (and provider = SiliconFlow Coding Plan) | 硅基流动 Coding Plan — no public fixed endpoint; use `SetGeneralLLMServiceConnector` with the base URL from the console | (escape hatch) | `openai_compatible` |
| `sk-` (and provider = DeepSeek) | DeepSeek | `SetDeepSeekLLMServiceConnector` | `deepseek` |
| `sk-` (and provider = Kimi) | 月之暗面 Kimi | `SetKimiLLMServiceConnector` | `kimi` |
| starts with `AIza` | Google Gemini | `SetGeminiLLMServiceConnector` | `gemini` |
| `sk-` (and provider = OpenAI) | OpenAI | `SetOpenAILLMServiceConnector` | `openai` |
| `xai-` | xAI Grok | `SetGrokLLMServiceConnector` | `grok` |
| `sk-or-` | OpenRouter (aggregator) | `SetOpenRouterLLMServiceConnector` | `openrouter` |
| `sk-ant-` | Anthropic Claude (official OpenAI-compat layer) | `SetClaudeLLMServiceConnector` | `anthropic` |
| Ark API key / IAM key (and provider = 火山引擎) | 豆包 Doubao / Volcano Ark | `SetDoubaoLLMServiceConnector` | `doubao` |
| Baidu key (and provider = 百度千帆) | 千帆 ERNIE | `SetQianfanLLMServiceConnector` | `qianfan` |
| APIPassword (and provider = 讯飞) | iFLYTEK Spark X2 | `SetSparkLLMServiceConnector` | `spark` |
| `sk-` (and provider = SiliconFlow, for Jev decision models) | SiliconFlow Jev replicas (Kev-4B / SemIf / diffusiongemma) | `SetSiliconFlowJevLLMServiceConnector` | `siliconflow` |
| empty / not applicable | Ollama (local) | `SetOllamaLLMServiceConnector` | `ollama` |
| anything else (custom base URL) | Custom OpenAI-compatible | `SetGeneralLLMServiceConnector` | `openai_compatible` |

> **Note on `sk-` collisions.** The bare `sk-` prefix is shared by many
> vendors (Bailian PAYG, MiniMax Open, MiMo Open, DeepSeek, Kimi,
> SiliconFlow). You cannot distinguish them from the key alone — you must
> ask the user which provider the key was issued from. `sk-sp-` is ALSO
> shared: Bailian Token Plan, Bailian Coding Plan, and SiliconFlow Coding
> Plan keys all use it — ask which subscription they bought.
>
> **ZhiPu keys carry no tier information.** Both tiers issue the same
> 49-char hex `{id}.{secret}` keys, and (verified live) both key kinds are
> accepted by both endpoints — the ENDPOINT is what selects the billing
> tier. Route by which subscription the user bought, not by key format.
>
> **GitHub Models is gone.** GitHub retired the service fully on
> 2026-07-30 (playground, catalog, inference API, BYOK). The old
> `SetGithubModelsLLMServiceConnector` was removed; `ghp_...` PATs no
> longer provide model inference. Migrate to Azure AI Foundry or another
> provider.

---

## 3. Pick by provider / use case

If the user does not have a key yet, route them by what they want to do:

| User says... | Recommend | Why |
|---|---|---|
| "I have an 阿里云百炼 / Bailian account and want to use Qwen" | `SetBailianLLMServiceConnector` (PAYG) if they have a metered key; otherwise ask whether they subscribed to Token Plan or Coding Plan | Bailian has three independent tiers — PAYG vs Token Plan vs Coding Plan. The keys are NOT interchangeable. |
| "I bought a 阿里云百炼 Token Plan / 套餐 / 千问套餐" | `SetBailianTokenPlanLLMServiceConnector` | Credits-based subscription (个人版, cn-beijing endpoint); Qwen3.8 flagships + DeepSeek / GLM / multimodal lineup. |
| "I bought a 阿里云百炼 Coding Plan / 编程套餐" | `SetBailianCodingPlanLLMServiceConnector` | Pro-tier subscription on the coding host; general chat models (qwen3.7-plus, glm-5, MiniMax-M2.5, kimi-k2.5) plus the qwen3-coder family. |
| "I bought a 硅基流动 SiliconFlow Coding Plan / 包月套餐" | `SetGeneralLLMServiceConnector` + the console-provided base URL + `sk-sp-` key | SiliconFlow does not publish a fixed Coding Plan endpoint; each subscription gets its base URL from the console. |
| "I want to run a local model" / "no API key" / "Ollama" | `SetOllamaLLMServiceConnector` | Local; no key needed (key field can stay empty). |
| "I have a vendor not listed" / "I have a weird base URL" | `SetGeneralLLMServiceConnector` | OpenAI-compatible escape hatch — user fills base URL + model themselves. |
| Any other named provider | Use the table in §2 by inferring the key prefix once they have one. | |

---

## 4. Endpoint reference (for diagnosis, not selection)

| Connector | Base URL | Notes |
|---|---|---|
| `SetBailianLLMServiceConnector` | `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions` | PAYG / 按量计费. OpenAI-compatible. Slim payload (no `response_format`). |
| `SetBailianTokenPlanLLMServiceConnector` | `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions` | Token Plan / 套餐 (个人版). MaaS workspace endpoint, NOT on the public `dashscope.aliyuncs.com` host. Key rotates on re-purchase. |
| `SetBailianCodingPlanLLMServiceConnector` | `https://coding.dashscope.aliyuncs.com/v1/chat/completions` | Coding Plan / 编程订阅. Different host again from Token Plan. Key prefix `sk-sp-` (same shape as Token Plan keys, different scope). |
| `SetZhiPuLLMServiceConnector` | `https://open.bigmodel.cn/api/paas/v4/chat/completions` | Standard BigModel. |
| `SetZhiPuCodeLLMServiceConnector` | `https://open.bigmodel.cn/api/coding/paas/v4/chat/completions` | Coding / Token Plan tier; same host, different path. Both endpoints expose the same model list. |
| `SetMiniMaxLLMServiceConnector` | `https://api.minimaxi.com/v1/chat/completions` | Open Platform. |
| `SetMiniMaxTokenPlanLLMServiceConnector` | `https://api.minimaxi.com/v1/chat/completions` | Same URL as Open Platform (verified live 2026-09-29); the **key prefix** (`sk-cp-`) is what tells the server to bill Token Plan. |
| `SetMiMoLLMServiceConnector` | `https://api.xiaomimimo.com/v1/chat/completions` | Standard. |
| `SetMiMoTokenPlanLLMServiceConnector` | `https://token-plan-cn.xiaomimimo.com/v1/chat/completions` | Different host. Verified against mimo.mi.com docs 2026-09-29. |
| `SetSiliconFlowLLMServiceConnector` | `https://api.siliconflow.cn/v1/chat/completions` | Many free-tier models. |
| `SetDeepSeekLLMServiceConnector` | `https://api.deepseek.com/chat/completions` | |
| `SetKimiLLMServiceConnector` | `https://api.moonshot.cn/v1/chat/completions` | PAYG-only — Kimi's API platform has no subscription tier. |
| `SetGeminiLLMServiceConnector` | `https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key=...` | Image inputs forwarded as `inline_data`. |
| `SetOllamaLLMServiceConnector` | `{host}/v1/chat/completions` | Default `host = http://127.0.0.1:11434`. No key required. |
| `SetDoubaoLLMServiceConnector` | `https://ark.cn-beijing.volces.com/api/v3/chat/completions` | Volcano Ark; model = doubao-seed-* id or `ep-` endpoint id. |
| `SetQianfanLLMServiceConnector` | `https://qianfan.baidubce.com/v2/chat/completions` | Qianfan ModelBuilder v2; ERNIE + hosted glm/deepseek. |
| `SetSparkLLMServiceConnector` | `https://spark-api-open.xf-yun.com/x2/chat/completions` | Spark X2 (model `spark-x`); legacy lite/generalv3 under `/v1/` via the general connector. |
| `SetOpenAILLMServiceConnector` | `https://api.openai.com/v1/chat/completions` | Sends `max_completion_tokens` (gpt-5+/o-series reject `max_tokens`). |
| `SetGrokLLMServiceConnector` | `https://api.x.ai/v1/chat/completions` | grok-4.x families. |
| `SetOpenRouterLLMServiceConnector` | `https://openrouter.ai/api/v1/chat/completions` | 400+ vendor models via `vendor/model` slugs. |
| `SetClaudeLLMServiceConnector` | `https://api.anthropic.com/v1/chat/completions` | Anthropic's official OpenAI SDK compat layer; slim payload. |
| `SetSiliconFlowJevLLMServiceConnector` | `https://api.siliconflow.cn/v1/systemone` | SiliconFlow-hosted Jev decision models (NOT chat): drive with the `CallJevDecision` node (`state` + `questions` JSON → noul/choice/score answers). Reuses the `siliconflow` key; free during the 2026-10-08 promo. (TypeSafe's own hosted endpoint is out of scope — same protocol if ever needed.) |
| `SetGeneralLLMServiceConnector` | user-supplied | For any OpenAI-compatible service not listed (incl. SiliconFlow Coding Plan with a console base URL, LM Studio `http://127.0.0.1:1234/v1/chat/completions`, Alibaba Model Studio intl `dashscope-intl.aliyuncs.com`, ZhiPu intl `api.z.ai`). |

---

## 5. Recommended models per connector

These are the **stable family names** baked into the dropdowns. The user
can pick a specific date-suffixed version (`qwen3.8-max-2026-06-08`) by
switching to `Custom` and pasting the full model id from
https://bailian.console.aliyun.com/.

### Bailian PAYG (`SetBailianLLMServiceConnector`)

- Qwen flagships: `qwen3.8-max`, `qwen3.8-flash`, `qwen3.7-max`,
  `qwen3.7-plus`, `qwen3.7-flash`, `qwen3.6-flash`, `qwen3.6-plus`,
  `qwen3.5-flash`, `qwen3.5-plus` — plus the stable aliases `qwen-plus`,
  `qwen-max`, `qwen-flash`, `qwen-turbo`, `qwen-long`
- Hosted third-party: `glm-5.3`, `glm-5.2`, `glm-5.1`, `glm-5`,
  `kimi-k3`, `kimi-k2.6`, `deepseek-v4-pro`, `deepseek-v4.1-flash`

### Bailian Token Plan (`SetBailianTokenPlanLLMServiceConnector`)

2026-09 docs lineup (text chat entries kept in the dropdown):
- Qwen: `auto`, `qwen3.8-max`, `qwen3.8-flash`, `qwen3.7-max`,
  `qwen3.7-plus`, `qwen3.6-flash`
- DeepSeek: `deepseek-v4.1-flash`, `deepseek-v4-pro`
- ZhiPu: `glm-5.3`, `glm-5.2`
- (Image / audio / video families — wan2.7, happyhorse, qwen-image-3.0 —
  exist on this tier but are not usable through the chat-completions node.)

### Bailian Coding Plan (`SetBailianCodingPlanLLMServiceConnector`)

- `qwen3.7-plus`, `qwen3.6-plus`, `qwen3.5-plus`,
  `qwen3-max-2026-01-23`, `qwen3-coder-plus`, `qwen3-coder-next`,
  `glm-5`, `glm-4.7`, `MiniMax-M2.5`, `kimi-k2.5`
- (Not coder-only anymore — general chat models are included per the
  2026-09 docs.)

### Other connectors (verified live 2026-09-29)

- ZhiPu (both tiers, same list): `glm-5.3`, `glm-5.3-flash`,
  `glm-5.3-flashx`, `glm-5.2`, `glm-5.1`, `glm-5-turbo`, `glm-5`,
  `glm-4.7`, `glm-4.6`, `glm-4.5`, `glm-4.5-air`
- Kimi: `kimi-k3` (flagship), `kimi-k2.7-code`,
  `kimi-k2.7-code-highspeed`, `kimi-k2.6` — `kimi-k2.5` and the whole
  `moonshot-v1-*` family went offline 2026-08-31.
- DeepSeek: `deepseek-v4-pro`, `deepseek-flash` (renamed from
  `deepseek-v4-flash`).
- SiliconFlow: `deepseek-ai/DeepSeek-V4-Pro`, `deepseek-ai/DeepSeek-V4-Flash`,
  `zai-org/GLM-5.3`, `zai-org/GLM-5.2`, `Pro/zai-org/GLM-5.1`,
  `moonshotai/Kimi-K2.7-Code`, `Qwen/Qwen3.8-27B`, `Qwen/Qwen3-VL-32B-*`
  and more (dropdown mirrors `GET /v1/models`).
- MiniMax (both tiers, same list): `MiniMax-M3`, `MiniMax-M2.7(-highspeed)`,
  `MiniMax-M2.5(-highspeed)`, `MiniMax-M2.1(-highspeed)`, `MiniMax-M2`.
- MiMo (both tiers): `mimo-v2.6-pro`, `mimo-v2.6-pro-ultraspeed`,
  `mimo-v2.6-flash`, `mimo-v2.5-pro`, `mimo-v2.5` (v2.5 marked
  "Offline Soon" on the product page).
- Gemini: `gemini-3.8-flash` (flagship), `gemini-3.7-flash`,
  `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`,
  `gemini-3.1-pro-preview`, `gemini-3-flash-preview`. There is no stable
  3.x Pro; the 2.5 family is restricted to pre-existing users.

### Platforms added 2026-09 (verified against official docs; no local keys)

- OpenAI: `gpt-6-astra(-pro)`, `gpt-6-luna(-pro)`, `gpt-5.5(-pro)`,
  `gpt-5.2-chat`, `gpt-5.1` — payload uses `max_completion_tokens`.
- xAI Grok: `grok-4.7` (frontier), `grok-4.6`, `grok-4.5`, `grok-4.3`,
  `grok-4.20(-multi-agent)`.
- Anthropic Claude (OpenAI-compat layer): `claude-sonnet-5-5`,
  `claude-opus-5-5`, `claude-fable-5-1`, `claude-haiku-4-5`, plus the
  5.0 generation. Dashed ids (`claude-opus-5-5`), not dotted.
- OpenRouter: `openrouter/auto` plus verified cross-vendor slugs
  (`openai/gpt-6-astra`, `anthropic/claude-opus-5.5`,
  `google/gemini-3.8-flash`, `x-ai/grok-4.7`, `deepseek/deepseek-v4-pro`,
  `z-ai/glm-5.3`, `moonshotai/kimi-k3`, `qwen/qwen3.5-397b-a17b`,
  `minimax/minimax-m3`, `mistralai/mistral-large-2512`); any other slug
  via `custom_model`.
- Doubao (Volcano Ark): `doubao-seed-2-0-pro-260215`,
  `doubao-seed-2-0-lite-260215`, `doubao-seed-2-0-mini-260215`,
  `doubao-seed-1-8-251228`, `doubao-seed-1-6`; Ark `ep-` endpoint ids
  via `custom_model`.
- Qianfan ERNIE: `ernie-5.1` (latest), `ernie-5.0`,
  `ernie-5.0-thinking-preview`, `ernie-4.5-turbo-128k/32k/vl`,
  plus hosted `glm-5.3` / `deepseek-v4.1-flash`.
- iFLYTEK Spark: `spark-x` on the `/x2/` endpoint (generation is
  selected by URL path, not model name).
- SiliconFlow Jev decision models: `Kev-4B` (default), `SemIf`,
  `diffusiongemma` — no "model recommendation" in the usual sense,
  they answer `noul` / `choice` / `score` questions rather than
  prompts, so pick by benchmark fit and use `CallJevDecision`.

### Deliberately NOT added (2026-09)

- 腾讯混元 Hunyuan: the OpenAI-compat endpoint exists
  (`api.hunyuan.cloud.tencent.com/v1`) but Tencent is migrating LLM
  serving to TokenHub; the t1/turbos/HY2.0 flagships were retired
  2026-06-10 and the surviving chat lineup is niche. Use
  `SetGeneralLLMServiceConnector` with the URL above if you need it.
- Perplexity (search-grounded, wrong fit; reachable via OpenRouter),
  StepFun (activity unconfirmed), Alibaba Model Studio intl / ZhiPu
  z.ai intl (same protocols as their CN connectors — use the general
  connector with the intl base URL).

---

## 6. Common pitfalls

### "I plugged my key into the wrong node and got a 401."

Most often this is **Bailian tier confusion**: PAYG `sk-` keys are scoped
to `dashscope.aliyuncs.com`, Token Plan `sk-sp-` keys are scoped to
`token-plan.cn-beijing.maas.aliyuncs.com`, and Coding Plan `sk-sp-` keys
are scoped to `coding.dashscope.aliyuncs.com`. Bailian's docs state the
namespaces are completely isolated — there is no automatic redirection.
Ask the user which subscription they bought; the key prefix plus the
subscription they named is the most reliable hint.

### "I picked the right connector but the wrong model."

- Bailian Token Plan dropped the legacy `qwen3-*` names (`qwen3-max`,
  `qwen3-vl-plus`, ...). Those models are gone from that tier — pick a
  `qwen3.8-*` / `qwen3.7-*` id instead.
- Bailian PAYG `qwen3.8-max` is a stable alias; the real snapshot ids
  carry date suffixes (`qwen3.8-max-0902`). If the user reports
  `model not found`, ask them to check the console and paste the full id
  into `custom_model`.
- ZhiPu tier: both endpoints serve the same model list, so a "wrong
  model" error on one tier usually means a typo, not a tier mismatch.

### "Kimi returns 400 `invalid temperature: only 1 is allowed`."

The current Kimi lineup locks sampling params server-side
(temperature=1.0, top_p=0.95, n=1, penalties=0). The connector already
omits them from the payload; do not add them back. If a downstream node
passes `temperature`, it is ignored by the Kimi payload builder by design.

### "My SiliconFlow Coding Plan key doesn't bill the subscription."

SiliconFlow Coding Plan (`sk-sp-` keys) requires the dedicated base URL
shown in the console after subscribing; it is not `api.siliconflow.cn`.
Use `SetGeneralLLMServiceConnector` with that URL.

### "I want to use a free model."

Several providers (SiliconFlow, 智谱 GLM Flash tier, Bailian
`qwen3.8-flash` at low-volume promo) offer free or near-free tiers. The
free lineup changes often — recommend the user check the provider's
pricing page. Enter the free model id into `custom_model`.

### "I don't want to paste my key into the node every time."

The connector nodes read from `mie_llm_keys.json` when their
`config_key` is set and the key field is left blank. The default
`config_key` for each connector matches its name suffix
(`SetBailianTokenPlanLLMServiceConnector` → `bailian_token_plan`, etc.).
Just paste the key into the matching slot in `mie_llm_keys.json` once,
and every workflow will pick it up.

### "My workflow has a red `SetGithubModelsLLMServiceConnector` node."

GitHub Models was fully retired on 2026-07-30 and the node was removed
from this plugin. Replace it with any other connector; `ghp_...` PATs no
longer grant model inference.

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
   key-vs-connector pairing per §2. If it fails with `400` and the body
   complains about a sampling parameter, the vendor locks that parameter —
   the connector should already omit it; report it as a bug if it does not.

---

## 8. Where to look if this file disagrees with the code

The source of truth is `services/llm.py` (the `api_url` class attributes
and the `INPUT_TYPES` dropdowns). If this guide and the code ever drift,
update the guide — the code is what ComfyUI loads at runtime, but the
guide is what an LLM reads to advise a user. Both must agree.

Tests that pin the most common regressions live in
`tests/test_llm_bailian_plans.py` (Bailian tiers),
`tests/test_llm_mimo.py` (MiniMax / MiMo precedent),
`tests/test_llm_platform_audit.py` (2026-09 dropdown / payload audit)
and `tests/test_llm_platform_expansion.py` (2026-09 added platforms).
