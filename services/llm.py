import re
import json
import requests
import time
import base64
import numpy as np
import cv2
import torch

try:
    from _mienodes_internal.core.utils import (
        mie_log,
        load_plugin_config,
        resolve_token,
        image_tensor_to_data_url,
        build_multimodal_user_content,
    )
except ImportError:
    from ..core.utils import (
        mie_log,
        load_plugin_config,
        resolve_token,
        image_tensor_to_data_url,
        build_multimodal_user_content,
    )

MY_CATEGORY = "🐑 MieNodes/🐑 LLM Service Config"


# 引入 time 模块用于在重试间增加延迟

def _drop_image_detail_auto(messages):
    """Drop `detail: "auto"` from any image_url content part.

    The OpenAI image_url spec allows `auto` / `low` / `high`, but
    MiniMax M-series vision models reject "auto" with HTTP 400
    (`invalid params, invalid image detail: auto`). OpenAI treats
    a missing `detail` field as "auto" internally, so removing the
    key is safe for OpenAI-compat services too. Gemini uses a
    different content shape (inline_data) and never sees this.

    Returns the input unchanged when no `detail: "auto"` is present
    (cheap fast path). When a change is needed, returns a new list
    with selectively-copied dicts - never mutates the caller's
    message structure, so the same list can be reused across calls
    (e.g. retry loops, or sending the same prompt to multiple
    providers in sequence).
    """
    if not messages:
        return messages
    new_messages = None
    for mi, msg in enumerate(messages):
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        new_content = None
        for pi, part in enumerate(content):
            if not (
                isinstance(part, dict)
                and part.get("type") == "image_url"
                and isinstance(part.get("image_url"), dict)
                and part["image_url"].get("detail") == "auto"
            ):
                continue
            if new_content is None:
                new_content = list(content)
            new_part = dict(part)
            new_part["image_url"] = dict(part["image_url"])
            del new_part["image_url"]["detail"]
            new_content[pi] = new_part
        if new_content is not None:
            if new_messages is None:
                new_messages = list(messages)
            new_messages[mi] = dict(msg)
            new_messages[mi]["content"] = new_content
    return new_messages if new_messages is not None else messages


class GeneralLLMServiceConnector:
    def __init__(self, api_url, manual_token, model, timeout=30, max_retries=3, retry_delay=5, 
                 config_file="mie_llm_keys.json", config_key=None, prefer_local_config=True):
        self.api_url = api_url
        self.manual_token = manual_token
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.config_file = config_file
        self.config_key = config_key
        self.prefer_local_config = prefer_local_config

    @property
    def api_token(self):
        return resolve_token(
            self.manual_token, 
            default_key=self.config_key, 
            config_file=self.config_file, 
            config_key=self.config_key, 
            prefer_local=self.prefer_local_config
        )

    def _provider_messages(self, messages):
        """Hook for converting OpenAI-style multimodal messages into the
        provider-native shape. Default is identity: OpenAI-compatible services
        already understand `image_url` content parts, so the default no-op is
        correct for them. `GeminiConnectorGeneral` overrides this to map
        `image_url` -> `inline_data`.

        Returning a fresh list is recommended so callers can safely mutate.
        """
        out = list(messages) if messages is not None else messages
        return self._sanitize_image_detail(out)

    def _sanitize_image_detail(self, messages):
        """Hook for provider-specific image_url `detail` value sanitization.

        Default is identity (preserves whatever the caller set). Some
        providers reject the OpenAI-default `detail: "auto"` value;
        override this in those connectors. Called from
        `_provider_messages` so subclasses that fully override
        `_provider_messages` (e.g. Gemini) opt out automatically.
        """
        return messages

    # Strips reasoning / chain-of-thought blocks that some models (DeepSeek R1,
    # GLM-Z, MiniMax M-series, etc.) emit before the final answer. Matches
    # both `<think>...</think>` and `<thinking>...</thinking>`.
    _THINK_BLOCK_RE = re.compile(
        r"<think>[\s\S]*?</think>"
        r"|<thinking>[\s\S]*?</thinking>",
        re.IGNORECASE,
    )
    # Some providers strip the opening tag server-side but leave the closing
    # one (seen live: SiliconFlow DeepSeek-V4-Flash returning `OK</think>OK`),
    # so orphan closers are removed as a second pass after paired blocks.
    _ORPHAN_THINK_CLOSE_RE = re.compile(r"</think(?:ing)?>", re.IGNORECASE)

    def _sanitize_response(self, text, preserve_thinking=False):
        """Strip `<think>` / `<thinking>` reasoning blocks from `text`.

        Several reasoning models emit their chain-of-thought inside the content
        field before the actual answer. For prompt-rewriter use cases the
        thinking is noise that pollutes downstream models' input, so we strip
        it by default. Pass `preserve_thinking=True` to keep it (useful for
        debug / `CheckLLMServiceConnectivity` style diagnostics).
        """
        if text is None or preserve_thinking:
            return text
        cleaned = self._THINK_BLOCK_RE.sub("", text)
        cleaned = self._ORPHAN_THINK_CLOSE_RE.sub("", cleaned)
        return cleaned.strip()

    @staticmethod
    def _payload_chars(payload):
        """Return ``(total_chars, last_user_chars)`` for an LLM request payload.

        Used by the connector's POST / retry log lines so a slow or failing
        call carries the payload size alongside the elapsed seconds — without
        this a 300s ReadTimeout leaves no clue whether the request was 8KB or
        80KB. ``messages[*].content`` is summed; non-message fields (model,
        response_format, stream) are not counted (constant per connector).
        """
        total = 0
        last_user_chars = 0
        try:
            for m in payload.get("messages") or []:
                content = m.get("content") or ""
                if isinstance(content, list):
                    # Multi-modal payloads: count only text parts.
                    content = "".join(
                        str(p.get("text", ""))
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
                n = len(str(content))
                total += n
                if m.get("role") == "user":
                    last_user_chars = n
        except Exception:
            pass
        return total, last_user_chars

    def generate_payload(self, messages, **kwargs):
        """
        生成标准的 OpenAI 兼容服务的 Payload。
        子类如果需要不同的默认参数或结构，可以重写此方法。
        """
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "response_format": {"type": "text"},
        }

    def invoke(self, messages, **kwargs):
        """
        调用 LLM 服务，并实现针对瞬时错误的重试机制。
        重试包括：Timeout, ConnectionError, 和 5xx 状态码。

        日志约定（按出现顺序）:
          - 每次 attempt 开头: ``[<model>] attempt N/M: POST <url> timeout=Ts``
          - 5xx 失败: 同前缀 + HTTP 状态码 + 耗时 + 响应体前 200 字符
          - Timeout/ConnectionError 失败: 同前缀 + 异常类型 + 耗时
          - 成功: ``[<model>] attempt N/M ok in Xs response_chars=N``
        外部 caller（如 Bernini 的 ``_chat``）会再记一次合并耗时，但不会覆盖
        单次 attempt 的真实耗时——两边互补。
        """
        # `preserve_thinking` is response-side, not payload-side; pop it
        # before generate_payload so it cannot leak into the request body.
        preserve_thinking = bool(kwargs.pop("preserve_thinking", False))
        payload = self.generate_payload(messages, **kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }

        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)
            attempt_idx = attempt + 1
            tag = f"[{self.model}] attempt {attempt_idx}/{self.max_retries}"
            req_chars, last_user_chars = self._payload_chars(payload)
            mie_log(
                f"{tag}: POST {self.api_url} timeout={self.timeout}s "
                f"request_chars={req_chars} last_user_chars={last_user_chars}"
            )
            attempt_t0 = time.perf_counter()

            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                attempt_elapsed = time.perf_counter() - attempt_t0

                if response.status_code == 200:
                    response_data = response.json()
                    try:
                        message = response_data["choices"][0]["message"]
                    except (KeyError, IndexError) as e:
                        raise ValueError(
                            f"Unexpected response format: {type(e).__name__}. "
                            f"Response: {response.text[:200]}...")
                    content = message.get("content") or ""
                    cleaned = self._sanitize_response(
                        content, preserve_thinking=preserve_thinking
                    )
                    # Reasoning models (MiniMax-M3, DeepSeek-R1 API, GLM-5.x)
                    # emit their chain-of-thought in a separate
                    # ``reasoning_content`` field while the real answer sits in
                    # ``content``. Some providers (e.g. MiniMax-M3 inline mode)
                    # instead put the whole `<think>...</think>` chain inside
                    # ``content`` so that ``_sanitize_response`` strips it; when
                    # that leaves ``content`` empty (chain consumed the whole
                    # token budget before the answer), fall back to
                    # ``reasoning_content`` so callers still get the model's
                    # final reasoning instead of a bare empty string.
                    if not cleaned:
                        reasoning = message.get("reasoning_content") or ""
                        if reasoning:
                            mie_log(
                                f"{tag} content empty after sanitize; "
                                f"falling back to reasoning_content "
                                f"({len(reasoning)} chars)"
                            )
                            cleaned = self._sanitize_response(
                                reasoning, preserve_thinking=preserve_thinking
                            )
                        else:
                            # HTTP 200 but zero usable text. finish_reason
                            # distinguishes the two causes we have seen live:
                            # "length" = the reasoning chain consumed the
                            # whole max_tokens budget before the answer
                            # started (raise the caller's budget); anything
                            # else = provider-side empty reply.
                            try:
                                finish = response_data["choices"][0].get(
                                    "finish_reason"
                                )
                            except (KeyError, IndexError):
                                finish = "<unknown>"
                            mie_log(
                                f"{tag} empty reply; finish_reason={finish!r} "
                                f"(length => raise the caller's max_tokens "
                                f"budget so reasoning cannot starve the "
                                f"answer)"
                            )
                    mie_log(
                        f"{tag} ok in {attempt_elapsed:.2f}s "
                        f"response_chars={len(cleaned or '')}"
                    )
                    return cleaned

                # 5xx: 瞬时错误，包含响应体前 200 字符方便诊断
                if 500 <= response.status_code < 600:
                    body_snip = (response.text or "").replace("\n", " ")[:200]
                    detail = (
                        f"{tag} got HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                        f"body={body_snip!r}"
                    )
                    if is_last_attempt:
                        raise Exception(
                            f"{detail}. Max retries ({self.max_retries}) exceeded."
                        )
                    mie_log(
                        f"{detail}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    continue

                # 4xx 等非重试错误：仍把响应体前 200 字符带出来
                body_snip = (response.text or "").replace("\n", " ")[:200]
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                    f"body={body_snip!r}"
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt_elapsed = time.perf_counter() - attempt_t0
                error_type = type(e).__name__
                detail = (
                    f"{tag} {error_type} after {attempt_elapsed:.2f}s "
                    f"(request_chars={req_chars}, last_user_chars={last_user_chars})"
                )
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(
                    f"{detail}. Retrying in {self.retry_delay} seconds... "
                    f"(Attempt {attempt_idx}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                raise Exception(f"A non-retryable request error occurred: {e}")

        # 理论上不会执行到这里，但以防万一
        raise Exception(
            f"LLM Service failed after {self.max_retries} attempts due to an unknown error."
        )

    def get_state(self):
        """返回用于比较状态的字符串表示"""
        # 恢复为原先的无分隔符形式，保证与历史行为一致（避免回归）
        return f"{self.api_url}{self.api_token}{self.model}"


class StandardOpenAICompatibleConnector(GeneralLLMServiceConnector):
    """
    针对 SiliconFlow, ZhiPu, Kimi 等具有标准 OpenAI 兼容参数的 API。
    """

    def generate_payload(self, messages, **kwargs):
        # 封装 OpenAI 兼容服务的通用参数
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "top_k": kwargs.get("top_k", 50),
            "frequency_penalty": kwargs.get("frequency_penalty", 0.5),
            "n": kwargs.get("n", 1),
            "response_format": {"type": "text"},
        }


# 适配SiliconFlow
class SiliconFlowConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://api.siliconflow.cn/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class ZhiPuConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard ZhiPu BigModel connector (NOT the Coding / Token Plan tier).

    Targets the public ZhiPu BigModel API at the standard
    `/api/paas/v4/chat/completions` endpoint. ZhiPu API keys are 49-char
    hex `{id}.{secret}` strings (NOT `eyJ...` JWTs), and the same key
    format is accepted by both the standard and the `/api/coding/...`
    endpoints — the endpoint (i.e. which subscription you bought) is
    what selects the billing tier, not the key format. For the Coding /
    Token Plan subscription endpoint use `ZhiPuCodeConnectorGeneral`
    and `SetZhiPuCodeLLMServiceConnector`.
    """
    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class ZhiPuCodeConnectorGeneral(StandardOpenAICompatibleConnector):
    """ZhiPu Coding / Token Plan connector.

    Targets the ZhiPu Coding Plan endpoint at `/api/coding/...`
    with a Coding / Token Plan subscription. Distinct from
    the standard ZhiPu BigModel API in URL and billing (the GLM-5.3
    series is served here too — both endpoints expose the same model
    list; verified live 2026-09-29). Keys share the 49-char hex format
    with the standard tier. Pair with `SetZhiPuCodeLLMServiceConnector`.
    """
    api_url = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class KimiConnectorGeneral(StandardOpenAICompatibleConnector):
    """Moonshot Kimi connector (api.moonshot.cn, pay-as-you-go only).

    The official Kimi API platform has NO subscription tier (docs:
    "Kimi API 开放平台是按量计费模式、无订阅制方案"); the Kimi for Coding
    subscription is a separate OAuth product for CLI tools, not an API-key
    chat endpoint, so no token-plan sibling connector exists.

    Payload note: per the official model parameter reference the current
    lineup locks sampling params server-side (kimi-k3 / kimi-k2.7-code /
    kimi-k2.6-thinking: temperature=1.0; every model: top_p=0.95, n=1,
    presence/frequency_penalty=0) and recommends NOT sending them
    explicitly — sending other values fails live with HTTP 400
    (`invalid temperature: only 1 is allowed for this model`). This
    payload therefore forwards only `max_tokens` and lets the server
    apply its own sampling defaults.
    """

    api_url = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
            "response_format": {"type": "text"},
        }


class BailianLLMServiceConnector(GeneralLLMServiceConnector):
    api_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        # 阿里百炼（通义千问）的 Payload 可能有所不同，这里保留其特殊性
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            # 可以根据需要添加其他参数
        }


class DeepSeekConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://api.deepseek.com/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class MiniMaxConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard MiniMax Open Platform connector.

    Targets the public MiniMax Open Platform API. Use with the
    standard `eyJ...` (JWT) API key issued from the MiniMax
    developer console. For the newer Token Plan / Coding Plan
    (`sk-cp-...` prefixed) keys, use `MiniMaxTokenPlanConnectorGeneral`
    and `SetMiniMaxTokenPlanLLMServiceConnector` instead.
    """
    api_url = "https://api.minimaxi.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def _sanitize_image_detail(self, messages):
        """Drop `detail: "auto"` from image_url parts. MiniMax rejects
        the OpenAI default with HTTP 400 (`invalid image detail: auto`);
        only `low` and `high` are accepted. OpenAI treats a missing
        field as "auto" internally, so stripping is a no-op there.
        """
        return _drop_image_detail_auto(messages)


class MiniMaxTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """MiniMax Token Plan / Coding Plan connector.

    Targets the MiniMax Token Plan endpoint with `sk-cp-...` prefixed
    API keys (the Token Plan / Coding Plan subscription). The endpoint
    URL is shared with the Open Platform for now; the key prefix is
    what distinguishes the two billing tracks. M3 ships first on the
    Token Plan tier.
    """
    api_url = "https://api.minimaxi.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def _sanitize_image_detail(self, messages):
        """Same as MiniMaxConnectorGeneral: drop `detail: "auto"`.
        See `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)


class MiMoConnectorGeneral(StandardOpenAICompatibleConnector):
    """Standard Xiaomi MiMo Open Platform connector.

    Targets the public Xiaomi MiMo API at
    `https://api.xiaomimimo.com/v1/chat/completions` with `sk-xxxxx`
    API keys. For the Token Plan / Coding Plan (`tp-xxxxx` keys), use
    `MiMoTokenPlanConnectorGeneral` and `SetMiMoTokenPlanLLMServiceConnector`
    instead.

    Key differences from the generic OpenAI-compat shape:
      - Uses `max_completion_tokens` (newer OpenAI standard) instead of
        `max_tokens`; the MiMo docs only show the `max_completion_tokens`
        spelling.
      - Drops `top_k`, `n`, and `response_format` from the payload; the
        MiMo docs never show these and they are likely to 400.
      - Uses MiMo-friendly defaults: `temperature=1.0`, `top_p=0.95`.
      - Drops `detail: "auto"` from `image_url` parts (the MiMo image
        understanding docs do not include a `detail` field; the OpenAI
        default `"auto"` is known to be rejected by some providers).
    """

    api_url = "https://api.xiaomimimo.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
        }

    def _sanitize_image_detail(self, messages):
        """Drop `detail: "auto"` from image_url parts. The MiMo image
        understanding docs never include a `detail` field; only `low` /
        `high` are forwarded when the caller sets them explicitly. See
        `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)


class MiMoTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Xiaomi MiMo Token Plan / Coding Plan connector.

    Targets the MiMo Token Plan endpoint at
    `https://token-plan-cn.xiaomimimo.com/v1/chat/completions` with
    `tp-xxxxx` API keys. Distinct from the standard tier in base URL,
    billing model, and API key format. The model lineup is shared with
    the standard tier. Pair with `SetMiMoTokenPlanLLMServiceConnector`.
    """

    api_url = "https://token-plan-cn.xiaomimimo.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 0.95),
        }

    def _sanitize_image_detail(self, messages):
        """Same as `MiMoConnectorGeneral`: drop `detail: "auto"`.
        See `_drop_image_detail_auto` for the rationale.
        """
        return _drop_image_detail_auto(messages)


class DoubaoConnectorGeneral(StandardOpenAICompatibleConnector):
    """ByteDance Volcano Ark (火山方舟) Doubao connector.

    Targets the Ark OpenAI-compatible endpoint at
    `https://ark.cn-beijing.volces.com/api/v3/chat/completions` with an
    Ark API Key (or IAM API Key) from the Volcano console. `model`
    accepts either the versioned doubao-seed ids (see the Set-node
    dropdown) or an inference Endpoint ID (`ep-xxxx`) created in the
    console. Pair with `SetDoubaoLLMServiceConnector`.
    """
    api_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class QianfanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Baidu Qianfan (百度千帆) ERNIE connector.

    Targets the Qianfan ModelBuilder v2 OpenAI-compatible endpoint at
    `https://qianfan.baidubce.com/v2/chat/completions` with a Bearer API
    key from the Qianfan console. Serves the ERNIE family plus hosted
    third-party models (glm / deepseek, see the Set-node dropdown).
    Pair with `SetQianfanLLMServiceConnector`.
    """
    api_url = "https://qianfan.baidubce.com/v2/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class SparkConnectorGeneral(GeneralLLMServiceConnector):
    """iFLYTEK Spark (讯飞星火) X2 connector.

    Targets the Spark-X2 OpenAI-compatible endpoint at
    `https://spark-api-open.xf-yun.com/x2/chat/completions`, authorized
    with the console-issued APIPassword as a Bearer token. Per the
    official Spark-X2 HTTP protocol doc the model value is `spark-x`
    and the generation is selected by the URL path (`/x2/` here;
    X1.5 lives under `/v2/`, the retired lite/generalv3 classics under
    `/v1/` — point `SetGeneralLLMServiceConnector` at those paths for
    legacy models). Slim payload: the doc only shows
    messages/stream/temperature-style params, so sampling extras are
    omitted. Pair with `SetSparkLLMServiceConnector`.
    """
    api_url = "https://spark-api-open.xf-yun.com/x2/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
        }


class OpenAIConnectorGeneral(GeneralLLMServiceConnector):
    """OpenAI connector.

    Targets `https://api.openai.com/v1/chat/completions`. Uses
    `max_completion_tokens` (the newer OpenAI spelling) instead of
    `max_tokens` — the gpt-5+/o-series models reject `max_tokens`
    server-side. Sampling params are omitted so the server applies its
    per-model defaults (reasoning-tier models pin temperature anyway).
    Pair with `SetOpenAILLMServiceConnector`.
    """
    api_url = "https://api.openai.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_completion_tokens": kwargs.get("max_tokens", 512),
        }


class GrokConnectorGeneral(StandardOpenAICompatibleConnector):
    """xAI Grok connector.

    Targets `https://api.x.ai/v1/chat/completions` — an
    OpenAI-compatible endpoint that accepts the standard parameter set
    (verified against the xAI docs; `/v1/completions` is legacy).
    Pair with `SetGrokLLMServiceConnector`.
    """
    api_url = "https://api.x.ai/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class OpenRouterConnectorGeneral(StandardOpenAICompatibleConnector):
    """OpenRouter aggregator connector.

    Targets `https://openrouter.ai/api/v1/chat/completions` — one API
    key routes to 400+ models across OpenAI / Anthropic / Google / xAI /
    DeepSeek / GLM / Qwen / Kimi / MiniMax / MiMo etc. Model ids are
    `<vendor>/<model>` slugs (live list at GET /api/v1/models; the
    Set-node dropdown carries a verified cross-vendor selection).
    Pair with `SetOpenRouterLLMServiceConnector`.
    """
    api_url = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class ClaudeConnectorGeneral(GeneralLLMServiceConnector):
    """Anthropic Claude connector (official OpenAI-compat layer).

    Targets `https://api.anthropic.com/v1/chat/completions` — Anthropic's
    official OpenAI SDK compatibility layer, authorized with a Bearer
    Claude API key. Per the compat docs many OpenAI fields
    (response_format, penalties, seed, logprobs) are silently ignored
    and `n` must be 1, so this payload stays slim: model / messages /
    stream / max_tokens. Model ids use Anthropic's dashed spelling
    (e.g. `claude-opus-5-5`). For full-feature access (prompt caching,
    structured outputs, thinking) use the native Anthropic API instead.
    Pair with `SetClaudeLLMServiceConnector`.
    """
    api_url = "https://api.anthropic.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def generate_payload(self, messages, **kwargs):
        return {
            "model": self.model,
            "messages": self._provider_messages(messages),
            "stream": False,
            "max_tokens": kwargs.get("max_tokens", 512),
        }


class SiliconFlowJevConnectorGeneral(GeneralLLMServiceConnector):
    """SiliconFlow Jev / System One decision-model connector.

    Jev models (TypeSafe AI's Jev + the open replicas SiliconFlow hosts:
    Kev-4B / SemIf / diffusiongemma) are NOT chat models: the standard
    `/v1/chat/completions` endpoint rejects them with 400. They are
    called on the bare `/v1/systemone` endpoint with a
    `state + questions` payload and answer with structured decisions
    instead of prose — three question types:
      - ``noul``   -> yes/no probability
      - ``choice`` -> picked option + full probability distribution
      - ``score``  -> position on an ordered scale + per-level probs
    Verified live against SiliconFlow's `/v1/systemone` on 2026-09-29
    (same API key as the standard SiliconFlow chat tier; open-replica
    models free until the 2026-10-08 promo ends).

    `ask()` is the real API. `invoke()` exists only so the connector can
    flow through `LLMServiceConnector` sockets — it requires a
    `questions` kwarg and otherwise raises a pointer to the
    `CallJevDecision` node.
    """

    api_url = "https://api.siliconflow.cn/v1/systemone"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)

    def ask(self, state, questions):
        """POST one `state + questions` decision request, return the raw
        response dict (``{model, answers, usage}``). Retry policy mirrors
        `invoke()`: 5xx / Timeout / ConnectionError retried, 4xx raised.
        """
        payload = {
            "model": self.model,
            "state": state,
            "questions": questions,
        }
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        for attempt in range(self.max_retries):
            is_last_attempt = attempt == self.max_retries - 1
            attempt_idx = attempt + 1
            tag = f"[{self.model}] ask {attempt_idx}/{self.max_retries}"
            mie_log(f"{tag}: POST {self.api_url} timeout={self.timeout}s")
            attempt_t0 = time.perf_counter()
            try:
                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=self.timeout
                )
                elapsed = time.perf_counter() - attempt_t0
                if response.status_code == 200:
                    data = response.json()
                    n_answers = len((data or {}).get("answers") or {})
                    mie_log(
                        f"{tag} ok in {elapsed:.2f}s questions_answered={n_answers}"
                    )
                    return data
                body_snip = (response.text or "").replace("\n", " ")[:200]
                if 500 <= response.status_code < 600 and not is_last_attempt:
                    mie_log(
                        f"{tag} got HTTP {response.status_code} in {elapsed:.2f}s "
                        f"body={body_snip!r}. Retrying in {self.retry_delay}s..."
                    )
                    time.sleep(self.retry_delay)
                    continue
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {elapsed:.2f}s "
                    f"body={body_snip!r}"
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                elapsed = time.perf_counter() - attempt_t0
                detail = f"{tag} {type(e).__name__} after {elapsed:.2f}s"
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(f"{detail}. Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
        raise Exception(
            f"Jev decision request failed after {self.max_retries} attempts."
        )

    def invoke(self, messages, **kwargs):
        questions = kwargs.pop("questions", None)
        if not questions:
            raise Exception(
                "Jev decision models answer structured questions, not chat "
                "prompts — CallLLMService cannot drive them. Use the "
                "CallJevDecision node (state + questions JSON) instead."
            )
        state = "\n".join(
            str(m.get("content", ""))
            for m in (messages or [])
            if isinstance(m, dict)
        )
        data = self.ask(state, questions)
        return json.dumps(data.get("answers", data), ensure_ascii=False)


class GeminiConnectorGeneral(GeneralLLMServiceConnector):
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_token, model, **kwargs):
        self.model = model
        # self.api_token = api_token  # Removed, using base class dynamic property
        api_url = f"{self.base_url}/{model}:generateContent"
        # 继承基类的 timeout, max_retries, retry_delay
        super().__init__(api_url, api_token, model, **kwargs)

    # OpenAI `data:<mime>;base64,<data>` -> mime / payload groups
    _DATA_URL_RE = re.compile(r"^data:([^;]+);base64,(.*)$", re.DOTALL)

    def _provider_messages(self, messages):
        """Convert OpenAI-style image_url data URLs into Gemini inline_data parts.

        Gemini expects a different content shape from OpenAI - it uses
        `parts` with `inline_data` for images and `text` for prose,
        not the `type: image_url` part shape. We rewrite the user (and
        model) messages in-place so the rest of `generate_payload` can
        iterate the rewritten structure uniformly.

        Non-data-URL `image_url` (e.g. `https://`) is left as-is and the
        Gemini API will resolve it; if the Gemini endpoint rejects it, the
        caller should pre-encode via `image_tensor_batch_to_data_urls`.
        """
        if not messages:
            return messages
        out = []
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                out.append(msg)
                continue
            new_parts = []
            for p in content:
                if not isinstance(p, dict):
                    new_parts.append(p)
                    continue
                ptype = p.get("type")
                if ptype == "text" and "text" in p:
                    new_parts.append({"text": p["text"]})
                elif ptype == "image_url":
                    url = (p.get("image_url") or {}).get("url", "")
                    m = self._DATA_URL_RE.match(url)
                    if m:
                        mime_type, b64 = m.group(1), m.group(2)
                        new_parts.append({"inline_data": {"mime_type": mime_type, "data": b64}})
                    elif url:
                        # Remote URL: Gemini can fetch it directly via file_data
                        new_parts.append({"file_data": {"mime_type": "image/jpeg", "file_uri": url}})
                else:
                    # Unknown part type - pass through unchanged so the API
                    # surfaces a clear error rather than us silently losing it.
                    new_parts.append(p)
            new_msg = dict(msg)
            new_msg["content"] = new_parts
            out.append(new_msg)
        return out

    def generate_payload(self, messages, **kwargs):
        contents = []
        for msg in self._provider_messages(messages):
            role = "user" if msg.get("role") == "user" else "model"
            parts = msg.get("content") or []
            if not isinstance(parts, list):
                parts = [{"text": str(parts)}]
            # Drop any empty parts so Gemini does not error
            parts = [p for p in parts if p]
            if not parts:
                parts = [{"text": ""}]
            contents.append({"role": role, "parts": parts})
        return {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": kwargs.get("max_tokens", 10240),
                "temperature": kwargs.get("temperature", 0.7),
                "topP": kwargs.get("top_p", 0.9),
                "topK": kwargs.get("top_k", 50)
            }
        }

    def invoke(self, messages, **kwargs):
        """
        重写 invoke 方法以处理 Gemini 特有的认证方式 (URL 参数) 和响应解析。
        日志格式与基类 ``GeneralLLMServiceConnector.invoke()`` 对齐，方便排查。
        """
        # Same response-side flag as the base class; pop before payload build.
        preserve_thinking = bool(kwargs.pop("preserve_thinking", False))
        payload = self.generate_payload(messages, **kwargs)
        headers = {"Content-Type": "application/json"}
        # Gemini 认证方式：Token 作为 URL 参数
        url = f"{self.api_url}?key={self.api_token}"

        for attempt in range(self.max_retries):
            is_last_attempt = (attempt == self.max_retries - 1)
            attempt_idx = attempt + 1
            tag = f"[{self.model}] attempt {attempt_idx}/{self.max_retries}"
            req_chars, last_user_chars = self._payload_chars(payload)
            mie_log(
                f"{tag}: POST {url} timeout={self.timeout}s "
                f"request_chars={req_chars} last_user_chars={last_user_chars}"
            )
            attempt_t0 = time.perf_counter()

            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                attempt_elapsed = time.perf_counter() - attempt_t0

                if response.status_code == 200:
                    response_data = response.json()
                    # 适配 Gemini 响应解析: candidates -> content -> parts -> text
                    if not response_data.get("candidates"):
                        raise ValueError(f"No candidates in response. Response: {response.text}")
                    parts = response_data["candidates"][0]["content"]["parts"]
                    text = parts[0].get("text", "") if parts else ""
                    cleaned = self._sanitize_response(text, preserve_thinking=preserve_thinking)
                    # Gemini thinking models put reasoning in parts flagged
                    # ``thought: true`` (or with a ``thoughtsContent`` key). If
                    # the first / non-thought text sanitizes to empty (think
                    # chain consumed the whole budget), fall back to the first
                    # reasoning part so callers still get the model's output.
                    # Mirrors the OpenAI-compat ``reasoning_content`` fallback.
                    if not cleaned:
                        for p in parts[1:]:
                            rtext = p.get("thoughtsContent") or (
                                p.get("text") if p.get("thought") else ""
                            )
                            if rtext:
                                mie_log(
                                    f"{tag} content empty after sanitize; "
                                    f"falling back to Gemini thought part "
                                    f"({len(rtext)} chars)"
                                )
                                cleaned = self._sanitize_response(
                                    rtext, preserve_thinking=preserve_thinking
                                )
                                if cleaned:
                                    break
                    mie_log(
                        f"{tag} ok in {attempt_elapsed:.2f}s "
                        f"response_chars={len(cleaned or '')}"
                    )
                    return cleaned

                if 500 <= response.status_code < 600:
                    body_snip = (response.text or "").replace("\n", " ")[:200]
                    detail = (
                        f"{tag} got HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                        f"body={body_snip!r}"
                    )
                    if is_last_attempt:
                        raise Exception(
                            f"{detail}. Max retries ({self.max_retries}) exceeded."
                        )
                    mie_log(
                        f"{detail}. Retrying in {self.retry_delay} seconds..."
                    )
                    time.sleep(self.retry_delay)
                    continue

                body_snip = (response.text or "").replace("\n", " ")[:200]
                raise Exception(
                    f"{tag} failed with HTTP {response.status_code} in {attempt_elapsed:.2f}s "
                    f"body={body_snip!r}"
                )

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                attempt_elapsed = time.perf_counter() - attempt_t0
                error_type = type(e).__name__
                detail = (
                    f"{tag} {error_type} after {attempt_elapsed:.2f}s "
                    f"(request_chars={req_chars}, last_user_chars={last_user_chars})"
                )
                if is_last_attempt:
                    raise Exception(
                        f"{detail}. Max retries ({self.max_retries}) exceeded."
                    )
                mie_log(
                    f"{detail}. Retrying in {self.retry_delay} seconds... "
                    f"(Attempt {attempt_idx}/{self.max_retries})"
                )
                time.sleep(self.retry_delay)
                continue

            except requests.exceptions.RequestException as e:
                raise Exception(f"[{self.model}] A non-retryable request error occurred: {e}")

            except Exception as e:
                # 捕获其他非网络错误，例如 ValueError（如 No candidates in response）
                raise Exception(f"[{self.model}] Unknown error during API call: {e}")

        raise Exception(
            f"[{self.model}] LLM Service failed after {self.max_retries} attempts due to an unknown error."
        )


class OllamaConnectorGeneral(StandardOpenAICompatibleConnector):
    """Ollama (local) OpenAI-compatible connector.

    Posts to ``{host}/v1/chat/completions``. Ollama ships an OpenAI-compat
    layer from 0.3 onwards that supports vision (``image_url`` content
    parts), JSON mode, tools, streaming, and reasoning models. The
    ``api_key`` header is required by the OpenAI-compat shape but is
    ignored by Ollama, so we pass through whatever the caller provided
    (or fall back to a placeholder so the Authorization header stays
    non-empty).

    ``num_ctx`` / ``keep_alive`` / Modelfile-level knobs are NOT exposed
    on the OpenAI-compat endpoint; configure those via Modelfile instead.

    ``host`` should be the bare Ollama server URL without a trailing
    slash and without ``/v1/chat/completions`` -- those are appended
    here. Examples:

      - ``http://127.0.0.1:11434`` (default, local install)
      - ``http://192.168.1.10:11434`` (LAN host running Ollama)
      - ``http://host.docker.internal:11434`` (Ollama in Docker, reached
        from another container on the same host)
    """

    def __init__(self, host, model, api_token=None, **kwargs):
        api_url = host.rstrip("/") + "/v1/chat/completions"
        # Ollama ignores the api_key but the Authorization header must
        # still be non-empty so requests does not choke on a bare Bearer.
        super().__init__(
            api_url,
            api_token or "ollama",
            model,
            **kwargs,
        )

    # NOTE: do NOT override _sanitize_image_detail. Ollama's OpenAI-compat
    # endpoint accepts ``detail`` on image_url parts (unlike MiniMax /
    # MiMo which reject ``auto``). The base-class identity behavior is
    # the correct default here.


class SetGeneralLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_url": ("STRING", {"default": "https://api.siliconflow.cn/v1/chat/completions"}),
                "api_token": ("STRING", {"default": ""}),
                "model_select": ("STRING", {"default": "deepseek-ai/DeepSeek-V3"}),
            },
            "optional": {
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openai_compatible"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_url, api_token, model_select, config_file="mie_llm_keys.json", config_key="openai_compatible", prefer_local_config=True):
        return (GeneralLLMServiceConnector(api_url, api_token, model_select, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetSiliconFlowLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-ai/DeepSeek-V4-Pro",
                        "deepseek-ai/DeepSeek-V4-Flash",
                        "deepseek-ai/DeepSeek-V3.2",
                        "Pro/deepseek-ai/DeepSeek-V3.2",
                        "deepseek-ai/DeepSeek-V3.1-Terminus",
                        "zai-org/GLM-5.3",
                        "zai-org/GLM-5.2",
                        "Pro/zai-org/GLM-5.1",
                        "zai-org/GLM-4.5V",
                        "Pro/moonshotai/Kimi-K2.6",
                        "moonshotai/Kimi-K2.7-Code",
                        "Qwen/Qwen3.8-27B",
                        "Qwen/Qwen3.6-35B-A3B",
                        "Qwen/Qwen3.6-27B",
                        "Qwen/Qwen3-VL-32B-Thinking",
                        "Qwen/Qwen3-VL-32B-Instruct",
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        "Custom",
                    ],
                    {"default": "deepseek-ai/DeepSeek-V4-Flash"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "siliconflow"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="siliconflow", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "deepseek-ai/DeepSeek-V4-Flash"  # 默认模型
        return (SiliconFlowConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetZhiPuLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "glm-5.3",
                        "glm-5.3-flash",
                        "glm-5.3-flashx",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5-turbo",
                        "glm-5",
                        "glm-4.7",
                        "glm-4.6",
                        "glm-4.5",
                        "glm-4.5-air",
                        "Custom",
                    ],
                    {"default": "glm-5.3"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "zhipu"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="zhipu", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "glm-5.3"  # 默认模型
        return (ZhiPuConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetZhiPuCodeLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "glm-5.3",
                        "glm-5.3-flash",
                        "glm-5.3-flashx",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5-turbo",
                        "glm-5",
                        "glm-4.7",
                        "glm-4.6",
                        "glm-4.5",
                        "glm-4.5-air",
                        "Custom",
                    ],
                    {"default": "glm-5.3"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "zhipu_code"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="zhipu_code", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "glm-5.3"
        return (ZhiPuCodeConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetKimiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "kimi-k3",
                        "kimi-k2.7-code",
                        "kimi-k2.7-code-highspeed",
                        "kimi-k2.6",
                        "Custom",
                    ],
                    {"default": "kimi-k3"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "kimi"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="kimi", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "kimi-k3"  # 默认模型
        return (KimiConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetDeepSeekLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "deepseek-v4-pro",
                        "deepseek-flash",
                        "Custom",
                    ],
                    {"default": "deepseek-flash"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "deepseek"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="deepseek", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "deepseek-flash"  # 默认模型
        return (DeepSeekConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetMiniMaxLLMServiceConnector(object):
    """Standard MiniMax Open Platform connector.

    Use this node when you have a standard MiniMax Open Platform API key
    (`eyJ...` JWT format) issued from the MiniMax developer console. For
    the Token Plan / Coding Plan (`sk-cp-...` prefixed) keys, use
    `SetMiniMaxTokenPlanLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "MiniMax-M2.7",
                        "MiniMax-M2.7-highspeed",
                        "MiniMax-M2.5",
                        "MiniMax-M2.5-highspeed",
                        "MiniMax-M2.1",
                        "MiniMax-M2.1-highspeed",
                        "MiniMax-M2",
                        "Custom",
                    ],
                    {"default": "MiniMax-M2.7"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "minimax_open_platform"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="minimax_open_platform", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "MiniMax-M2.7"
        return (MiniMaxConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetMiniMaxTokenPlanLLMServiceConnector(object):
    """MiniMax Token Plan / Coding Plan connector.

    Use this node when you have a Token Plan / Coding Plan API key
    (`sk-cp-...` prefix). M3 is the headline model on this tier; the
    older M2.7 / M2.5 lineup is kept for back-compat.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "MiniMax-M3",
                        "MiniMax-M2.7",
                        "MiniMax-M2.7-highspeed",
                        "MiniMax-M2.5",
                        "MiniMax-M2.5-highspeed",
                        "MiniMax-M2.1",
                        "MiniMax-M2.1-highspeed",
                        "MiniMax-M2",
                        "Custom",
                    ],
                    {"default": "MiniMax-M3"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "minimax"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="minimax", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "MiniMax-M3"
        return (MiniMaxTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetMiMoLLMServiceConnector(object):
    """Standard Xiaomi MiMo Open Platform connector.

    Use this node when you have a standard MiMo API key (`sk-xxxxx`
    format) issued from the MiMo developer console. For the Token Plan
    / Coding Plan (`tp-xxxxx` prefixed) keys, use
    `SetMiMoTokenPlanLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "mimo-v2.6-pro",
                        "mimo-v2.6-pro-ultraspeed",
                        "mimo-v2.6-flash",
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "Custom",
                    ],
                    {"default": "mimo-v2.6-pro"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "mimo"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="mimo", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "mimo-v2.6-pro"
        return (MiMoConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetMiMoTokenPlanLLMServiceConnector(object):
    """Xiaomi MiMo Token Plan / Coding Plan connector.

    Use this node when you have a Token Plan / Coding Plan API key
    (`tp-xxxxx` prefix). The Token Plan is a fixed-fee subscription
    with its own base URL and billing; the model lineup is shared
    with the standard tier.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "mimo-v2.6-pro",
                        "mimo-v2.6-pro-ultraspeed",
                        "mimo-v2.6-flash",
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "Custom",
                    ],
                    {"default": "mimo-v2.6-pro"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "mimo_token_plan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="mimo_token_plan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "mimo-v2.6-pro"
        return (MiMoTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetDoubaoLLMServiceConnector(object):
    """ByteDance Volcano Ark (火山方舟) Doubao connector.

    Use this node with an Ark API Key from the Volcano console
    (https://console.volcengine.com/ark). The model may be a versioned
    doubao-seed id from the dropdown, or paste an inference Endpoint ID
    (`ep-xxxx`) into `custom_model` with model_select set to `Custom`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "doubao-seed-2-0-pro-260215",
                        "doubao-seed-2-0-lite-260215",
                        "doubao-seed-2-0-mini-260215",
                        "doubao-seed-1-8-251228",
                        "doubao-seed-1-6",
                        "Custom",
                    ],
                    {"default": "doubao-seed-2-0-pro-260215"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "doubao model id or Ark Endpoint ID (ep-xxxx)",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "doubao"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="doubao", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "doubao-seed-2-0-pro-260215"
        return (DoubaoConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetQianfanLLMServiceConnector(object):
    """Baidu Qianfan (百度千帆) ERNIE connector.

    Use this node with a Qianfan ModelBuilder API key
    (https://console.bce.baidu.com/qianfan). Serves the ERNIE family
    plus hosted third-party models (glm / deepseek).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "ernie-5.1",
                        "ernie-5.0",
                        "ernie-5.0-thinking-preview",
                        "ernie-4.5-turbo-128k",
                        "ernie-4.5-turbo-32k",
                        "ernie-4.5-turbo-vl",
                        "glm-5.3",
                        "deepseek-v4.1-flash",
                        "Custom",
                    ],
                    {"default": "ernie-5.1"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "qianfan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="qianfan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "ernie-5.1"
        return (QianfanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetSparkLLMServiceConnector(object):
    """iFLYTEK Spark (讯飞星火) X2 connector.

    Use this node with the APIPassword issued from the iFLYTEK console
    (https://console.xfyun.cn). Serves the current Spark-X2 generation
    via the `/x2/` endpoint (model value `spark-x`); the retired
    lite/generalv3 classics live under `/v1/` — reach them with
    `SetGeneralLLMServiceConnector` instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "spark-x",
                        "Custom",
                    ],
                    {"default": "spark-x"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "spark"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="spark", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "spark-x"
        return (SparkConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetOpenAILLMServiceConnector(object):
    """OpenAI connector (gpt-6 / gpt-5 families).

    Use this node with an OpenAI API key (https://platform.openai.com).
    The payload uses `max_completion_tokens` because the newer
    gpt-5+/o-series models reject the legacy `max_tokens` field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gpt-6-astra",
                        "gpt-6-astra-pro",
                        "gpt-6-luna",
                        "gpt-6-luna-pro",
                        "gpt-5.5",
                        "gpt-5.5-pro",
                        "gpt-5.2-chat",
                        "gpt-5.1",
                        "Custom",
                    ],
                    {"default": "gpt-6-astra"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openai"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="openai", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "gpt-6-astra"
        return (OpenAIConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetGrokLLMServiceConnector(object):
    """xAI Grok connector.

    Use this node with an xAI API key (https://console.x.ai). Serves the
    grok-4.x families via the OpenAI-compatible /v1/chat/completions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "grok-4.7",
                        "grok-4.6",
                        "grok-4.5",
                        "grok-4.3",
                        "grok-4.20",
                        "grok-4.20-multi-agent",
                        "Custom",
                    ],
                    {"default": "grok-4.7"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "grok"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="grok", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "grok-4.7"
        return (GrokConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetOpenRouterLLMServiceConnector(object):
    """OpenRouter aggregator connector.

    Use this node with an OpenRouter API key (https://openrouter.ai) —
    one key routes to 400+ vendor models using `vendor/model` slugs.
    The dropdown carries a cross-vendor selection verified against the
    live catalog; browse everything at https://openrouter.ai/models and
    paste any slug into `custom_model`.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "openrouter/auto",
                        "openai/gpt-6-astra",
                        "anthropic/claude-opus-5.5",
                        "google/gemini-3.8-flash",
                        "x-ai/grok-4.7",
                        "deepseek/deepseek-v4-pro",
                        "z-ai/glm-5.3",
                        "moonshotai/kimi-k3",
                        "qwen/qwen3.5-397b-a17b",
                        "minimax/minimax-m3",
                        "mistralai/mistral-large-2512",
                        "Custom",
                    ],
                    {"default": "openrouter/auto"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Any OpenRouter slug, e.g. openai/gpt-5.2 or vendor/model:free",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "openrouter"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="openrouter", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "openrouter/auto"
        return (OpenRouterConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetClaudeLLMServiceConnector(object):
    """Anthropic Claude connector (official OpenAI-compat layer).

    Use this node with a Claude API key (https://console.claude.com).
    Goes through Anthropic's official OpenAI SDK compatibility layer —
    fine for plain chat / prompt rewriting; features the layer ignores
    (structured outputs, prompt caching, detailed thinking) need the
    native Anthropic API instead.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "claude-sonnet-5-5",
                        "claude-opus-5-5",
                        "claude-fable-5-1",
                        "claude-haiku-4-5",
                        "claude-opus-5",
                        "claude-sonnet-5",
                        "Custom",
                    ],
                    {"default": "claude-sonnet-5-5"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "anthropic"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="anthropic", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "claude-sonnet-5-5"
        return (ClaudeConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetSiliconFlowJevLLMServiceConnector(object):
    """SiliconFlow Jev / System One decision-model connector.

    Use this node to reach the Jev decision-model replicas hosted on
    SiliconFlow (Kev-4B / SemIf / diffusiongemma) via
    `https://api.siliconflow.cn/v1/systemone` — it reuses your existing
    `siliconflow` API key, and the replicas are free during the
    2026-10-08 promo window. Jev models are decision models, not chat
    models — pair this connector with the `CallJevDecision` node, not
    `CallLLMService`. (TypeSafe's own hosted `jev-latest` is out of
    scope here; its protocol is identical if you ever need it.)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "Kev-4B",
                        "SemIf",
                        "diffusiongemma",
                        "Custom",
                    ],
                    {"default": "Kev-4B"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom decision model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "siliconflow"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="siliconflow", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "Kev-4B"
        return (SiliconFlowJevConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetOllamaLLMServiceConnector(object):
    """Set Ollama (local) LLM service connector.

    Connects to a locally running Ollama instance via its OpenAI-compatible
    ``/v1/chat/completions`` endpoint (Ollama 0.3+). Ollama does NOT require
    an API key -- the ``Authorization: Bearer ...`` header is sent but ignored
    by the server, so an empty ``api_token`` is fine.

    Vision models (llava, llama3.2-vision, qwen2.5vl, gemma3, etc.) work
    through the OpenAI-compat layer: just connect an ``IMAGE`` input to the
    downstream ``CallLLMService`` node and it will be forwarded as an
    ``image_url`` content part. Reasoning models (deepseek-r1, qwen3) emit
    ``<think>...</think>`` blocks; the base class strips them automatically.

    Cold-start note: Ollama loads the model from disk into VRAM on first call,
    which can take 30-90s for 7B+ models. The base class retries on timeout,
    but bumping the ``timeout`` input above the 30s base default avoids the
    wasted first attempt for big models.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "host": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "model": ("STRING", {
                    "default": "",
                    "placeholder": "Enter local Ollama model name (e.g. qwen2.5, llama3.2, deepseek-r1, llava)",
                }),
            },
            "optional": {
                "api_token": ("STRING", {"default": ""}),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "ollama"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
                "timeout": ("INT", {"default": 60, "min": 1, "max": 600, "step": 5,
                    "tooltip": "Per-request HTTP timeout in seconds. Default 60s to absorb Ollama's cold-start cost (30-90s for 7B+ models). The base class retries on timeout, but a longer single-attempt timeout avoids the wasted retry."}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, host, model, api_token=None, config_file="mie_llm_keys.json", config_key="ollama", prefer_local_config=True, timeout=60):
        if not model:
            # Sensible default if the user left the field blank. Users can pull
            # other models with `ollama pull <name>` and then edit this field.
            model = "qwen2.5"
        return (OllamaConnectorGeneral(
            host,
            model,
            api_token=api_token,
            config_file=config_file,
            config_key=config_key,
            prefer_local_config=prefer_local_config,
            timeout=timeout,
        ),)


class SetGeminiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gemini-3.8-flash",
                        "gemini-3.7-flash",
                        "gemini-3.5-flash",
                        "gemini-3.5-flash-lite",
                        "gemini-3.1-flash-lite",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash-preview",
                        "Custom",
                    ],
                    {"default": "gemini-3.8-flash"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "gemini"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="gemini", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "gemini-3.8-flash"
        return (GeminiConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetBailianLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen3.8-max",
                        "qwen3.8-flash",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.7-flash",
                        "qwen3.6-flash",
                        "qwen3.6-plus",
                        "qwen3.5-flash",
                        "qwen3.5-plus",
                        "qwen-plus",
                        "qwen-max",
                        "qwen-flash",
                        "qwen-turbo",
                        "qwen-long",
                        "glm-5.3",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5",
                        "kimi-k3",
                        "kimi-k2.6",
                        "deepseek-v4-pro",
                        "deepseek-v4.1-flash",
                        "Custom",
                    ],
                    {"default": "qwen3.8-max"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "自定义模型名（当选择Custom时生效）",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian", prefer_local_config=True):
            model = model_select if model_select != "Custom" else custom_model
            if not model:
                model = "qwen3.8-max"
            return (BailianLLMServiceConnector(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class BailianTokenPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Alibaba Bailian Token Plan connector (subscription tier, Beijing).

    Targets the Bailian Token Plan (个人版, Credits-based) endpoint at
    `https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions`
    with `sk-sp-...` prefixed subscription API keys issued from
    https://bailian.console.aliyun.com/ (key rotates when the plan is
    re-purchased). Covers the Qwen3.8 / Qwen3.7 flagships plus DeepSeek,
    GLM and multimodal families. NOT interchangeable with the PAYG `sk-`
    key or with the Coding Plan key scoped to
    `coding.dashscope.aliyuncs.com`. Pair with
    `SetBailianTokenPlanLLMServiceConnector`.
    """
    api_url = "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class BailianCodingPlanConnectorGeneral(StandardOpenAICompatibleConnector):
    """Alibaba Bailian Coding Plan connector (subscription tier).

    Targets the Bailian Coding Plan endpoint at
    `https://coding.dashscope.aliyuncs.com/v1/chat/completions` with
    `sk-sp-...` prefixed subscription API keys (per the official Coding
    Plan docs — NOT `sk-cp-`). The Coding Plan model list now includes
    general chat models (qwen3.7-plus, glm-5, MiniMax-M2.5, kimi-k2.5)
    alongside the qwen3-coder family; keys are NOT interchangeable with
    the PAYG `sk-` key or with the Token Plan `sk-sp-` key scoped to the
    `token-plan.cn-beijing.maas.aliyuncs.com` host. Pair with
    `SetBailianCodingPlanLLMServiceConnector`.
    """
    api_url = "https://coding.dashscope.aliyuncs.com/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class SetBailianTokenPlanLLMServiceConnector(object):
    """Alibaba Bailian Token Plan connector (multimodal subscription tier).

    Use this node when you have a Token Plan API key (`sk-sp-...` prefix
    issued from https://bailian.console.aliyun.com/). The Token Plan is a
    fixed-fee subscription that grants access to the full Bailian
    multimodal lineup (text / vision / image / audio) via a separate
    endpoint. NOT interchangeable with the PAYG `sk-` key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "auto",
                        "qwen3.8-max",
                        "qwen3.8-flash",
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.6-flash",
                        "deepseek-v4.1-flash",
                        "deepseek-v4-pro",
                        "glm-5.3",
                        "glm-5.2",
                        "Custom",
                    ],
                    {"default": "qwen3.8-max"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian_token_plan"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian_token_plan", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "qwen3.8-max"
        return (BailianTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetBailianCodingPlanLLMServiceConnector(object):
    """Alibaba Bailian Coding Plan connector (subscription tier).

    Use this node when you have a Coding Plan API key (`sk-sp-...` prefix
    issued from https://bailian.console.aliyun.com/ — the official docs
    use the `sk-sp-` prefix for Coding Plan keys too, same shape as Token
    Plan keys but scoped to the `coding.dashscope.aliyuncs.com` host).
    The Coding Plan model list covers both coder models (qwen3-coder-*)
    and general chat models (qwen3.7-plus, glm-5, MiniMax-M2.5, kimi-k2.5).
    NOT interchangeable with the PAYG `sk-` key or with the Token Plan
    `sk-sp-` key.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen3.7-plus",
                        "qwen3.6-plus",
                        "qwen3-coder-plus",
                        "qwen3-coder-next",
                        "glm-5",
                        "MiniMax-M2.5",
                        "kimi-k2.5",
                        "qwen3.5-plus",
                        "qwen3-max-2026-01-23",
                        "glm-4.7",
                        "Custom",
                    ],
                    {"default": "qwen3.7-plus"},
                ),
            },
            "optional": {
                "custom_model": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Enter custom model name (used when model_select is 'Custom')",
                    },
                ),
                "config_file": ("STRING", {"default": "mie_llm_keys.json"}),
                "config_key": ("STRING", {"default": "bailian_coding"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="bailian_coding", prefer_local_config=True):
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "qwen3.7-plus"
        return (BailianCodingPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class CheckLLMServiceConnectivity(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("log",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, llm_service_connector):
        try:
            # 只发一个空消息（有些API需要messages至少有一条，给个简单的提示）
            test_messages = [{"role": "user", "content": "你是什么模型？"}]
            result = llm_service_connector.invoke(test_messages)
            # 只要没报错，说明服务可联通
            return mie_log(f"LLM服务接口可联通 (HTTP 200 + 正常响应), 返回内容: {result}"),
        except Exception as e:
            return mie_log(f"LLM服务检测失败: {str(e)}"),


class CallJevDecision(object):
    """Ask a Jev / System One decision model a set of structured questions.

    Jev decision models do not write prose — they answer structured
    questions about a piece of state text. `questions_json` holds the
    Jev question spec as a JSON object mapping question names to specs:

        {
          "urgency":  {"type": "noul",
                       "instructions": "Does this message express urgency?"},
          "topic":    {"type": "choice",
                       "instructions": "Classify the inquiry",
                       "criteria": {"billing": "Refunds, payments",
                                    "technical": "Defects, troubleshooting"}},
          "severity": {"type": "score",
                       "instructions": "How severe is the issue?",
                       "criteria": ["Low", "Medium", "High"]}
        }

    - `noul`   -> yes/no probability
    - `choice` -> picked option + full probability distribution
    - `score`  -> position on an ordered scale + per-level probabilities

    Returns the full response (`model` / `answers` / `usage`) as a JSON
    string; parse `answers.<name>` downstream for routing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "state": ("STRING", {"default": "", "multiline": True,
                                     "placeholder": "The text to judge, e.g. a customer email or a dialogue line"}),
                "questions_json": ("STRING", {"default": "", "multiline": True,
                                              "placeholder": '{"q1": {"type": "noul", "instructions": "Is this urgent?"}}'}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("decision_json",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, llm_service_connector, state, questions_json):
        if not hasattr(llm_service_connector, "ask"):
            raise Exception(
                "CallJevDecision needs a Jev connector from the "
                "SetSiliconFlowJevLLMServiceConnector node (got a chat connector)."
            )
        try:
            questions = json.loads(questions_json) if questions_json.strip() else {}
        except json.JSONDecodeError as e:
            raise Exception(
                f"questions_json is not valid JSON: {e}. Keep it as a JSON "
                "object mapping question names to noul/choice/score specs."
            )
        if not isinstance(questions, dict) or not questions:
            raise Exception(
                "questions_json must be a non-empty JSON object mapping "
                "question names to specs (see the node tooltip)."
            )
        data = llm_service_connector.ask(state, questions)
        return (json.dumps(data, ensure_ascii=False, indent=2),)


# 通用调用节点：对接任意已创建的 LLMServiceConnector
class CallLLMService(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_service_connector": ("LLMServiceConnector",),
                "input_text": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0}),
                "max_tokens": ("INT", {"default": 512, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0}),
                "image": ("IMAGE",),
                "image_detail": (["auto", "low", "high"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "call"
    CATEGORY = MY_CATEGORY

    @staticmethod
    def _single_image_data_url(image):
        """Backward-compat shim: delegate to the shared `core.utils` helper."""
        return image_tensor_to_data_url(image)

    # Old private name kept as an alias for any external caller.
    _image_to_data_url = _single_image_data_url

    def call(self, llm_service_connector, input_text, temperature=0.7, top_p=0.9, max_tokens=512, seed=None, image=None, image_detail="auto"):
        """
        一个简单的通用节点，将纯文本 / 单图包装为用户消息并调用任意 LLMServiceConnector 的 invoke 方法。
        该节点不会改变底层 connector 的行为或 state。

        文本-only 路径保留 `content: <str>` 形态以维持历史行为；只有带图时才
        切到 `content: [<part>, ...]` 多模态形态。
        """
        image_urls = []
        if image is not None:
            url = image_tensor_to_data_url(image)
            if url:
                image_urls.append(url)
        if image_urls:
            content = build_multimodal_user_content(input_text, image_urls, image_detail=image_detail)
            messages = [{"role": "user", "content": content}]
        else:
            # Text-only: keep content as a plain string for back-compat.
            messages = [{"role": "user", "content": input_text if input_text is not None else ""}]
        # 将可选参数直接转发给 connector.invoke
        result = llm_service_connector.invoke(messages, seed=seed, temperature=temperature, top_p=top_p,
                                              max_tokens=max_tokens)
        return (result.strip(),)
