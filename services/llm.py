import re
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
        return cleaned.strip()

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
            mie_log(f"{tag}: POST {self.api_url} timeout={self.timeout}s")
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
                    f"{tag} {error_type} after {attempt_elapsed:.2f}s"
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
    `/api/paas/v4/chat/completions` endpoint with regular `eyJ...`
    API keys. For the Coding / Token Plan subscription
    (`/api/coding/...` endpoint), use `ZhiPuCodeConnectorGeneral`
    and `SetZhiPuCodeLLMServiceConnector` instead.
    """
    api_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class ZhiPuCodeConnectorGeneral(StandardOpenAICompatibleConnector):
    """ZhiPu Coding / Token Plan connector.

    Targets the ZhiPu Coding Plan endpoint at `/api/coding/...`
    with a Token Plan / Coding Plan subscription key. Distinct from
    the standard ZhiPu BigModel API in URL, billing, and model
    lineup (GLM-5 / GLM-4.7 series rather than GLM-4 / GLM-Z1).
    Pair with `SetZhiPuCodeLLMServiceConnector`.
    """
    api_url = "https://open.bigmodel.cn/api/coding/paas/v4/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class KimiConnectorGeneral(StandardOpenAICompatibleConnector):
    api_url = "https://api.moonshot.cn/v1/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        super().__init__(self.api_url, api_token, model, **kwargs)


class GithubModelsConnectorGeneral(GeneralLLMServiceConnector):
    api_url = "https://models.github.ai/inference/chat/completions"

    def __init__(self, api_token, model, **kwargs):
        # 继承 GeneralLLMServiceConnector 的默认 Payload
        super().__init__(self.api_url, api_token, model, **kwargs)


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
            mie_log(f"{tag}: POST {url} timeout={self.timeout}s")
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
                detail = f"{tag} {error_type} after {attempt_elapsed:.2f}s"
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


class SetGithubModelsLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "openai/gpt-4.1",
                        "openai/gpt-4.1-mini",
                        "openai/gpt-4.1-nano",
                        "openai/gpt-5-chat",
                        "openai/gpt-5-mini",
                        "openai/o4-mini",
                        "deepseek/deepseek-v3-0324",
                        "deepseek/deepseek-r1-0528",
                        "meta/llama-4-maverick-17b-128e-instruct-fp8",
                        "meta/llama-4-scout-17b-16e-instruct",
                        "meta/llama-3.3-70b-instruct",
                        "Custom",
                    ],
                    {"default": "openai/gpt-4.1"},
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
                "config_key": ("STRING", {"default": "github_models"}),
                "prefer_local_config": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("LLMServiceConnector",)
    RETURN_NAMES = ("llm_service_connector",)
    FUNCTION = "execute"
    CATEGORY = MY_CATEGORY

    def execute(self, api_token, model_select, custom_model="", config_file="mie_llm_keys.json", config_key="github_models", prefer_local_config=True):
        # 确定最终使用的模型
        model = model_select if model_select != "Custom" else custom_model
        if not model:
            model = "openai/gpt-4.1"  # 默认模型
        return (GithubModelsConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


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
                        "deepseek-ai/DeepSeek-V3",
                        "Pro/zai-org/GLM-5.1",
                        "THUDM/GLM-4-32B-0414",
                        "zai-org/GLM-4.5V",
                        "Pro/moonshotai/Kimi-K2.6",
                        "Qwen/Qwen3.6-35B-A3B",
                        "Qwen/Qwen3.5-397B-A17B",
                        "Qwen/Qwen3-VL-32B-Instruct",
                        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        "Qwen/Qwen3-8B",
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
                    {"default": "glm-5.1"},
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
            model = "glm-5.1"  # 默认模型
        return (ZhiPuConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetZhiPuCodeLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
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
                    {"default": "glm-5.1"},
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
            model = "glm-5.1"
        return (ZhiPuCodeConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetKimiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "kimi-k2.7-code",
                        "kimi-k2.7-code-highspeed",
                        "kimi-k2.6",
                        "kimi-k2.5",
                        "moonshot-v1-128k",
                        "moonshot-v1-128k-vision-preview",
                        "moonshot-v1-32k",
                        "moonshot-v1-32k-vision-preview",
                        "moonshot-v1-8k",
                        "moonshot-v1-8k-vision-preview",
                        "moonshot-v1-auto",
                        "Custom",
                    ],
                    {"default": "kimi-k2.6"},
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
            model = "kimi-k2.6"  # 默认模型
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
                        "deepseek-v4-flash",
                        "Custom",
                    ],
                    {"default": "deepseek-v4-flash"},
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
            model = "deepseek-v4-flash"  # 默认模型
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
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "mimo-v2-omni",
                        "mimo-v2-flash",
                        "mimo-v2-pro",
                        "Custom",
                    ],
                    {"default": "mimo-v2.5-pro"},
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
            model = "mimo-v2.5-pro"
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
                        "mimo-v2.5-pro",
                        "mimo-v2.5",
                        "mimo-v2-omni",
                        "mimo-v2-flash",
                        "mimo-v2-pro",
                        "Custom",
                    ],
                    {"default": "mimo-v2.5-pro"},
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
            model = "mimo-v2.5-pro"
        return (MiMoTokenPlanConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetGeminiLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "gemini-3.1-pro",
                        "gemini-3.1-pro-preview",
                        "gemini-3-flash",
                        "gemini-3.1-flash-lite",
                        "gemini-2.5-pro",
                        "gemini-2.5-flash",
                        "gemini-2.5-flash-lite",
                        "Custom",
                    ],
                    {"default": "gemini-3.1-pro"},
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
            model = "gemini-3.1-pro"
        return (GeminiConnectorGeneral(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


class SetBailianLLMServiceConnector(object):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_token": ("STRING", {"default": ""}),
                "model_select": (
                    [
                        "qwen3.7-max",
                        "qwen3.7-plus",
                        "qwen3.6-flash",
                        "qwen3.6-plus",
                        "qwen3.5-flash",
                        "qwen3.5-plus",
                        "qwen-plus",
                        "qwen-max",
                        "qwen-flash",
                        "qwen-turbo",
                        "qwen-long",
                        "glm-5.2",
                        "glm-5.1",
                        "glm-5",
                        "kimi-k2.6",
                        "deepseek-v4-pro",
                        "Custom",
                    ],
                    {"default": "qwen3.7-max"},
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
            model = "qwen3.7-max"
        return (BailianLLMServiceConnector(api_token, model, config_file=config_file, config_key=config_key, prefer_local_config=prefer_local_config),)


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
