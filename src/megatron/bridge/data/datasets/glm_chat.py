# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GLM5.2 text chat preprocessing with independently computed assistant masks.

Role-token boundaries are only meaningful under the verified reference template.
Neither strategy appends EOS.
"""

import copy
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase


GLM_MASK_STRATEGIES = frozenset({"glm_generation", "glm_role_tokens"})
CHAT_MASK_STRATEGIES = GLM_MASK_STRATEGIES | {"template"}
_GENERATION_START = "<|assistant|>{% generation %}\n{%- set content"
_ORIGINAL_START = "<|assistant|>\n{%- set content"
_ORIGINAL_END = "{%- elif m.role == 'tool' -%}"
_GENERATION_END = "{% endgeneration %}" + _ORIGINAL_END
_ROLE_MARKERS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|observation|>",
}


@lru_cache(maxsize=1)
def _reference_template() -> str:
    return Path(__file__).with_name("glm5p2_chat_template.jinja").read_text(encoding="utf-8")


@lru_cache(maxsize=8)
def _validate_template(template: str, strategy: str) -> None:
    original = template
    if strategy == "glm_generation":
        if template.count(_GENERATION_START) != 1 or template.count(_GENERATION_END) != 1:
            raise ValueError("glm_generation requires the supported GLM template with generation annotations")
        original = template.replace(_GENERATION_START, _ORIGINAL_START).replace(_GENERATION_END, _ORIGINAL_END)
    if original != _reference_template():
        raise ValueError(f"Unsupported GLM chat template for {strategy}; expected the reference template")


def add_glm_generation_tags(template: str) -> str:
    """Return the supported GLM template with assistant generation annotations.

    This transformation is idempotent and never modifies a tokenizer or file.

    Args:
        template: Original GLM5.2 template, or its already annotated version.

    Returns:
        A template whose rendered text is unchanged.

    Raises:
        ValueError: The template does not match the supported revision.
    """
    if _GENERATION_START in template:
        _validate_template(template, "glm_generation")
        return template
    _validate_template(template, "glm_role_tokens")
    return template.replace(_ORIGINAL_START, _GENERATION_START).replace(_ORIGINAL_END, _GENERATION_END)


def _hf_tokenizer(tokenizer: Any) -> PreTrainedTokenizerBase:
    # Core's current wrapper uses _tokenizer.tokenizer; legacy uses _tokenizer.
    pending = [tokenizer]
    visited = set()
    while pending:
        candidate = pending.pop()
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        if isinstance(candidate, PreTrainedTokenizerBase):
            return candidate
        for name in ("_tokenizer", "tokenizer"):
            nested = getattr(candidate, name, None)
            if nested is not None and id(nested) not in visited:
                pending.append(nested)
        if len(visited) > 8:
            break
    raise ValueError("GLM preprocessing requires a Hugging Face tokenizer or a supported Core wrapper")


def _template_options(options: dict | None) -> dict:
    if options is not None and not isinstance(options, dict):
        raise ValueError("chat_template_kwargs must be a dictionary")
    if any(not isinstance(key, str) for key in options or {}):
        raise ValueError("chat_template_kwargs keys must be strings")
    result = {"clear_thinking": True, "enable_thinking": True, "reasoning_effort": "max"}
    unknown = set(options or {}) - result.keys()
    if unknown:
        raise ValueError(f"Unsupported GLM chat_template_kwargs: {sorted(unknown)}")
    result.update(options or {})
    if result["clear_thinking"] is not True:
        raise ValueError("GLM SFT requires clear_thinking=True to retain reasoning only in the last user turn")
    if not isinstance(result["enable_thinking"], bool):
        raise ValueError("enable_thinking must be a bool")
    if result["reasoning_effort"] not in ("max", "high"):
        raise ValueError("reasoning_effort must be 'max' or 'high'")
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError("Non-finite JSON value")


def _normalize_glm_chat(source: dict | list, tool_schemas: list | None) -> tuple[list[dict], list | None]:
    if isinstance(source, dict):
        messages = source.get("messages")
        tools = source.get("tools")
        if tools is None:
            tools = tool_schemas
    else:
        messages, tools = source, tool_schemas
    if not isinstance(messages, list) or not messages:
        raise ValueError("GLM SFT requires a non-empty OpenAI messages list")
    if tools is not None and (not isinstance(tools, list) or any(not isinstance(t, dict) for t in tools)):
        raise ValueError("GLM tools must be a list of tool definitions")
    messages, tools = copy.deepcopy((messages, tools))
    for index, tool in enumerate(tools or []):
        definition = tool.get("function", tool)
        if (
            tool.get("type", "function") != "function"
            or not isinstance(definition, dict)
            or not isinstance(definition.get("name"), str)
            or not definition["name"]
        ):
            raise ValueError(f"tools[{index}]: a named function definition is required")
        try:
            json.dumps(tool, allow_nan=False)
        except (TypeError, ValueError):
            raise ValueError(f"tools[{index}]: tool definition must be finite JSON") from None
    for index, message in enumerate(messages):
        location = f"message[{index}]"
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message["role"] not in _ROLE_MARKERS
        ):
            raise ValueError(f"{location}: unsupported or missing role")
        if message["role"] == "assistant" and message.get("content") is None:
            message["content"] = ""
        if not isinstance(message.get("content"), str):
            raise ValueError(f"{location}: GLM SFT currently supports text content only")
        if message.get("reasoning_content") is not None and not isinstance(message["reasoning_content"], str):
            raise ValueError(f"{location}: reasoning_content must be a string or null")
        calls = message.get("tool_calls")
        if calls is None:
            calls = []
        if not isinstance(calls, list) or (calls and message["role"] != "assistant"):
            raise ValueError(f"{location}: tool_calls must be an assistant list")
        for call_index, call in enumerate(calls):
            call_location = f"{location}.tool_calls[{call_index}]"
            if not isinstance(call, dict) or call.get("type", "function") != "function":
                raise ValueError(f"{call_location}: only function calls are supported")
            function = call.get("function")
            if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
                raise ValueError(f"{call_location}: a named function is required")
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments, parse_constant=_reject_json_constant)
                except ValueError:
                    raise ValueError(f"{call_location}: arguments must contain valid JSON") from None
            if not isinstance(arguments, dict):
                raise ValueError(f"{call_location}: arguments must be a JSON object")
            try:
                json.dumps(arguments, allow_nan=False)
            except (TypeError, ValueError):
                raise ValueError(f"{call_location}: arguments must be a finite JSON object") from None
            function["arguments"] = arguments
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("GLM SFT requires a user turn")
    if messages[-1]["role"] != "assistant":
        raise ValueError("GLM SFT requires the last message to be assistant")
    last = messages[-1]
    if not (last["content"].strip() or last.get("reasoning_content") or last.get("tool_calls")):
        raise ValueError("GLM SFT requires a non-empty final assistant target")
    return messages, tools


def _role_boundaries(
    hf: PreTrainedTokenizerBase, ids: list[int], messages: list[dict], tools: list | None, options: dict
) -> list[tuple[int, str]]:
    role_ids = {}
    vocab = hf.get_vocab()
    for role, marker in _ROLE_MARKERS.items():
        token_id = vocab.get(marker)
        if token_id is None or hf.encode(marker, add_special_tokens=False) != [token_id]:
            raise ValueError(f"GLM role marker {marker} must encode as one vocabulary token")
        role_ids[role] = token_id
    if len(set(role_ids.values())) != len(role_ids):
        raise ValueError("GLM role markers must have distinct token IDs")
    id_roles = {token_id: role for role, token_id in role_ids.items()}
    boundaries = [(index, id_roles[token_id]) for index, token_id in enumerate(ids) if token_id in id_roles]
    expected = ["system"] if options["enable_thinking"] else []
    if tools:
        expected.append("system")
    for index, message in enumerate(messages):
        role = message["role"]
        if role == "tool" and index > 0 and messages[index - 1]["role"] == "tool":
            continue
        expected.append(role)
    if [role for _, role in boundaries] != expected:
        raise ValueError(
            "GLM role-token sequence does not match message structure; "
            "literal role tokens in rendered message/tool content are unsupported"
        )
    return boundaries


def preprocess_glm_chat(
    source: dict | list,
    tokenizer: Any,
    tool_schemas: list | None = None,
    *,
    strategy: str,
    template_kwargs: dict | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize a complete GLM conversation and compute assistant supervision.

    Args:
        source: OpenAI messages, optionally with per-sample tools.
        tokenizer: HF tokenizer or Core tokenizer wrapper.
        tool_schemas: Tools used when the sample does not provide tools.
        strategy: ``glm_generation`` or ``glm_role_tokens``.
        template_kwargs: Thinking options shared by full and prompt rendering.

    Returns:
        Input tokens, assistant loss mask, inference context and answer tensors.

    Raises:
        ValueError: Unsupported template, invalid input or inconsistent boundaries.
    """
    if strategy not in GLM_MASK_STRATEGIES:
        raise ValueError(f"Unknown GLM mask strategy: {strategy}")
    hf = _hf_tokenizer(tokenizer)
    template = getattr(tokenizer, "chat_template", None)
    if not isinstance(template, str):
        template = hf.chat_template
    if not isinstance(template, str):
        raise ValueError("GLM SFT requires a single string chat template")
    _validate_template(template, strategy)
    options = _template_options(template_kwargs)
    messages, tools = _normalize_glm_chat(source, tool_schemas)
    if not options["enable_thinking"]:
        last_user = max(index for index, message in enumerate(messages) if message["role"] == "user")
        for message in messages[last_user + 1 :]:
            if message["role"] == "assistant" and (
                message.get("reasoning_content") or "</think>" in message["content"]
            ):
                raise ValueError("enable_thinking=False is incompatible with reasoning in the last user turn")
    render_kwargs = dict(chat_template=template, tools=tools, tokenize=True, return_dict=True, **options)
    encoded = hf.apply_chat_template(
        messages,
        add_generation_prompt=False,
        return_assistant_tokens_mask=strategy == "glm_generation",
        **render_kwargs,
    )
    ids = encoded["input_ids"]
    boundaries = _role_boundaries(hf, ids, messages, tools, options)
    if strategy == "glm_generation":
        # Keep this independent of the role-mask implementation for A/B validation.
        mask = encoded["assistant_masks"]
    else:
        mask = [0] * len(ids)
        for index, (start, role) in enumerate(boundaries):
            if role == "assistant":
                end = boundaries[index + 1][0] if index + 1 < len(boundaries) else len(ids)
                mask[start + 1 : end] = [1] * (end - start - 1)
    if len(ids) != len(mask) or not any(mask[1:]):
        raise ValueError("GLM assistant mask must match input length and contain next-token supervision")
    prompt_ids = hf.apply_chat_template(messages[:-1], add_generation_prompt=True, **render_kwargs)["input_ids"]
    if len(prompt_ids) >= len(ids) or ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("GLM inference prompt is not a strict token prefix of the full conversation")
    return {
        "input_ids": torch.tensor(ids, dtype=torch.long),
        "loss_mask": torch.tensor(mask, dtype=torch.bool),
        "context_ids": torch.tensor(prompt_ids, dtype=torch.long),
        "answer_ids": torch.tensor(ids[len(prompt_ids) :], dtype=torch.long),
    }
