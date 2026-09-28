import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from megatron.bridge.data.datasets.glm_chat import (
    GLM_MASK_STRATEGIES,
    add_glm_generation_tags,
    preprocess_glm_chat,
)
from megatron.bridge.data.datasets.packed_sequence import tokenize_dataset
from megatron.bridge.data.datasets.sft import GPTSFTChatDataset, create_sft_dataset
from megatron.bridge.data.datasets.utils import _chat_preprocess


pytestmark = pytest.mark.unit
TEMPLATE = (Path(__file__).parent / "fixtures/glm5p2_chat_template.jinja").read_text(encoding="utf-8")
ROLES = ["<|system|>", "<|user|>", "<|assistant|>", "<|observation|>"]


@pytest.fixture
def tokenizer_pair():
    special = [
        "[UNK]",
        "<|endoftext|>",
        "[gMASK]",
        "<sop>",
        *ROLES,
        "<think>",
        "</think>",
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
        "<tool_response>",
        "</tool_response>",
    ]
    words = ["OLD_REASON", "OLD_ANSWER", "NEW_CALL_REASON", "NEW_FINAL_REASON", "FINAL_ANSWER", "TOOL_A", "TOOL_B"]
    vocabulary = {word: index for index, word in enumerate(special + words)}
    pair = {}
    for strategy in sorted(GLM_MASK_STRATEGIES):
        backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            eos_token="<|endoftext|>",
            pad_token="<|endoftext|>",
            additional_special_tokens=special[2:],
        )
        tokenizer.chat_template = add_glm_generation_tags(TEMPLATE) if strategy == "glm_generation" else TEMPLATE
        pair[strategy] = tokenizer
    return pair


@pytest.fixture
def conversation():
    return {
        "uuid": "synthetic-agentic",
        "messages": [
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "OLD_ANSWER", "reasoning_content": "OLD_REASON"},
            {"role": "user", "content": "new question"},
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "NEW_CALL_REASON",
                "tool_calls": [
                    {"id": "a", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}},
                    {"id": "b", "type": "function", "function": {"name": "g", "arguments": {"nested": [True, None]}}},
                ],
            },
            {"role": "tool", "tool_call_id": "a", "content": "TOOL_A"},
            {"role": "tool", "tool_call_id": "b", "content": "TOOL_B"},
            {"role": "assistant", "content": "FINAL_ANSWER", "reasoning_content": "NEW_FINAL_REASON"},
        ],
        "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    }


def _pair_results(pair, source, **kwargs):
    results = {
        strategy: _chat_preprocess(source, tokenizer, chat_template_mask_strategy=strategy, **kwargs)
        for strategy, tokenizer in pair.items()
    }
    for key in ("input_ids", "loss_mask", "context_ids", "answer_ids"):
        assert torch.equal(results["glm_generation"][key], results["glm_role_tokens"][key]), key
    return results["glm_role_tokens"]


def test_generation_patch_is_exact_and_idempotent():
    modified = add_glm_generation_tags(TEMPLATE)
    assert modified != TEMPLATE
    assert add_glm_generation_tags(modified) == modified
    with pytest.raises(ValueError, match="reference template"):
        add_glm_generation_tags(TEMPLATE + " ")


@pytest.mark.parametrize("effort", ["max", "high"])
def test_independent_masks_and_supervised_content(tokenizer_pair, conversation, effort):
    original = copy.deepcopy(conversation)
    result = _pair_results(tokenizer_pair, conversation, chat_template_kwargs={"reasoning_effort": effort})
    assert conversation == original
    hf = tokenizer_pair["glm_role_tokens"]
    ids, mask = result["input_ids"], result["loss_mask"]
    for word in ("OLD_ANSWER", "NEW_CALL_REASON", "NEW_FINAL_REASON", "FINAL_ANSWER"):
        positions = ids == hf.convert_tokens_to_ids(word)
        assert positions.any() and mask[positions].all()
    assert not (ids == hf.convert_tokens_to_ids("OLD_REASON")).any()
    for word in ("TOOL_A", "TOOL_B", *ROLES):
        positions = ids == hf.convert_tokens_to_ids(word)
        assert positions.any() and not mask[positions].any()
    assert (ids == hf.convert_tokens_to_ids("<|observation|>")).sum() == 1
    call_tags = ids == hf.convert_tokens_to_ids("<tool_call>")
    assert mask[call_tags].sum() == 2
    assert (~mask[call_tags]).sum() == 1  # The system prompt contains a format example.
    assert torch.equal(torch.cat([result["context_ids"], result["answer_ids"]]), ids)
    assert result["context_ids"][-1].item() == hf.convert_tokens_to_ids("<think>")
    assert result["answer_ids"][0].item() == hf.convert_tokens_to_ids("NEW_FINAL_REASON")
    assert result["input_ids"].dtype == torch.long and result["loss_mask"].dtype == torch.bool
    assert tokenizer_pair["glm_role_tokens"].chat_template == TEMPLATE


def test_embedded_reasoning_and_final_tool_call(tokenizer_pair, conversation):
    for message in conversation["messages"]:
        if message["role"] == "assistant":
            reason = message.pop("reasoning_content")
            message["content"] = f"<think>{reason}</think>" + (message["content"] or "")
    _pair_results(tokenizer_pair, conversation)
    conversation["messages"] = conversation["messages"][:4]
    _pair_results(tokenizer_pair, conversation)


def test_no_thinking_and_empty_tools(tokenizer_pair):
    source = {
        "messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "FINAL_ANSWER"}],
        "tools": [],
    }
    result = _pair_results(
        tokenizer_pair, source, tool_schemas=[{"name": "fallback"}], chat_template_kwargs={"enable_thinking": False}
    )
    hf = tokenizer_pair["glm_role_tokens"]
    assert hf.convert_tokens_to_ids("<|system|>") not in result["input_ids"]
    assert result["context_ids"][-1] == hf.convert_tokens_to_ids("</think>")


@pytest.mark.parametrize("tools", [None, []])
def test_tools_fallback(tokenizer_pair, conversation, tools):
    conversation["tools"] = tools
    _pair_results(tokenizer_pair, conversation, tool_schemas=[{"name": "fallback"}])
    del conversation["tools"]
    _pair_results(tokenizer_pair, conversation, tool_schemas=[{"name": "fallback"}])


@pytest.mark.parametrize("arguments", ["not json", "[]", "null", "NaN", 2, [1], {"x": float("nan")}])
def test_bad_arguments_have_same_errors(tokenizer_pair, conversation, arguments):
    conversation["messages"][3]["tool_calls"][0]["function"]["arguments"] = arguments
    errors = []
    for strategy, hf in tokenizer_pair.items():
        with pytest.raises(ValueError, match=r"message\[3\].tool_calls\[0\].*arguments") as error:
            preprocess_glm_chat(conversation, hf, strategy=strategy)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


@pytest.mark.parametrize("where", ["user", "tool", "reasoning", "schema", "arguments"])
def test_literal_role_token_rejected_by_both(tokenizer_pair, conversation, where):
    literal = "literal <|assistant|>"
    if where == "user":
        conversation["messages"][0]["content"] = literal
    elif where == "tool":
        conversation["messages"][4]["content"] = literal
    elif where == "reasoning":
        conversation["messages"][-1]["reasoning_content"] = literal
    elif where == "schema":
        conversation["tools"][0]["function"]["description"] = literal
    else:
        conversation["messages"][3]["tool_calls"][0]["function"]["arguments"] = {"x": literal}
    for strategy, hf in tokenizer_pair.items():
        with pytest.raises(ValueError, match="role-token sequence"):
            preprocess_glm_chat(conversation, hf, strategy=strategy)


@pytest.mark.parametrize(
    "options",
    [
        {"clear_thinking": False},
        {"reasoning_effort": "low"},
        {"tokenize": False},
        {"enable_thinking": "yes"},
        {"enable_thinking": False},
    ],
)
def test_invalid_options(tokenizer_pair, conversation, options):
    for strategy, hf in tokenizer_pair.items():
        with pytest.raises(ValueError):
            preprocess_glm_chat(conversation, hf, strategy=strategy, template_kwargs=options)


@pytest.mark.parametrize(
    "messages",
    [
        [],
        [{"role": "assistant", "content": "hi"}],
        [{"role": "user", "content": "hi"}],
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}],
    ],
)
def test_invalid_conversation(tokenizer_pair, messages):
    for strategy, hf in tokenizer_pair.items():
        with pytest.raises(ValueError):
            preprocess_glm_chat({"messages": messages}, hf, strategy=strategy)


def test_legacy_wrapper_matches_current_path(tokenizer_pair, conversation):
    for strategy, hf in tokenizer_pair.items():
        legacy = SimpleNamespace(_tokenizer=hf, legacy=True)
        actual = _chat_preprocess(conversation, legacy, chat_template_mask_strategy=strategy)
        expected = preprocess_glm_chat(conversation, hf, strategy=strategy)
        for key in expected:
            assert torch.equal(actual[key], expected[key])


@pytest.mark.parametrize("strategy", sorted(GLM_MASK_STRATEGIES))
def test_unknown_template_is_rejected(tokenizer_pair, conversation, strategy):
    tokenizer_pair[strategy].chat_template += " "
    with pytest.raises(ValueError, match="reference template"):
        preprocess_glm_chat(conversation, tokenizer_pair[strategy], strategy=strategy)


@pytest.mark.parametrize("strategy", sorted(GLM_MASK_STRATEGIES))
def test_prompt_mismatch_is_not_hidden(tokenizer_pair, conversation, strategy, monkeypatch):
    hf = tokenizer_pair[strategy]
    original = hf.apply_chat_template

    def render(*args, **kwargs):
        result = original(*args, **kwargs)
        if kwargs.get("add_generation_prompt"):
            result["input_ids"][0] = -1
        return result

    monkeypatch.setattr(hf, "apply_chat_template", render)
    with pytest.raises(ValueError, match="strict token prefix"):
        preprocess_glm_chat(conversation, hf, strategy=strategy)


@pytest.mark.parametrize("strategy", sorted(GLM_MASK_STRATEGIES))
def test_non_atomic_role_marker_rejected(tokenizer_pair, conversation, strategy, monkeypatch):
    hf = tokenizer_pair[strategy]
    original = hf.encode

    def encode(text, **kwargs):
        return [1, 2] if text == "<|assistant|>" else original(text, **kwargs)

    monkeypatch.setattr(hf, "encode", encode)
    with pytest.raises(ValueError, match="must encode as one"):
        preprocess_glm_chat(conversation, hf, strategy=strategy)


def test_generation_mask_is_never_replaced(tokenizer_pair, conversation, monkeypatch):
    hf = tokenizer_pair["glm_generation"]
    original = hf.apply_chat_template

    def render(*args, **kwargs):
        result = original(*args, **kwargs)
        result["assistant_masks"] = [0] * len(result["input_ids"])
        return result

    monkeypatch.setattr(hf, "apply_chat_template", render)
    with pytest.raises(ValueError, match="next-token supervision"):
        preprocess_glm_chat(conversation, hf, strategy="glm_generation")


@pytest.mark.parametrize(
    "field,value",
    [
        ("tools", {}),
        ("tools", [{"function": None}]),
        ("content", []),
        ("role", []),
        ("tool_calls", {}),
        ("reasoning_content", 3),
    ],
)
def test_invalid_field_types(tokenizer_pair, conversation, field, value):
    if field == "tools":
        conversation[field] = value
    else:
        conversation["messages"][-1][field] = value
    errors = []
    for strategy, hf in tokenizer_pair.items():
        with pytest.raises(ValueError) as error:
            preprocess_glm_chat(conversation, hf, strategy=strategy)
        errors.append(str(error.value))
    assert errors[0] == errors[1]


def test_real_core_wrapper_and_factory(tokenizer_pair, conversation, tmp_path):
    from megatron.core.tokenizers import MegatronTokenizer

    hf = tokenizer_pair["glm_role_tokens"]
    hf.save_pretrained(tmp_path / "tokenizer")
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(conversation) + "\n", encoding="utf-8")
    wrapper = MegatronTokenizer.from_pretrained(
        tokenizer_path=str(tmp_path / "tokenizer"), metadata_path={"library": "huggingface"}
    )
    dataset = create_sft_dataset(
        path,
        wrapper,
        seq_length=4096,
        chat=True,
        use_hf_tokenizer_chat_template=True,
        chat_template_mask_strategy="glm_role_tokens",
        chat_template_kwargs={"reasoning_effort": "high"},
        hf_dataset=True,
        index_mapping_dir=str(tmp_path / "cache"),
        memmap_workers=1,
    )
    result = dataset[0]
    expected = _chat_preprocess(
        conversation,
        hf,
        chat_template_mask_strategy="glm_role_tokens",
        chat_template_kwargs={"reasoning_effort": "high"},
    )
    assert torch.equal(result["input_ids"], expected["input_ids"])
    assert result["metadata"]["uuid"] == conversation["uuid"]
    batch = dataset.collate_fn([result])
    length = len(result["input_ids"]) - 1
    assert torch.equal(batch["tokens"][0, :length], result["input_ids"][:-1])
    assert torch.equal(batch["labels"][0, :length], result["input_ids"][1:])
    assert torch.equal(batch["loss_mask"][0, :length].bool(), result["loss_mask"][1:])
    assert not batch["loss_mask"][0, length:].any()


def _length_dataset(length, reserve=0):
    dataset = object.__new__(GPTSFTChatDataset)
    dataset.file_path = "fixture.jsonl"
    dataset.max_seq_length = length
    dataset.tokens_to_generate = reserve
    dataset.tokenizer = SimpleNamespace(eos_id=0)
    dataset.pad_to_max_length = False
    dataset.pad_seq_length_to_mult = 1
    dataset.ceil_to_power_2 = False
    dataset.get_attention_mask_from_fusion = True
    return dataset


def _sample(length=17, context=5):
    return {
        "input_ids": torch.arange(length),
        "loss_mask": torch.tensor([False] * context + [True] * (length - context)),
        "context_ids": torch.arange(context),
        "answer_ids": torch.arange(context, length),
        "metadata": {"uuid": "boundary"},
    }


def test_collate_exact_length_and_overflow():
    dataset = _length_dataset(16)
    assert dataset.collate_fn([_sample()])["tokens"].shape == (1, 16)
    with pytest.raises(ValueError, match="training_length=17.*Manually increase"):
        dataset.collate_fn([_sample(), _sample(18)])
    with pytest.raises(ValueError, match="tokens_to_generate=12"):
        _length_dataset(16, 12).collate_fn([_sample()])
    with pytest.raises(ValueError, match="Invalid chat loss_mask"):
        bad = _sample()
        bad["loss_mask"].zero_()
        dataset.collate_fn([bad])


@pytest.mark.parametrize("strategy", sorted(GLM_MASK_STRATEGIES))
def test_packing_rejected_before_side_effects(strategy, tmp_path):
    kwargs = {"chat_template_mask_strategy": strategy, "chat": True, "use_hf_tokenizer_chat_template": True}
    before = copy.deepcopy(kwargs)
    with pytest.raises(ValueError, match="do not yet support sequence packing"):
        tokenize_dataset(tmp_path / "missing.jsonl", None, 4096, 1234, dataset_kwargs=kwargs)
    assert kwargs == before
    for name in ("packed.npy", "packed.parquet"):
        with pytest.raises(ValueError, match="do not yet support sequence packing"):
            create_sft_dataset(tmp_path / name, None, **kwargs)
