# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import TokenizersBackend

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.reasoning.apertus_reasoning_parser import (
    INNER_PREFIX,
    INNER_SUFFIX,
    ApertusReasoningParser,
)

INNER = (INNER_PREFIX, INNER_SUFFIX)
ASSISTANT_START = "<|assistant_start|>"
TOOLS_START, TOOLS_END = "<|tools_prefix|>", "<|tools_suffix|>"
TOOL_PAYLOAD = '[{"get_weather": {"city": "Bern"}}]'


@pytest.fixture
def apertus_tokenizer() -> TokenizersBackend:
    tokenizer = TokenizersBackend(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
        unk_token="<unk>",
    )
    tokenizer.add_tokens([*INNER, ASSISTANT_START, TOOLS_START, TOOLS_END])
    return tokenizer


@pytest.mark.parametrize("streaming", [True, False], ids=["stream", "full"])
def test_reasoning_delimiters(apertus_tokenizer, streaming):
    parser = ApertusReasoningParser(apertus_tokenizer)
    output = [INNER[0], "Thought", INNER[1], "Answer"]

    assert run_reasoning_extraction(parser, output, streaming=streaming) == (
        "Thought",
        "Answer",
    )
    assert run_reasoning_extraction(
        ApertusReasoningParser(apertus_tokenizer), ["Answer"], streaming=streaming
    ) == (None, "Answer")

    completed_reasoning_ids = apertus_tokenizer.encode("".join(output[:3]))
    if streaming:
        assert parser.is_reasoning_end_streaming(
            completed_reasoning_ids, apertus_tokenizer.encode(INNER[1])
        )
    else:
        assert parser.is_reasoning_end(completed_reasoning_ids)


@pytest.mark.parametrize(
    "prompt, output, streaming, expected",
    [
        pytest.param(
            [ASSISTANT_START, INNER[0]],
            ["Thought", INNER[1], "Answer"],
            True,
            ("Thought", "Answer"),
            id="continue-open-stream",
        ),
        pytest.param(
            [ASSISTANT_START, INNER[0]],
            ["Thought"],
            False,
            ("Thought", None),
            id="continue-open-full",
        ),
        pytest.param(
            [INNER[0], ASSISTANT_START],
            ["Answer"],
            True,
            (None, "Answer"),
            id="new-turn",
        ),
    ],
)
def test_prompt_controls_reasoning(
    apertus_tokenizer, prompt, output, streaming, expected
):
    parser = ApertusReasoningParser(apertus_tokenizer)
    parser.set_prompt_token_ids(apertus_tokenizer.encode("".join(prompt)))

    assert run_reasoning_extraction(parser, output, streaming=streaming) == expected


def test_tool_call_ends_reasoning(apertus_tokenizer):
    parser = ApertusReasoningParser(apertus_tokenizer)
    call = TOOLS_START + TOOL_PAYLOAD + TOOLS_END
    output = INNER[0] + "Thought" + call

    assert run_reasoning_extraction(parser, [output]) == (
        "Thought",
        call,
    )
    call_ids = apertus_tokenizer.encode(call)
    output_ids = apertus_tokenizer.encode(output)
    assert parser.is_reasoning_end_streaming(output_ids, call_ids)
    assert parser.extract_content_ids(output_ids) == call_ids


@pytest.mark.parametrize(
    "prompt, start_chunks",
    [
        pytest.param("", [INNER[0]], id="generated-open"),
        pytest.param(ASSISTANT_START + INNER[0], [], id="prompt-open"),
    ],
)
def test_streaming_reasoning_to_tool_handoff(apertus_tokenizer, prompt, start_chunks):
    parser_cls = ParserManager.get_parser(
        reasoning_parser_name="apertus",
        tool_parser_name="apertus",
        enable_auto_tools=True,
    )
    assert parser_cls is not None
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=[{"type": "function", "function": {"name": "get_weather"}}],
        tool_choice="auto",
    )
    parser = parser_cls(apertus_tokenizer, tools=request.tools)
    prompt_token_ids = apertus_tokenizer.encode(prompt, add_special_tokens=False)
    chunks = [*start_chunks, "Thought", TOOLS_START, TOOL_PAYLOAD, TOOLS_END]
    deltas = []
    for i, chunk in enumerate(chunks):
        delta = parser.parse_delta(
            chunk,
            apertus_tokenizer.encode(chunk, add_special_tokens=False),
            request,
            prompt_token_ids=prompt_token_ids if i == 0 else None,
            finished=i == len(chunks) - 1,
        )
        if delta is not None:
            deltas.append(delta)

    assert "".join(delta.reasoning or "" for delta in deltas) == "Thought"
    assert not any(delta.content for delta in deltas)
    functions = [
        call.function
        for delta in deltas
        for call in delta.tool_calls
        if call.function is not None
    ]
    assert [function.name for function in functions if function.name] == ["get_weather"]
    arguments = "".join(function.arguments or "" for function in functions)
    assert json.loads(arguments) == {"city": "Bern"}
