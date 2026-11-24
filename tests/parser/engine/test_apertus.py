# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the engine-based Apertus parser, through the serving entry point.

Every case runs as a full response and streamed in several chunk sizes, so the
two serving modes must agree.
"""

import json

import pytest
import regex as re

from tests.parser.engine.conftest import make_mock_tokenizer
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import ParserManager
from vllm.parser.apertus import (
    INNER_PREFIX,
    INNER_SUFFIX,
    TOOLS_PREFIX,
    TOOLS_SUFFIX,
    ApertusParser,
)

ASSISTANT_START = "<|assistant_start|>"
OUTPUT_START, OUTPUT_END = "<|tool_output_start|>", "<|tool_output_end|>"
VOCAB = {
    token: 1000 + i
    for i, token in enumerate(
        (
            INNER_PREFIX,
            INNER_SUFFIX,
            TOOLS_PREFIX,
            TOOLS_SUFFIX,
            ASSISTANT_START,
            OUTPUT_START,
            OUTPUT_END,
        )
    )
}
_SPECIAL_RE = re.compile("(" + "|".join(map(re.escape, VOCAB)) + ")")

TOOLS = [
    {"type": "function", "function": {"name": name}}
    for name in ("get_weather", "get_time", "calculate")
]


def _call(*elements: str) -> str:
    return f"{TOOLS_PREFIX}[{', '.join(elements)}]{TOOLS_SUFFIX}"


BERN = '{"get_weather": {"city": "Bern"}}'
BERN_CALL = ("get_weather", {"city": "Bern"})


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Special tokens are one token each; other text is one token per char."""
    tokens: list[tuple[int, str]] = []
    for part in _SPECIAL_RE.split(text):
        if part in VOCAB:
            tokens.append((VOCAB[part], part))
        else:
            tokens.extend((ord(char), char) for char in part)
    return tokens


def _run(prompt: str, output: str, mode: str, tools: bool = True):
    """Parse ``output`` after ``prompt``.

    ``mode`` is ``full`` or ``stream-<chunk size>``, through the parser the
    serving layer composes; an ``engine-`` prefix uses the engine directly.
    """
    request = ChatCompletionRequest(
        model="test-model",
        messages=[],
        **({"tools": TOOLS, "tool_choice": "auto"} if tools else {}),
    )
    tokenizer = make_mock_tokenizer(VOCAB)
    if mode.startswith("engine-"):
        mode = mode.removeprefix("engine-")
        parser = ApertusParser(tokenizer, request.tools)
    else:
        parser_cls = ParserManager.get_parser(
            reasoning_parser_name="apertus",
            tool_parser_name="apertus" if tools else None,
            enable_auto_tools=tools,
        )
        assert parser_cls is not None
        parser = parser_cls(tokenizer, tools=request.tools)
    prompt_ids = [token_id for token_id, _ in _tokenize(prompt)]
    tokens = _tokenize(output)

    if mode == "full":
        parser.set_prompt_token_ids(prompt_ids)
        reasoning, content, functions = parser.parse(
            output,
            request,
            enable_auto_tools=tools,
            model_output_token_ids=[token_id for token_id, _ in tokens],
        )
        calls = [(f.name, json.loads(f.arguments)) for f in functions or []]
        return reasoning, content.strip() if content else None, calls

    chunk_size = len(tokens) if mode == "stream-all" else int(mode.split("-")[1])
    batches = [tokens[i : i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    reasoning, content = "", ""
    streamed: dict[int, list[str]] = {}
    for i, batch in enumerate(batches):
        delta = parser.parse_delta(
            "".join(text for _, text in batch),
            [token_id for token_id, _ in batch],
            request,
            prompt_token_ids=prompt_ids if i == 0 else None,
            finished=i == len(batches) - 1,
        )
        if delta is None:
            continue
        reasoning += delta.reasoning or ""
        content += delta.content or ""
        for tool_call in delta.tool_calls:
            name_args = streamed.setdefault(tool_call.index, ["", ""])
            if tool_call.function is not None:
                name_args[0] += tool_call.function.name or ""
                name_args[1] += tool_call.function.arguments or ""
    calls = [(name, json.loads(args)) for name, args in streamed.values()]
    return reasoning or None, content.strip() or None, calls


SERVING_MODES = ["full", "stream-1", "stream-4", "stream-all"]
MODES = [*SERVING_MODES, "engine-full", "engine-stream-1"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "prompt, output, expected",
    [
        pytest.param(ASSISTANT_START, "Answer", (None, "Answer", []), id="answer"),
        pytest.param(
            ASSISTANT_START,
            f"{INNER_PREFIX}Thought{INNER_SUFFIX}Answer",
            ("Thought", "Answer", []),
            id="thought-answer",
        ),
        pytest.param(
            ASSISTANT_START + INNER_PREFIX,
            f"Thought{INNER_SUFFIX}Answer",
            ("Thought", "Answer", []),
            id="prompt-open",
        ),
        pytest.param(
            ASSISTANT_START + INNER_PREFIX,
            "Thought",
            ("Thought", None, []),
            id="prompt-open-unclosed",
        ),
        pytest.param(
            INNER_PREFIX + ASSISTANT_START,
            "Answer",
            (None, "Answer", []),
            id="new-turn-after-open-block",
        ),
        # Streaming cannot reclassify text already sent, so a stray end marker
        # is dropped and both modes return content.
        pytest.param(
            ASSISTANT_START,
            f"Thought{INNER_SUFFIX}Answer",
            (None, "ThoughtAnswer", []),
            id="stray-inner-suffix",
        ),
        pytest.param(
            ASSISTANT_START, _call(BERN), (None, None, [BERN_CALL]), id="tool-call"
        ),
        pytest.param(
            ASSISTANT_START,
            f"{INNER_PREFIX}Thought{_call(BERN)}",
            ("Thought", None, [BERN_CALL]),
            id="tool-call-from-thought",
        ),
        pytest.param(
            ASSISTANT_START + INNER_PREFIX,
            f"Thought{_call(BERN)}",
            ("Thought", None, [BERN_CALL]),
            id="tool-call-from-prompt-open-thought",
        ),
        pytest.param(
            ASSISTANT_START + INNER_PREFIX + f"Need the weather.{_call(BERN)}"
            f'{OUTPUT_START}{{"temp": 18}}{OUTPUT_END}',
            f"It is warm.{INNER_SUFFIX}It is 18 degrees.",
            ("It is warm.", "It is 18 degrees.", []),
            id="thought-continues-after-tool-output",
        ),
        pytest.param(
            ASSISTANT_START,
            f"I will check. {_call(BERN)}",
            (None, "I will check.", [BERN_CALL]),
            id="text-before-tool-call",
        ),
        pytest.param(
            ASSISTANT_START,
            _call(BERN, '{"get_time": {"zone": "CET"}}'),
            (None, None, [BERN_CALL, ("get_time", {"zone": "CET"})]),
            id="two-tool-calls",
        ),
        pytest.param(
            ASSISTANT_START,
            _call('{"get_weather": {"nested": {"inner": "value"}, "list": ["a"]}}'),
            (
                None,
                None,
                [("get_weather", {"nested": {"inner": "value"}, "list": ["a"]})],
            ),
            id="nested-arguments",
        ),
        pytest.param(
            ASSISTANT_START,
            _call('{"calculate": {"values": [1.5e3, -2e-3]}}'),
            (None, None, [("calculate", {"values": [1.5e3, -2e-3]})]),
            id="scientific-notation",
        ),
        pytest.param(
            ASSISTANT_START,
            _call(BERN).removesuffix(TOOLS_SUFFIX),
            (None, None, [BERN_CALL]),
            id="missing-tools-suffix",
        ),
        pytest.param(
            ASSISTANT_START,
            f'{TOOLS_PREFIX}[{{"get_weather": {{"city": "Bern"}}',
            (None, None, [BERN_CALL]),
            id="truncated-tool-call",
        ),
    ],
)
def test_apertus_parsing(prompt, output, expected, mode):
    assert _run(prompt, output, mode) == expected


@pytest.mark.parametrize("mode", SERVING_MODES)
def test_reasoning_without_tool_parser(mode):
    """``--reasoning-parser apertus`` alone must still split the model's own
    deliberation block after a fresh turn."""
    output = f"{INNER_PREFIX}Thought{INNER_SUFFIX}Answer"
    assert _run(ASSISTANT_START, output, mode, tools=False) == (
        "Thought",
        "Answer",
        [],
    )
