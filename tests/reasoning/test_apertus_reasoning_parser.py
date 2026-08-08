# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.reasoning.utils import run_reasoning_extraction
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import Parser, ParserManager
from vllm.reasoning import ReasoningParser, ReasoningParserManager

parser_name = "apertus"

INNER_VOCAB = {
    "<|assistant_start|>": 5,
    "<|inner_prefix|>": 32,
    "<|inner_suffix|>": 33,
    "<|tools_prefix|>": 40,
    "<|tools_suffix|>": 41,
    "<think>": 500,
    "</think>": 501,
}
THINK_VOCAB = {
    "<think>": 32,
    "</think>": 33,
    "<|inner_prefix|>": 900,
    "<|inner_suffix|>": 901,
}


class MockTokenizer:
    def __init__(self, vocab: dict[str, int]):
        self._vocab = vocab

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def tokenize(self, text: str) -> list[str]:
        # Only special tokens matter to the parser; deltas are fed one per token.
        return [text] if text in self._vocab else []


def make_parser(vocab: dict[str, int]) -> ReasoningParser:
    parser_cls = ReasoningParserManager.get_reasoning_parser(parser_name)
    return parser_cls(MockTokenizer(vocab))


# Stand-in id for any run of ordinary text between two special tokens.
PLAIN_TEXT_ID = 1000
# Any token the reasoning parser has no opinion about.
OTHER_ID = 99
TURN_START_ID = INNER_VOCAB["<|assistant_start|>"]

TOOL_CALL = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}]<|tools_suffix|>'
DELIBERATED_TOOL_CALL = "<|inner_prefix|>Check the weather.<|inner_suffix|>" + TOOL_CALL

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
    },
}


def make_delegating_parser(vocab: dict[str, int]) -> Parser:
    """The reasoning and tool parsers as the server composes them."""
    parser_cls = ParserManager.get_parser(
        tool_parser_name=parser_name,
        reasoning_parser_name=parser_name,
        enable_auto_tools=True,
    )
    return parser_cls(MockTokenizer(vocab), [WEATHER_TOOL])


def split_deltas(text: str) -> list[tuple[str, list[int]]]:
    """Split model output into deltas, each special token on its own."""
    specials = sorted(INNER_VOCAB, key=len, reverse=True)
    deltas: list[tuple[str, list[int]]] = []
    buffer = ""
    index = 0
    while index < len(text):
        special = next((s for s in specials if text.startswith(s, index)), None)
        if special is None:
            buffer += text[index]
            index += 1
            continue
        if buffer:
            deltas.append((buffer, [PLAIN_TEXT_ID]))
            buffer = ""
        deltas.append((special, [INNER_VOCAB[special]]))
        index += len(special)
    if buffer:
        deltas.append((buffer, [PLAIN_TEXT_ID]))
    return deltas


def run_stream(
    parser: Parser, model_output: str, prompt_token_ids: list[int]
) -> tuple[str, str, list[str]]:
    """Stream ``model_output`` and return (reasoning, content, tool names)."""
    request = ChatCompletionRequest(
        model="apertus",
        messages=[{"role": "user", "content": "weather in Bern?"}],
        tools=[WEATHER_TOOL],
        tool_choice="auto",
        stream=True,
    )
    deltas = split_deltas(model_output)
    reasoning, content, tool_names = "", "", []
    for i, (delta_text, delta_token_ids) in enumerate(deltas):
        message = parser.parse_delta(
            delta_text,
            delta_token_ids,
            request,
            prompt_token_ids=prompt_token_ids,
            finished=i == len(deltas) - 1,
        )
        if message is None:
            continue
        reasoning += message.reasoning or ""
        content += message.content or ""
        tool_names += [
            call.function.name
            for call in message.tool_calls or []
            if call.function is not None and call.function.name
        ]
    return reasoning, content, tool_names


@pytest.mark.parametrize(
    "vocab, start, end",
    [
        (INNER_VOCAB, "<|inner_prefix|>", "<|inner_suffix|>"),
        (THINK_VOCAB, "<think>", "</think>"),
    ],
)
def test_delimiters_follow_the_tokenizer(vocab, start, end):
    """The emitted (lower-id) delimiter pair wins, whichever scheme the
    tokenizer build registers there."""
    parser = make_parser(vocab)

    assert (parser.start_token, parser.end_token) == (start, end)
    assert (parser.start_token_id, parser.end_token_id) == (32, 33)


@pytest.mark.parametrize("streaming", [True, False])
def test_deliberation_block_is_reasoning(streaming: bool):
    parser = make_parser(INNER_VOCAB)
    output = ["<|inner_prefix|>", "Let me think", "<|inner_suffix|>", "The answer"]

    reasoning, content = run_reasoning_extraction(
        reasoning_parser=parser, model_output=output, streaming=streaming
    )

    assert reasoning == "Let me think"
    assert content == "The answer"


def test_output_without_deliberation_is_all_content():
    """A direct tool call carries no inner block; it must reach the tool parser
    as content rather than being swallowed as reasoning."""
    parser = make_parser(INNER_VOCAB)
    output = '<|tools_prefix|>[{"get_weather": {"city": "Bern"}}]<|tools_suffix|>'

    reasoning, content = run_reasoning_extraction(
        reasoning_parser=parser, model_output=[output]
    )

    assert reasoning is None
    assert content == output


def test_direct_tool_call_streaming_leaves_reasoning_phase():
    """The same case as above, but through the streaming path: the phase
    must flip as soon as it is clear no inner block is coming."""
    parser = make_parser(INNER_VOCAB)
    tools_prefix_id = 99  # not in vocab, irrelevant to the reasoning parser

    assert parser.is_reasoning_end_streaming(
        input_ids=[tools_prefix_id], delta_ids=[tools_prefix_id]
    )


def test_open_deliberation_streaming_keeps_reasoning_phase():
    """The flip above must not fire while a block is in flight, or the tool
    parser would be handed the deliberation."""
    parser = make_parser(INNER_VOCAB)
    start_id, end_id = parser.start_token_id, parser.end_token_id

    assert not parser.is_reasoning_end_streaming([start_id, OTHER_ID], [OTHER_ID])
    assert parser.is_reasoning_end_streaming([start_id, OTHER_ID, end_id], [end_id])


def test_prompt_resuming_deliberation_keeps_reasoning_phase():
    """The template holds the block open across tool calls, so a resumed turn
    keeps deliberating even though the start token is only in the prompt."""
    parser = make_parser(INNER_VOCAB)
    end_id = parser.end_token_id
    parser.adjust_initial_state_from_prompt(
        [TURN_START_ID, OTHER_ID, parser.start_token_id, OTHER_ID]
    )

    assert not parser.is_reasoning_end_streaming([OTHER_ID], [OTHER_ID])
    assert parser.is_reasoning_end_streaming([OTHER_ID, end_id], [end_id])


def test_unclosed_deliberation_in_history_does_not_reopen_reasoning():
    """A block left unclosed by an earlier turn must not make the new turn
    wait for an end token that will never come."""
    parser = make_parser(INNER_VOCAB)
    parser.adjust_initial_state_from_prompt(
        [parser.start_token_id, OTHER_ID, TURN_START_ID]
    )

    assert parser.is_reasoning_end_streaming([OTHER_ID], [OTHER_ID])


@pytest.mark.parametrize(
    "model_output, expected_reasoning",
    [
        (TOOL_CALL, ""),
        (DELIBERATED_TOOL_CALL, "Check the weather."),
    ],
    ids=["without_deliberation", "with_deliberation"],
)
def test_streamed_tool_call_reaches_the_tool_parser(model_output, expected_reasoning):
    """Regression: without a deliberation block the reasoning phase never
    ended, so the tool parser never ran and the raw tool call was streamed to
    the user as content."""
    parser = make_delegating_parser(INNER_VOCAB)

    reasoning, content, tool_names = run_stream(
        parser, model_output, [OTHER_ID, TURN_START_ID]
    )

    assert tool_names == ["get_weather"]
    assert content == ""
    assert reasoning == expected_reasoning


def test_streamed_resumed_deliberation_is_reasoning():
    """A turn resumed inside a block deliberates until the end token, and the
    marker itself never reaches the user."""
    parser = make_delegating_parser(INNER_VOCAB)
    prompt_token_ids = [TURN_START_ID, INNER_VOCAB["<|inner_prefix|>"], OTHER_ID]

    reasoning, content, tool_names = run_stream(
        parser, "Still checking.<|inner_suffix|>It is sunny.", prompt_token_ids
    )

    assert reasoning == "Still checking."
    assert content == "It is sunny."
    assert tool_names == []
