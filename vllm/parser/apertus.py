# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apertus parser for deliberation and tool calls.

Apertus deliberates between `<|inner_prefix|>` and `<|inner_suffix|>` and
calls tools with a JSON array of single-key objects::

    <|inner_prefix|>Check<|tools_prefix|>[{"get_time": {"zone": "CET"}}]<|tools_suffix|>

A tool call ends the deliberation, so the model may call tools from inside it
without emitting `<|inner_suffix|>`. The tool output then continues inside the
open block, so in the prompt only `<|inner_suffix|>` or a turn marker closes it.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import regex as re

from vllm.parser.engine.events import EventType
from vllm.parser.engine.parser_engine import ParserEngine
from vllm.parser.engine.parser_engine_config import (
    ParserEngineConfig,
    ParserState,
    ToolCallBodyCollection,
    Transition,
)
from vllm.parser.utils import extract_json_member_object

if TYPE_CHECKING:
    from vllm.tokenizers import TokenizerLike
    from vllm.tool_parsers.abstract_tool_parser import Tool

INNER_PREFIX, INNER_SUFFIX = "<|inner_prefix|>", "<|inner_suffix|>"
TOOLS_PREFIX, TOOLS_SUFFIX = "<|tools_prefix|>", "<|tools_suffix|>"
TURN_BOUNDARIES = frozenset(
    {
        "<|assistant_start|>",
        "<|assistant_end|>",
        "<|user_end|>",
        "<|system_end|>",
        "<|developer_end|>",
    }
)

# The function name is the key of each ``{"name": {...arguments}}`` element.
_NAME_KEY_RE = re.compile(r'\s*\{\s*"((?:[^"\\]|\\.)*)"\s*:')


def _raw_name_key(raw: str) -> str | None:
    match = _NAME_KEY_RE.match(raw)
    return match.group(1) if match else None


def _apertus_arg_converter(raw_args: str, partial: bool) -> str:
    key = _raw_name_key(raw_args)
    span = None if key is None else extract_json_member_object(raw_args, key)
    if span is None:
        return "" if partial else "{}"
    return span


@functools.cache
def apertus_config() -> ParserEngineConfig:
    return ParserEngineConfig(
        name="apertus",
        initial_state=ParserState.CONTENT,
        wait_for_reasoning=True,
        turn_boundary_tokens=TURN_BOUNDARIES,
        terminals={
            "THINK_START": INNER_PREFIX,
            "THINK_END": INNER_SUFFIX,
            "TOOL_START": TOOLS_PREFIX,
            "TOOL_END": TOOLS_SUFFIX,
        },
        token_id_terminals={
            "THINK_START": INNER_PREFIX,
            "THINK_END": INNER_SUFFIX,
            "TOOL_START": TOOLS_PREFIX,
            "TOOL_END": TOOLS_SUFFIX,
        },
        transitions={
            (ParserState.CONTENT, "THINK_START"): Transition(
                ParserState.REASONING,
                (EventType.REASONING_START,),
            ),
            (ParserState.REASONING, "THINK_START"): Transition(
                ParserState.REASONING,
                (),
            ),
            (ParserState.REASONING, "THINK_END"): Transition(
                ParserState.CONTENT,
                (EventType.REASONING_END,),
            ),
            # The array splitter emits one TOOL_CALL_START per element, and
            # its closing "]" returns to CONTENT before <|tools_suffix|>.
            (ParserState.REASONING, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.REASONING_END,),
            ),
            (ParserState.CONTENT, "TOOL_START"): Transition(
                ParserState.TOOL_ARGS,
                (EventType.REASONING_END,),
            ),
            (ParserState.TOOL_ARGS, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.CONTENT, "TOOL_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
            (ParserState.CONTENT, "THINK_END"): Transition(
                ParserState.CONTENT,
                (),
            ),
        },
        arg_converter=_apertus_arg_converter,
        tool_call_body_collection=ToolCallBodyCollection.JSON_ARRAY,
        stream_arg_deltas=True,
    )


class ApertusParser(ParserEngine):
    CONFIG_NAME = "apertus"

    def __init__(
        self,
        tokenizer: TokenizerLike,
        tools: list[Tool] | None = None,
        **kwargs,
    ) -> None:
        self._prompt_opens_reasoning = False
        kwargs.setdefault("parser_engine_config", apertus_config())
        super().__init__(tokenizer, tools, **kwargs)

    def _last_marker(
        self, token_ids: Sequence[int]
    ) -> Literal["open", "closed"] | None:
        for token_id in reversed(token_ids):
            if token_id == self._reasoning_end_token_id:
                return "closed"
            if token_id == self._reasoning_start_token_id:
                return "open"
            if token_id in self._turn_boundary_token_ids:
                return None
        return None

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # Unlike the engine default, a tool call does not close the block here:
        # its output continues inside the deliberation.
        return self._last_marker(input_ids) == "closed"

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        super().set_prompt_token_ids(prompt_token_ids)
        self._prompt_opens_reasoning = self._last_marker(prompt_token_ids) == "open"

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self.set_prompt_token_ids(prompt_token_ids)
        self._reset()

    def _reset(self, initial_state: ParserState | None = None) -> None:
        if initial_state is None and self._prompt_opens_reasoning:
            initial_state = ParserState.REASONING
        super()._reset(initial_state=initial_state)

    def _try_extract_name(self, idx: int) -> str | None:
        return _raw_name_key(self._tool_slots[idx].args) or None
