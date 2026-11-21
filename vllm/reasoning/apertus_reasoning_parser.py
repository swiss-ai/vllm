# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Apertus deliberation and tool handoff."""

from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Supported spellings of the reasoning delimiters.
_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)

TOOLS_PREFIX = "<|tools_prefix|>"
# A turn boundary after the last reasoning start means the prompt is not inside
# a thinking section (e.g. "<|assistant_start|>" with thinking off).
TURN_MARKERS = (
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|user_end|>",
    "<|system_end|>",
    "<|developer_end|>",
)


def _get(message, key):
    return (
        message.get(key) if isinstance(message, dict) else getattr(message, key, None)
    )


def _after_tool(request) -> bool:
    messages = getattr(request, "messages", None) or []
    return bool(messages) and _get(messages[-1], "role") == "tool"


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Parse Apertus deliberation with either tokenizer's delimiter scheme."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.tools_prefix_id = self.vocab.get(TOOLS_PREFIX)
        self.turn_ids = {self.vocab[t] for t in TURN_MARKERS if t in self.vocab}
        # True only when the prompt token ids show an open thinking section.
        self.open_from_prompt = False

    @cached_property
    def _delimiter_pair(self) -> tuple[str, str]:
        # The model emits the lower-ID pair when both aliases are registered.
        inner, think = _CANDIDATE_PAIRS
        vocab = self.vocab

        think_id = vocab.get(think[0])
        if think_id is None or think[1] not in vocab:
            return inner

        inner_id = vocab.get(inner[0])
        # The original sort breaks ID ties by marker text, favoring <think>.
        if inner_id is None or inner[1] not in vocab or think_id <= inner_id:
            return think

        return inner

    @property
    def start_token(self) -> str:
        """Token that starts reasoning (resolved to the emitted id in __init__)."""
        return self._delimiter_pair[0]

    @property
    def end_token(self) -> str:
        """Token that ends reasoning."""
        return self._delimiter_pair[1]

    # ---- prompt state (from token ids, never from the request) ----

    def _last_marker(self, input_ids: Sequence[int]) -> str:
        """'open' (reasoning start), 'closed' (reasoning end) or 'turn' (a turn
        boundary, or no marker at all) -- whichever comes last."""
        for token_id in reversed(input_ids):
            if token_id == self.end_token_id:
                return "closed"
            if token_id == self.start_token_id:
                return "open"
            if token_id in self.turn_ids:
                return "turn"
        return "turn"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        """Reasoning has ended only after a reasoning end token. After a turn
        boundary the reply may still open a thinking section of its own, so the
        stream starts undecided: text is content until a reasoning start appears
        (see ``extract_reasoning_streaming``), and ``<|tools_prefix|>`` hands
        the rest to the tool parser."""
        return self._last_marker(input_ids) == "closed"

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        # Recent vLLM calls this in both modes before parsing.
        self.open_from_prompt = self._last_marker(prompt_token_ids) == "open"

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        # Streaming; called when is_reasoning_end(prompt) is False, which also
        # covers a prompt that ends at a turn boundary.
        self.open_from_prompt = self._last_marker(prompt_token_ids) == "open"

    # ---- non-streaming ----

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        start, end = self.start_token, self.end_token
        opened_here = start in model_output
        if not opened_here and self.open_from_prompt:
            model_output = start + model_output
        # No deliberation block at all (direct tool call or plain answer) -> all
        # content, so the tool parser sees any call and answers are not swallowed.
        if start not in model_output and end not in model_output:
            return None, model_output
        # A reasoning end without a start: the template opened the section and
        # vLLM did not pass the prompt (older vLLM); text before it is reasoning.
        text = (
            model_output.partition(start)[2] if start in model_output else model_output
        )
        if end in text:
            reasoning, _, content = text.partition(end)
            return reasoning or None, content or None
        index = text.find(TOOLS_PREFIX)
        if index >= 0:
            return text[:index] or None, text[index:]
        # After a tool result the model often answers without closing a section
        # the prompt left open; that text is the answer.
        if _after_tool(request) and not opened_here:
            return None, text or None
        return text or None, None

    # ---- streaming ----

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        return self.end_token_id in delta_ids or (
            self.tools_prefix_id is not None and self.tools_prefix_id in delta_ids
        )

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        if self.end_token_id in input_ids:
            return input_ids[input_ids.index(self.end_token_id) + 1 :]
        if self.tools_prefix_id is not None and self.tools_prefix_id in input_ids:
            return input_ids[input_ids.index(self.tools_prefix_id) :]
        return []

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if len(delta_token_ids) == 1 and delta_token_ids[0] in (
            self.start_token_id,
            self.end_token_id,
        ):
            return None
        in_thinking = (
            self.open_from_prompt
            or self.start_token_id in previous_token_ids
            or self.start_token_id in delta_token_ids
        ) and self.end_token_id not in previous_token_ids
        if (
            in_thinking
            and self.end_token_id not in delta_token_ids
            and TOOLS_PREFIX in delta_text
        ):
            head, _, tail = delta_text.partition(TOOLS_PREFIX)
            head = head.replace(self.start_token, "")
            return DeltaMessage(reasoning=head or None, content=TOOLS_PREFIX + tail)
        if (
            self.open_from_prompt
            and self.start_token_id not in previous_token_ids
            and self.start_token_id not in delta_token_ids
        ):
            if self.end_token_id in delta_token_ids:
                head, _, tail = delta_text.partition(self.end_token)
                return DeltaMessage(reasoning=head or None, content=tail or None)
            if self.end_token_id in previous_token_ids:
                return DeltaMessage(content=delta_text)
            return DeltaMessage(reasoning=delta_text)
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )

    def get_streaming_fallback_content(
        self, previous_text: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> str | None:
        """The stream ended inside the reasoning phase: after a tool result, text
        with no reasoning start and no call is the model's answer."""
        if (
            not _after_tool(request)
            or self.start_token in previous_text
            or TOOLS_PREFIX in previous_text
        ):
            return None
        return previous_text.replace(self.end_token, "").strip() or None
