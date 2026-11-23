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

_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)

TOOLS_PREFIX = "<|tools_prefix|>"
TURN_MARKERS = (
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|user_end|>",
    "<|system_end|>",
    "<|developer_end|>",
)


def _after_tool(request: "ChatCompletionRequest | ResponsesRequest") -> bool:
    messages = getattr(request, "messages", None)
    if not messages:
        return False
    last_message = messages[-1]
    role = (
        last_message.get("role")
        if isinstance(last_message, dict)
        else getattr(last_message, "role", None)
    )
    return role == "tool"


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Parse Apertus deliberation with either tokenizer's delimiter scheme."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self.tools_prefix_id = self.vocab.get(TOOLS_PREFIX)
        self.turn_ids = {self.vocab[t] for t in TURN_MARKERS if t in self.vocab}
        self.open_from_prompt = False

    @cached_property
    def _delimiter_pair(self) -> tuple[str, str]:
        vocab = self.vocab
        pairs = (
            (start, end)
            for start, end in _CANDIDATE_PAIRS
            if start in vocab and end in vocab
        )
        # Match the tokenizer plugin's ordering, including ties.
        return min(
            pairs,
            key=lambda pair: (vocab[pair[0]], pair),
            default=_CANDIDATE_PAIRS[0],
        )

    @property
    def start_token(self) -> str:
        return self._delimiter_pair[0]

    @property
    def end_token(self) -> str:
        return self._delimiter_pair[1]

    def _last_marker(self, input_ids: Sequence[int]) -> str:
        for token_id in reversed(input_ids):
            if token_id == self.end_token_id:
                return "closed"
            if token_id == self.start_token_id:
                return "open"
            if token_id in self.turn_ids:
                return "turn"
        return "turn"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        # A new assistant turn can emit its own reasoning start.
        return self._last_marker(input_ids) == "closed"

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        self.open_from_prompt = self._last_marker(prompt_token_ids) == "open"

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        self.set_prompt_token_ids(prompt_token_ids)

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        before_start, start, after_start = model_output.partition(self.start_token)
        text = after_start if start else before_start
        reasoning, end, content = text.partition(self.end_token)
        if end:
            return reasoning or None, content or None
        if not start and not self.open_from_prompt:
            return None, model_output

        index = text.find(TOOLS_PREFIX)
        if index >= 0:
            return text[:index] or None, text[index:]
        # After a tool result the model often answers without closing a section
        # the prompt left open; that text is the answer.
        if not start and _after_tool(request):
            return None, text or None
        return text or None, None

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
        if (
            TOOLS_PREFIX in delta_text
            and self.end_token_id not in delta_token_ids
            and self.end_token_id not in previous_token_ids
            and (
                self.open_from_prompt
                or self.start_token_id in previous_token_ids
                or self.start_token_id in delta_token_ids
            )
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
