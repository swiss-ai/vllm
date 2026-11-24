# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Apertus deliberation and tool handoff."""

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Literal

from vllm.entrypoints.generate.base.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

INNER_PREFIX, INNER_SUFFIX = "<|inner_prefix|>", "<|inner_suffix|>"
TOOLS_PREFIX = "<|tools_prefix|>"
TURN_MARKERS = (
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|user_end|>",
    "<|system_end|>",
    "<|developer_end|>",
)


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Parse Apertus deliberation and its handoff to tool calls."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs) -> None:
        super().__init__(tokenizer, *args, **kwargs)
        self.tools_prefix_id = self.vocab.get(TOOLS_PREFIX)
        self.turn_ids = {self.vocab[t] for t in TURN_MARKERS if t in self.vocab}
        self.open_from_prompt = False

    @property
    def start_token(self) -> str:
        return INNER_PREFIX

    @property
    def end_token(self) -> str:
        return INNER_SUFFIX

    def _get_last_marker(
        self, input_ids: Sequence[int]
    ) -> Literal["open", "closed", "turn"]:
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
        return self._get_last_marker(input_ids) == "closed"

    def set_prompt_token_ids(self, prompt_token_ids: Sequence[int]) -> None:
        self.open_from_prompt = self._get_last_marker(prompt_token_ids) == "open"

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
