# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reasoning parser for Apertus models.

Apertus wraps its thinking between a start/end pair of special tokens. The
canonical pair is ``<|inner_prefix|>``/``<|inner_suffix|>``, but some tokenizer
builds register ``<think>``/``</think>`` at the emitted ids instead. The parser
selects whichever pair the loaded tokenizer exposes at the lower start-token id.
"""

from collections.abc import Iterable, Sequence
from functools import cached_property
from typing import TYPE_CHECKING

from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest

# Candidate (start, end) delimiter pairs, in fallback order.
_CANDIDATE_PAIRS = (
    ("<|inner_prefix|>", "<|inner_suffix|>"),
    ("<think>", "</think>"),
)

# Opening a new assistant turn closes any thinking block left open earlier.
_TURN_START_TOKEN = "<|assistant_start|>"


class ApertusReasoningParser(BaseThinkingReasoningParser):
    """Reasoning parser for the Apertus thinking block."""

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._turn_start_token_id = self.vocab.get(_TURN_START_TOKEN)
        self._prompt_reasoning_open = False

    @cached_property
    def _pair(self) -> tuple[str, str]:
        vocab = self.vocab
        present = sorted(
            (vocab[start], start, end)
            for start, end in _CANDIDATE_PAIRS
            if start in vocab and end in vocab
        )
        return (present[0][1], present[0][2]) if present else _CANDIDATE_PAIRS[0]

    @property
    def start_token(self) -> str:
        return self._pair[0]

    @property
    def end_token(self) -> str:
        return self._pair[1]

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        # With no thinking block at all (direct tool call or plain answer),
        # the base class would label the whole output as reasoning, hiding tool
        # calls from the tool parser. Return it as content instead.
        if self.start_token not in model_output and self.end_token not in model_output:
            return None, model_output
        return super().extract_reasoning(model_output, request)

    def _reasoning_open(self, token_ids: Sequence[int], default: bool = False) -> bool:
        """Whether a thinking block is open at the end of ``token_ids``.

        The scan stops at an assistant turn boundary: the chat template keeps
        the block open across tool calls, but a block left unclosed by an
        earlier turn does not carry into a fresh one.

        Args:
            token_ids: Tokens to scan, oldest first.
            default: Result when the scan reaches the start of ``token_ids``
                without seeing a delimiter or a turn boundary.
        """
        for i in range(len(token_ids) - 1, -1, -1):
            token_id = token_ids[i]
            if token_id == self.start_token_id:
                return True
            if token_id in (self.end_token_id, self._turn_start_token_id):
                return False
        return default

    def adjust_initial_state_from_prompt(self, prompt_token_ids: Sequence[int]) -> None:
        # Callers of the streaming hooks below pass generated tokens only, so
        # a turn resumed inside a thinking block has to be remembered here.
        self._prompt_reasoning_open = self._reasoning_open(prompt_token_ids)

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        # The base class only flips the phase once the end token appears.
        # If the start token never appeared either, this is a direct tool
        # call with no thinking block, so treat reasoning as already over.
        if not self._reasoning_open(input_ids, default=self._prompt_reasoning_open):
            return True
        return super().is_reasoning_end_streaming(input_ids, delta_ids)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> "DeltaMessage | None":
        # The base class keys off the start token to know it is inside a
        # thinking block. When the prompt opened the block, that token is not
        # among the generated ids, so stand in for it.
        if (
            self._prompt_reasoning_open
            and self.start_token_id not in previous_token_ids
        ):
            previous_token_ids = [self.start_token_id, *previous_token_ids]
        return super().extract_reasoning_streaming(
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        )
