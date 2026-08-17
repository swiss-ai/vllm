# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numeric parity of Apertus2 sequence-parallel MoE against a single GPU.

TP=2 x DP=2 with expert parallelism is the smallest topology that activates
``ParallelConfig.use_sequence_parallel_moe`` (allgather backend, TP>1, DP>1),
exercising the chunk/all-gather/trim path and the replicated shared experts.
Greedy top-k logprobs must match the TP=1 reference within kernel noise.

Requires 4 GPUs and a local Apertus2 MoE checkpoint
(``APERTUS2_SP_PARITY_MODEL`` overrides the default path); skips otherwise.
"""

import asyncio
import os

import pytest
import torch

from tests.models.utils import check_logprobs_close
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

DEFAULT_MODEL = (
    "/iopsstor/scratch/cscs/mvasilev/hf-export/"
    "chonk-3b-parameter-collapse-muon-wd01-split-fc1-iter_0015456"
)
MODEL = os.environ.get("APERTUS2_SP_PARITY_MODEL", DEFAULT_MODEL)

# Odd lengths on purpose: sequence_parallel_chunk pads to a multiple of
# tp_size, and the exit trim back to the pre-chunk token count is exactly
# the step that a wrong implementation gets away with on even batches.
PROMPT_LENGTHS = [1, 7, 13, 29, 63, 101, 257, 511]
MAX_TOKENS = 32
NUM_LOGPROBS = 5

pytestmark = [
    pytest.mark.distributed(num_gpus=4),
    pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires 4 GPUs"),
    pytest.mark.skipif(
        not os.path.isfile(os.path.join(MODEL, "config.json")),
        reason=f"Apertus2 checkpoint not found: {MODEL}",
    ),
]


def _token_prompts() -> list[TokensPrompt]:
    generator = torch.Generator().manual_seed(1234)
    return [
        TokensPrompt(
            prompt_token_ids=torch.randint(
                1000, 30000, (length,), generator=generator
            ).tolist()
        )
        for length in PROMPT_LENGTHS
    ]


async def _greedy_logprobs(
    engine_args: AsyncEngineArgs, expect_sequence_parallel: bool
) -> list[tuple[list[int], str, list[dict]]]:
    engine = AsyncLLM.from_engine_args(engine_args)
    try:
        parallel_config = engine.vllm_config.parallel_config
        assert parallel_config.use_sequence_parallel_moe is expect_sequence_parallel

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=MAX_TOKENS,
            logprobs=NUM_LOGPROBS,
            ignore_eos=True,
            output_kind=RequestOutputKind.FINAL_ONLY,
        )

        async def one(idx: int, prompt: TokensPrompt):
            final = None
            async for out in engine.generate(
                request_id=f"parity-{idx}",
                prompt=prompt,
                sampling_params=sampling_params,
            ):
                final = out
            completion = final.outputs[0]
            return (
                list(completion.token_ids),
                completion.text,
                completion.logprobs,
            )

        return await asyncio.gather(
            *(one(idx, prompt) for idx, prompt in enumerate(_token_prompts()))
        )
    finally:
        engine.shutdown()


def test_sequence_parallel_moe_matches_single_gpu() -> None:
    common = dict(
        model=MODEL,
        trust_remote_code=True,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        seed=0,
    )
    reference_args = AsyncEngineArgs(tensor_parallel_size=1, **common)
    sp_args = AsyncEngineArgs(
        tensor_parallel_size=2,
        data_parallel_size=2,
        data_parallel_backend="mp",
        enable_expert_parallel=True,
        **common,
    )

    reference = asyncio.run(
        _greedy_logprobs(reference_args, expect_sequence_parallel=False)
    )
    sequence_parallel = asyncio.run(
        _greedy_logprobs(sp_args, expect_sequence_parallel=True)
    )

    check_logprobs_close(
        outputs_0_lst=reference,
        outputs_1_lst=sequence_parallel,
        name_0="tp1",
        name_1="tp2_dp2_ep_sp",
    )
