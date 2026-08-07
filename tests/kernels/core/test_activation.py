# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import pytest
import torch

from tests.kernels.allclose_default import get_default_atol, get_default_rtol
from tests.kernels.utils import opcheck
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    ReLUSquaredActivation,
    SiluAndMul,
    SiluAndMulWithClamp,
    SSSGLUAndMul,
    SwigluOAIAndMul,
    SwigluStepAndMul,
    sssglu_and_mul_triton,
    swiglustep_and_mul_triton,
)
from vllm.utils.torch_utils import set_random_seed

DTYPES = [torch.half, torch.bfloat16, torch.float]
NUM_TOKENS = [7, 83, 2048]  # Arbitrary values for testing
D = [512, 13824]  # Arbitrary values for testing
SEEDS = [0]
CUDA_DEVICES = [
    f"cuda:{i}" for i in range(1 if torch.accelerator.device_count() == 1 else 2)
]


@pytest.mark.parametrize(
    "activation",
    [
        "silu_and_mul",
        "mul_and_silu",
        "gelu",
        "gelu_tanh",
        "fatrelu",
        "swigluoai_and_mul",
        "swiglustep_and_mul",
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_act_and_mul(
    default_vllm_config,
    activation: str,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, 2 * d, dtype=dtype)
    if activation == "silu_and_mul":
        layer = SiluAndMul(compile_native=False)
        fn = torch.ops._C.silu_and_mul
    if activation == "mul_and_silu":
        layer = MulAndSilu()
        fn = torch.ops._C.mul_and_silu
    elif activation == "gelu":
        layer = GeluAndMul(approximate="none")
        fn = torch.ops._C.gelu_and_mul
    elif activation == "gelu_tanh":
        layer = GeluAndMul(approximate="tanh")
        fn = torch.ops._C.gelu_tanh_and_mul
    elif activation == "fatrelu":
        threshold = random.uniform(0, 1)
        layer = FatreluAndMul(threshold)
        fn = torch.ops._C.fatrelu_and_mul
    elif activation == "swigluoai_and_mul":
        layer = SwigluOAIAndMul()
        fn = torch.ops._C.swigluoai_and_mul
    elif activation == "swiglustep_and_mul":
        layer = SwigluStepAndMul()
        fn = swiglustep_and_mul_triton
    out = layer(x)
    ref_out = layer.forward_native(x)
    if activation in ["swigluoai_and_mul", "swiglustep_and_mul"]:
        rtol = {
            # For fp16, change the relative tolerance from 1e-3 to 2e-3
            torch.float16: 2e-3,
            torch.bfloat16: 2e-2,
            torch.float: 1.3e-6,
        }

        def _get_rtol(output) -> float:
            return rtol[output.dtype]

        torch.testing.assert_close(
            out, ref_out, atol=get_default_atol(out), rtol=_get_rtol(out)
        )
    else:
        # The SiluAndMul, MulAndSilu, GELU and FatReLU implementations are
        # equivalent to the native PyTorch implementations, so we can do exact
        # comparison.
        torch.testing.assert_close(out, ref_out, atol=0.0, rtol=0.0)

    d = x.shape[-1] // 2
    output_shape = x.shape[:-1] + (d,)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    if activation == "fatrelu":
        opcheck(fn, (out, x, threshold))
    elif activation == "swigluoai_and_mul":
        opcheck(fn, (out, x, layer.alpha, layer.limit))
    elif activation != "swiglustep_and_mul":
        opcheck(fn, (out, x))


def _sssglu_and_mul_reference(x: torch.Tensor) -> torch.Tensor:
    gate, up = x.chunk(2, dim=-1)
    gate = gate.float()
    up = up.float()
    shifted_gate = gate - 1.0
    output = (0.5 + shifted_gate / (1.0 + shifted_gate.abs())) * up
    return output.to(x.dtype)


SSSGLU_AND_MUL_IMPLEMENTATIONS = ["native", "triton_wrapper", "custom_op"]


def _run_sssglu_and_mul(
    x: torch.Tensor,
    implementation: str,
    default_vllm_config,
) -> torch.Tensor:
    if implementation == "native":
        return SSSGLUAndMul().forward_native(x)

    if implementation == "triton_wrapper":
        d = x.shape[-1] // 2
        output = torch.empty(
            x.shape[:-1] + (d,),
            dtype=x.dtype,
            device=x.device,
        )
        sssglu_and_mul_triton(output, x)
        return output

    assert implementation == "custom_op"
    default_vllm_config.compilation_config.custom_ops = ["all"]
    return SSSGLUAndMul()(x)


@pytest.mark.parametrize("implementation", SSSGLU_AND_MUL_IMPLEMENTATIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "num_tokens,d",
    [
        (1, 7),
        (7, 576),
        (3, 1024),
        (2, 2048),
    ],
)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul(
    default_vllm_config,
    implementation: str,
    dtype: torch.dtype,
    num_tokens: int,
    d: int,
    device: str,
) -> None:
    set_random_seed(0)
    x = torch.randn((num_tokens, 2 * d), dtype=dtype, device=device)

    anchors = torch.tensor(
        [-1000.0, 0.0, 0.96875, 1.0, 1.03125, 2.0, 1000.0],
        dtype=dtype,
        device=device,
    )
    num_anchors = min(d, anchors.numel())
    x[0, :num_anchors] = anchors[:num_anchors]
    x[0, d : d + num_anchors] = torch.arange(
        1,
        num_anchors + 1,
        dtype=dtype,
        device=device,
    )

    expected = _sssglu_and_mul_reference(x)
    output = _run_sssglu_and_mul(x, implementation, default_vllm_config)

    if dtype == torch.bfloat16:
        atol, rtol = 2.0e-2, 2.0e-2
    else:
        atol, rtol = 1.0e-5, 1.0e-4

    assert output.shape == expected.shape
    assert output.dtype == dtype
    torch.testing.assert_close(output, expected, atol=atol, rtol=rtol)


@pytest.mark.parametrize("implementation", ["triton_wrapper", "custom_op"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul_bf16_cast_boundary(
    default_vllm_config,
    implementation: str,
    device: str,
) -> None:
    gate = torch.tensor(
        [-1.0, -0.0009765625, 0.5, 0.99609375, 1.5, 1.9921875, 3.0],
        dtype=torch.bfloat16,
        device=device,
    )
    up = torch.tensor(
        [-8.0, -8.0, -8.0, -3.0, -8.0, -8.0, -8.0],
        dtype=torch.bfloat16,
        device=device,
    )
    x = torch.cat((gate, up)).unsqueeze(0)

    expected = _sssglu_and_mul_reference(x)
    shifted_gate = gate - 1.0
    premature_cast = (
        0.5 + shifted_gate / (1.0 + shifted_gate.abs())
    ) * up
    assert not torch.equal(premature_cast, expected[0])

    output = _run_sssglu_and_mul(x, implementation, default_vllm_config)
    torch.testing.assert_close(output, expected, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("implementation", SSSGLU_AND_MUL_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul_zero_tokens(
    default_vllm_config,
    implementation: str,
    device: str,
) -> None:
    d = 576
    x = torch.empty((0, 2 * d), dtype=torch.bfloat16, device=device)

    output = _run_sssglu_and_mul(x, implementation, default_vllm_config)
    assert output.shape == (0, d)
    assert output.dtype == x.dtype


@pytest.mark.parametrize("implementation", ["native", "custom_op"])
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul_3d(
    default_vllm_config,
    implementation: str,
    device: str,
) -> None:
    d = 17
    x = torch.randn(
        (2, 3, 2 * d),
        dtype=torch.bfloat16,
        device=device,
    )

    expected = _sssglu_and_mul_reference(x)
    output = _run_sssglu_and_mul(x, implementation, default_vllm_config)

    assert output.shape == (2, 3, d)
    assert output.dtype == x.dtype
    torch.testing.assert_close(output, expected, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize("implementation", SSSGLU_AND_MUL_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul_noncontiguous(
    default_vllm_config,
    implementation: str,
    device: str,
) -> None:
    d = 17
    storage = torch.randn((6, 4 * d), dtype=torch.bfloat16, device=device)
    x = storage[:, ::2]
    assert x.shape == (6, 2 * d)
    assert not x.is_contiguous()

    expected = _sssglu_and_mul_reference(x)
    output = _run_sssglu_and_mul(x, implementation, default_vllm_config)

    assert output.shape == (6, d)
    assert output.dtype == x.dtype
    torch.testing.assert_close(output, expected, atol=2.0e-2, rtol=2.0e-2)


@pytest.mark.parametrize("implementation", SSSGLU_AND_MUL_IMPLEMENTATIONS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_sssglu_and_mul_rejects_odd_dimension(
    default_vllm_config,
    implementation: str,
    device: str,
) -> None:
    x = torch.randn((3, 11), dtype=torch.float32, device=device)

    with pytest.raises(ValueError):
        _run_sssglu_and_mul(x, implementation, default_vllm_config)


SWIGLU_LIMITS = [3.0, 7.0, 15.0]


@pytest.mark.parametrize("swiglu_limit", SWIGLU_LIMITS)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_silu_and_mul_with_clamp(
    default_vllm_config,
    swiglu_limit: float,
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    """SiluAndMulWithClamp: cuda kernel must match native reference."""
    set_random_seed(seed)
    torch.set_default_device(device)
    # Use large values to ensure clamping is exercised.
    x = torch.randn(num_tokens, 2 * d, dtype=dtype) * swiglu_limit * 2

    layer = SiluAndMulWithClamp(swiglu_limit, compile_native=False)
    out = layer(x)
    ref_out = layer.forward_native(x)

    rtol = {
        torch.float16: 2e-3,
        torch.bfloat16: 2e-2,
        torch.float: 1.3e-6,
    }
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=rtol[out.dtype]
    )

    # Verify clamping is actually being applied: the clamped output should
    # differ from the unclamped SiluAndMul output when inputs are large.
    unclamped_out = SiluAndMul.forward_native(x)
    assert not torch.equal(ref_out.float(), unclamped_out.float()), (
        "Input was not large enough to exercise the clamp; increase scale"
    )

    # Verify gate clamping semantics with a controlled scalar case.
    # gate=large_val is clamped to limit first, then silu(limit) * 1.0.
    x_gate = torch.tensor(
        [[swiglu_limit * 20.0, 1.0]], dtype=torch.float32, device=device
    )
    out_gate = SiluAndMulWithClamp(swiglu_limit, compile_native=False)(x_gate)
    expected_gate = torch.nn.functional.silu(
        torch.tensor(swiglu_limit, dtype=torch.float32)
    ).item()
    torch.testing.assert_close(
        out_gate,
        torch.tensor([[expected_gate]], dtype=torch.float32, device=device),
        atol=1e-3,
        rtol=1e-3,
    )

    # Verify up clamping semantics: up >> limit gets clamped to limit.
    x_up = torch.tensor(
        [[1.0, swiglu_limit * 20.0]], dtype=torch.float32, device=device
    )
    out_up = SiluAndMulWithClamp(swiglu_limit, compile_native=False)(x_up)
    silu_1 = torch.nn.functional.silu(torch.tensor(1.0)).item()
    torch.testing.assert_close(
        out_up,
        torch.tensor([[silu_1 * swiglu_limit]], dtype=torch.float32, device=device),
        atol=1e-3,
        rtol=1e-3,
    )

    # opcheck
    out_buf = torch.empty(x.shape[:-1] + (d,), dtype=dtype, device=device)
    opcheck(torch.ops._C.silu_and_mul_with_clamp, (out_buf, x, swiglu_limit))


@pytest.mark.parametrize(
    "activation",
    [
        (FastGELU, torch.ops._C.gelu_fast),
        (NewGELU, torch.ops._C.gelu_new),
        (QuickGELU, torch.ops._C.gelu_quick),
        (ReLUSquaredActivation, torch.ops._C.relu_squared),
    ],
)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("d", D)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", CUDA_DEVICES)
@torch.inference_mode()
def test_activation(
    default_vllm_config,
    activation: type[torch.nn.Module],
    num_tokens: int,
    d: int,
    dtype: torch.dtype,
    seed: int,
    device: str,
) -> None:
    set_random_seed(seed)
    torch.set_default_device(device)
    x = torch.randn(num_tokens, d, dtype=dtype)
    layer = activation[0]()
    fn = activation[1]
    out = layer(x)
    ref_out = layer.forward_native(x)
    torch.testing.assert_close(
        out, ref_out, atol=get_default_atol(out), rtol=get_default_rtol(out)
    )

    out = torch.empty_like(x)
    opcheck(fn, (out, x))
