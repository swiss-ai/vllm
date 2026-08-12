# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
)

import vllm.model_executor.models.apertus2 as apertus2
from vllm.config import CompilationMode, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.models.apertus2 import (
    Apertus2Attention,
    merge_branch_with_residual,
    quantile_balancing_routing,
    quantile_balancing_routing_native,
)


_REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)


def _quantile_balancing_reference(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits_fp32 = logits.to(torch.float32)
    selected_ids = torch.topk(
        logits_fp32 - qb_beta.to(torch.float32),
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices
    selected_weights = torch.sigmoid(logits_fp32).gather(1, selected_ids)
    if renormalize:
        denominator = selected_weights.sum(dim=-1, keepdim=True) + 1e-20
        selected_weights = selected_weights / denominator
    return selected_weights.to(torch.float32), selected_ids.to(torch.int32)


def _canonicalize_routes(
    weights: torch.Tensor,
    ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    order = ids.argsort(dim=-1)
    return weights.gather(1, order), ids.gather(1, order)


def _assert_routes_match(
    actual: tuple[torch.Tensor, torch.Tensor],
    expected: tuple[torch.Tensor, torch.Tensor],
) -> None:
    actual_weights, actual_ids = _canonicalize_routes(*actual)
    expected_weights, expected_ids = _canonicalize_routes(*expected)
    torch.testing.assert_close(actual_ids, expected_ids, atol=0, rtol=0)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=1e-6,
        rtol=1e-6,
    )


def _qb_inputs(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [-100.0, -20.0, -5.0, -1.0, -0.01, 0.0, 0.01, 1.0, 20.0, 100.0],
            [100.0, 20.0, 5.0, 1.0, 0.01, 0.0, -0.01, -1.0, -20.0, -100.0],
            [3.0001, 3.0, 2.9999, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    qb_beta = torch.tensor(
        [0.75, -0.5, 0.25, -0.125, 0.0625, -0.03125, 0.015625, 1.5, -2.0, 4.0],
        dtype=torch.float32,
        device=device,
    )
    return logits, qb_beta


@pytest.mark.parametrize("renormalize", [False, True])
def test_quantile_balancing_routing_native_matches_reference(
    renormalize: bool,
) -> None:
    logits, qb_beta = _qb_inputs("cpu")
    logits_before = logits.clone()
    qb_beta_before = qb_beta.clone()

    actual = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=8,
        renormalize=renormalize,
    )
    expected = _quantile_balancing_reference(
        logits,
        qb_beta,
        top_k=8,
        renormalize=renormalize,
    )

    _assert_routes_match(actual, expected)
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    torch.testing.assert_close(logits, logits_before, atol=0, rtol=0)
    torch.testing.assert_close(qb_beta, qb_beta_before, atol=0, rtol=0)


def test_quantile_balancing_differs_from_deepseek_correction_bias() -> None:
    logits = torch.tensor([[10.0, 1.0, -5.0]])
    qb_beta = torch.tensor([8.0, 0.5, 0.0])

    weights, ids = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=1,
        renormalize=False,
    )
    deepseek_like_id = torch.topk(
        torch.sigmoid(logits) - qb_beta,
        k=1,
        dim=-1,
    ).indices

    assert ids.item() == 0
    assert deepseek_like_id.item() == 1
    torch.testing.assert_close(weights, torch.sigmoid(logits[:, :1]))


def test_quantile_balancing_true_tie_has_documented_topk_semantics() -> None:
    logits = torch.tensor([[2.0, 3.0, -4.0]])
    qb_beta = torch.tensor([1.0, 2.0, 0.0])

    weights, ids = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=1,
        renormalize=False,
    )

    selected_id = ids.item()
    assert selected_id in (0, 1)
    expected_weight = torch.sigmoid(logits[0, selected_id])
    torch.testing.assert_close(weights[0, 0], expected_weight)


def test_quantile_balancing_normalization_epsilon_keeps_zeros_finite() -> None:
    logits = torch.full((2, 10), -1000.0)
    qb_beta = torch.arange(10, dtype=torch.float32)

    weights, _ = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=8,
        renormalize=True,
    )

    assert torch.isfinite(weights).all()
    torch.testing.assert_close(weights, torch.zeros_like(weights), atol=0, rtol=0)


def test_quantile_balancing_zero_and_padded_tokens() -> None:
    qb_beta = torch.linspace(-0.5, 0.5, 10)
    empty_logits = torch.empty((0, 10), dtype=torch.float32)

    empty_weights, empty_ids = quantile_balancing_routing_native(
        empty_logits,
        qb_beta,
        top_k=8,
        renormalize=True,
    )
    assert empty_weights.shape == (0, 8)
    assert empty_ids.shape == (0, 8)

    actual_logits, _ = _qb_inputs("cpu")
    padded_logits = torch.full((8, 10), -50.0)
    padded_logits[: actual_logits.shape[0]].copy_(actual_logits)
    actual_routes = quantile_balancing_routing_native(
        actual_logits,
        qb_beta,
        top_k=8,
        renormalize=True,
    )
    padded_routes = quantile_balancing_routing_native(
        padded_logits,
        qb_beta,
        top_k=8,
        renormalize=True,
    )

    _assert_routes_match(
        (padded_routes[0][:3], padded_routes[1][:3]),
        actual_routes,
    )


def test_quantile_balancing_accepts_fp32_logits_from_bf16_hidden_states() -> None:
    hidden_states = torch.tensor(
        [[1.0, -2.0, 3.0, -4.0], [-1.0, 0.5, 2.0, -3.0]],
        dtype=torch.bfloat16,
    )
    gate_weight = torch.arange(40, dtype=torch.float32).view(10, 4) / 17
    logits = hidden_states.float() @ gate_weight.transpose(0, 1)
    qb_beta = torch.linspace(-1.0, 1.0, 10, dtype=torch.float32)

    actual = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=8,
        renormalize=False,
    )
    expected = _quantile_balancing_reference(
        logits,
        qb_beta,
        top_k=8,
        renormalize=False,
    )

    assert logits.dtype == torch.float32
    assert qb_beta.dtype == torch.float32
    _assert_routes_match(actual, expected)


@_REQUIRES_CUDA
@torch.inference_mode()
def test_quantile_balancing_compiled_fullgraph_matches_reference() -> None:
    torch._dynamo.reset()
    logits, qb_beta = _qb_inputs("cuda")

    for current_logits in (logits, logits.repeat(3, 1)[:8]):
        logits_before = current_logits.clone()
        qb_beta_before = qb_beta.clone()
        actual = quantile_balancing_routing(
            current_logits,
            qb_beta,
            top_k=8,
            renormalize=True,
        )
        expected = _quantile_balancing_reference(
            current_logits,
            qb_beta,
            top_k=8,
            renormalize=True,
        )

        _assert_routes_match(actual, expected)
        torch.testing.assert_close(current_logits, logits_before, atol=0, rtol=0)
        torch.testing.assert_close(qb_beta, qb_beta_before, atol=0, rtol=0)


@_REQUIRES_CUDA
@torch.inference_mode()
def test_quantile_balancing_cuda_graph_replay_does_not_mutate_inputs() -> None:
    static_logits, static_qb_beta = _qb_inputs("cuda")
    quantile_balancing_routing_native(
        static_logits,
        static_qb_beta,
        top_k=8,
        renormalize=True,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_weights, graph_ids = quantile_balancing_routing_native(
            static_logits,
            static_qb_beta,
            top_k=8,
            renormalize=True,
        )
    torch.cuda.synchronize()

    for offset in (0.0, 0.375):
        next_logits, next_qb_beta = _qb_inputs("cuda")
        static_logits.copy_(next_logits + offset)
        static_qb_beta.copy_(next_qb_beta - offset / 3)
        logits_before = static_logits.clone()
        qb_beta_before = static_qb_beta.clone()
        expected = _quantile_balancing_reference(
            static_logits,
            static_qb_beta,
            top_k=8,
            renormalize=True,
        )

        graph.replay()
        torch.cuda.synchronize()

        _assert_routes_match((graph_weights, graph_ids), expected)
        torch.testing.assert_close(static_logits, logits_before, atol=0, rtol=0)
        torch.testing.assert_close(static_qb_beta, qb_beta_before, atol=0, rtol=0)


def test_apertus2_moe_wires_qb_latent_and_shared_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_args = {}
    projection_calls = []
    shared_args = {}
    factory_args = {}

    class FakeGate(nn.Module):
        def __init__(self, output_size: int) -> None:
            super().__init__()
            self.output_size = output_size
            self.last_input = None

        def forward(self, hidden_states: torch.Tensor):
            self.last_input = hidden_states
            logits = torch.zeros(
                hidden_states.shape[0],
                self.output_size,
                dtype=torch.float32,
            )
            return logits, None

    class FakeExperts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_hidden_states = None
            self.last_router_logits = None

        def forward(self, hidden_states, router_logits):
            self.last_hidden_states = hidden_states
            self.last_router_logits = router_logits
            return hidden_states

    def make_gate(**kwargs):
        gate_args.update(kwargs)
        return FakeGate(kwargs["output_size"])

    def make_projection(**kwargs):
        projection = nn.Identity()
        projection_calls.append((kwargs, projection))
        return projection

    def make_shared_experts(**kwargs):
        shared_args.update(kwargs)
        return nn.Identity()

    def make_experts(**kwargs):
        factory_args.update(kwargs)
        return FakeExperts()

    monkeypatch.setattr(apertus2, "GateLinear", make_gate)
    monkeypatch.setattr(apertus2, "ReplicatedLinear", make_projection)
    monkeypatch.setattr(apertus2, "Apertus2MLP", make_shared_experts)
    monkeypatch.setattr(apertus2, "FusedMoEFactory", make_experts)

    config = SimpleNamespace(
        hidden_size=4,
        hidden_act="sssglu",
        n_routed_experts=5,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        moe_latent_size=2,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        use_quantile_balancing=True,
    )
    moe = apertus2.Apertus2MoE(config, prefix="model.layers.1.mlp")

    assert gate_args == {
        "input_size": 4,
        "output_size": 5,
        "bias": False,
        "out_dtype": torch.float32,
        "force_fp32_compute": True,
        "prefix": "model.layers.1.mlp.gate",
    }
    assert [call[0] for call in projection_calls] == [
        {
            "input_size": 4,
            "output_size": 2,
            "bias": False,
            "quant_config": None,
            "prefix": "model.layers.1.mlp.latent_down_proj",
        },
        {
            "input_size": 2,
            "output_size": 4,
            "bias": False,
            "quant_config": None,
            "prefix": "model.layers.1.mlp.latent_up_proj",
        },
    ]
    assert shared_args["hidden_size"] == 4
    assert shared_args["intermediate_size"] == 3
    assert shared_args["reduce_results"] is False
    assert factory_args["hidden_size"] == 2
    assert factory_args["intermediate_size"] == 3
    assert factory_args["shared_experts"] is moe.shared_experts
    assert factory_args["routed_input_transform"] is moe.latent_down_proj
    assert factory_args["routed_output_transform"] is moe.latent_up_proj
    assert factory_args["activation"] == "sssglu"
    assert factory_args["routed_scaling_factor"] == 2.5
    assert factory_args["apply_routed_scale_to_output"] is True
    assert moe.gate.qb_beta.dtype == torch.float32
    assert moe.gate.e_score_correction_bias.dtype == torch.float32

    routing_call = {}
    expected_weights = torch.tensor([[0.25, 0.75]], dtype=torch.float32)
    expected_ids = torch.tensor([[1, 3]], dtype=torch.int32)

    def fake_qb(*, logits, qb_beta, top_k, renormalize):
        routing_call.update(
            logits=logits,
            qb_beta=qb_beta,
            top_k=top_k,
            renormalize=renormalize,
        )
        return expected_weights, expected_ids

    monkeypatch.setattr(apertus2, "quantile_balancing_routing", fake_qb)
    routing_logits = torch.randn(1, 5)
    actual_weights, actual_ids = factory_args["custom_routing_function"](
        hidden_states=torch.randn(1, 2),
        gating_output=routing_logits,
        topk=2,
        renormalize=True,
    )

    assert routing_call["logits"] is routing_logits
    assert routing_call["qb_beta"] is moe.gate.qb_beta
    assert routing_call["top_k"] == 2
    assert routing_call["renormalize"] is True
    assert actual_weights is expected_weights
    assert actual_ids is expected_ids

    hidden_states = torch.randn(2, 3, 4)
    output = moe(hidden_states)
    assert output.shape == hidden_states.shape
    assert moe.gate.last_input.shape == (6, 4)
    assert moe.experts.last_hidden_states.shape == (6, 4)
    assert moe.experts.last_router_logits.shape == (6, 5)


def test_apertus2_moe_rejects_non_qb_routing() -> None:
    with pytest.raises(ValueError, match="only QB routing"):
        apertus2.Apertus2MoE(SimpleNamespace(use_quantile_balancing=False))


def _affine_post_norm(value: torch.Tensor) -> torch.Tensor:
    return value * 2.0 + 3.0


def _plain_residual(
    residual_stream: torch.Tensor,
    branch_output: torch.Tensor,
) -> torch.Tensor:
    return merge_branch_with_residual(
        residual_stream,
        branch_output,
        residual_multiplier=0.25,
    )


def _sandwich_residual(
    residual_stream: torch.Tensor,
    branch_output: torch.Tensor,
) -> torch.Tensor:
    return merge_branch_with_residual(
        residual_stream,
        branch_output,
        residual_multiplier=0.25,
        post_norm=_affine_post_norm,
    )


@pytest.mark.parametrize("branch_kind", ["attention", "feed_forward"])
@pytest.mark.parametrize("sandwich", [False, True])
def test_merge_branch_with_residual_matches_reference(
    branch_kind: str,
    sandwich: bool,
) -> None:
    values = {
        "attention": ([1.0, -2.0, 4.0], [3.0, 5.0, -7.0]),
        "feed_forward": ([-3.0, 2.0, 8.0], [6.0, -1.0, 0.5]),
    }
    residual_stream = torch.tensor(values[branch_kind][0])
    branch_output = torch.tensor(values[branch_kind][1])
    residual_before = residual_stream.clone()
    branch_before = branch_output.clone()

    if sandwich:
        actual = _sandwich_residual(residual_stream, branch_output)
        expected = residual_stream + 0.25 * _affine_post_norm(branch_output)
        wrong_order = residual_stream + _affine_post_norm(0.25 * branch_output)
        assert not torch.equal(actual, wrong_order)
    else:
        actual = _plain_residual(residual_stream, branch_output)
        expected = residual_stream + 0.25 * branch_output

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual_stream, residual_before, atol=0, rtol=0)
    torch.testing.assert_close(branch_output, branch_before, atol=0, rtol=0)


@_REQUIRES_CUDA
@pytest.mark.parametrize("sandwich", [False, True])
@torch.inference_mode()
def test_merge_branch_with_residual_compiled_fullgraph(sandwich: bool) -> None:
    torch._dynamo.reset()
    residual_stream = torch.randn(4, 16, device="cuda")
    branch_output = torch.randn(4, 16, device="cuda")
    residual_before = residual_stream.clone()
    branch_before = branch_output.clone()
    function = _sandwich_residual if sandwich else _plain_residual
    compiled = torch.compile(function, backend="inductor", fullgraph=True)

    actual = compiled(residual_stream, branch_output)
    expected = function(residual_stream, branch_output)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(residual_stream, residual_before, atol=0, rtol=0)
    torch.testing.assert_close(branch_output, branch_before, atol=0, rtol=0)


@_REQUIRES_CUDA
@pytest.mark.parametrize("sandwich", [False, True])
@torch.inference_mode()
def test_merge_branch_with_residual_cuda_graph_replay(sandwich: bool) -> None:
    static_residual = torch.randn(4, 16, device="cuda")
    static_branch = torch.randn(4, 16, device="cuda")
    function = _sandwich_residual if sandwich else _plain_residual
    function(static_residual, static_branch)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_output = function(static_residual, static_branch)
    torch.cuda.synchronize()

    for offset in (0.0, 1.25):
        static_residual.copy_(torch.randn_like(static_residual) + offset)
        static_branch.copy_(torch.randn_like(static_branch) - offset)
        residual_before = static_residual.clone()
        branch_before = static_branch.clone()
        expected = function(static_residual, static_branch)

        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, expected, atol=0, rtol=0)
        torch.testing.assert_close(static_residual, residual_before, atol=0, rtol=0)
        torch.testing.assert_close(static_branch, branch_before, atol=0, rtol=0)


class _QKVProjection(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int,
        **kwargs,
    ) -> None:
        super().__init__()
        del hidden_size, kwargs
        self.output_size = (total_num_heads + 2 * total_num_kv_heads) * head_size

    def forward(self, hidden_states: torch.Tensor):
        values = torch.arange(
            self.output_size,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        return values.expand(hidden_states.shape[0], -1), None


class _OutputProjection(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _RecordingNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states + 1


class _RecordingRoPE(nn.Module):
    def __init__(self, is_neox_style: bool) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.calls = 0

    def forward(self, positions, query, key):
        del positions
        self.calls += 1
        return query + 10, key + 10


class _RecordingAttention(nn.Module):
    def __init__(
        self,
        *args,
        per_layer_sliding_window: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        del args, kwargs
        self.sliding_window = per_layer_sliding_window
        self.calls = 0

    def forward(self, query, key, value):
        del key, value
        self.calls += 1
        return query


@pytest.fixture
def stub_apertus2_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apertus2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(apertus2, "QKVParallelLinear", _QKVProjection)
    monkeypatch.setattr(apertus2, "RowParallelLinear", _OutputProjection)
    monkeypatch.setattr(apertus2, "RMSNorm", _RecordingNorm)
    monkeypatch.setattr(apertus2, "Attention", _RecordingAttention)
    monkeypatch.setattr(
        apertus2,
        "get_rope",
        lambda *args, **kwargs: _RecordingRoPE(kwargs["is_neox_style"]),
    )


@pytest.mark.parametrize(
    ("layer_idx", "expected_rope", "expected_window"),
    [
        (0, True, 513),
        (1, False, 513),
        (2, True, None),
        (3, False, None),
    ],
)
def test_apertus2_attention_schedules_are_independent(
    stub_apertus2_attention,
    layer_idx: int,
    expected_rope: bool,
    expected_window: int | None,
) -> None:
    config = SimpleNamespace(
        head_dim=4,
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "full_attention",
        ],
        no_rope_layers=[1, 0, 1, 0],
        sliding_window=513,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    )
    attention = Apertus2Attention(
        config=config,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        prefix=f"model.layers.{layer_idx}.self_attn",
    )

    output = attention(torch.arange(2), torch.ones(2, 8))

    assert attention.use_rope is expected_rope
    assert attention.attn.sliding_window == expected_window
    assert attention.q_norm.calls == 1
    assert attention.k_norm.calls == 1
    assert attention.rotary_emb.calls == int(expected_rope)
    assert attention.rotary_emb.is_neox_style
    assert attention.attn.calls == 1
    assert output.shape == (2, 8)


def test_apertus2_attention_defaults_to_rope(stub_apertus2_attention) -> None:
    config = SimpleNamespace(
        head_dim=4,
        layer_types=["full_attention"],
        sliding_window=None,
        rms_norm_eps=1e-5,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
    )

    attention = Apertus2Attention(
        config=config,
        hidden_size=8,
        num_heads=2,
        num_kv_heads=1,
        prefix="model.layers.0.self_attn",
    )

    assert attention.use_rope


def test_apertus2_rope_matches_hugging_face_at_head_dim_112() -> None:
    head_dim = 112
    num_heads = 2
    num_kv_heads = 1
    positions = torch.tensor([0, 511, 512, 8191])
    rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10_000.0,
        "partial_rotary_factor": 1.0,
    }
    query = torch.randn(len(positions), num_heads * head_dim)
    key = torch.randn(len(positions), num_kv_heads * head_dim)

    hf_config = LlamaConfig(
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=8192,
        rope_parameters=rope_parameters,
    )
    hf_rope = LlamaRotaryEmbedding(hf_config)
    hf_query = query.view(1, len(positions), num_heads, head_dim).transpose(1, 2)
    hf_key = key.view(1, len(positions), num_kv_heads, head_dim).transpose(1, 2)
    cos, sin = hf_rope(hf_query, positions.unsqueeze(0))
    expected_query, expected_key = apply_rotary_pos_emb(hf_query, hf_key, cos, sin)
    expected_query = expected_query.transpose(1, 2).reshape_as(query)
    expected_key = expected_key.transpose(1, 2).reshape_as(key)

    with set_current_vllm_config(VllmConfig()):
        native_rope = get_rope(
            head_dim,
            max_position=8192,
            rope_parameters=rope_parameters,
            is_neox_style=True,
            dtype=query.dtype,
        )
        actual_query, actual_key = native_rope.forward_native(
            positions, query.clone(), key.clone()
        )

    torch.testing.assert_close(actual_query, expected_query)
    assert actual_key is not None
    torch.testing.assert_close(actual_key, expected_key)


def _model_vllm_config(quant_config=None, dtype=torch.bfloat16):
    config = SimpleNamespace(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        embedding_multiplier=3.0,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=config, dtype=dtype),
        cache_config=None,
        quant_config=quant_config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize("unsupported_mode", ["quantized", "float16"])
def test_apertus2_model_rejects_unsupported_precision_before_building_layers(
    monkeypatch: pytest.MonkeyPatch,
    unsupported_mode: str,
) -> None:
    monkeypatch.setattr(
        apertus2,
        "get_pp_group",
        lambda: pytest.fail("PP state must not be queried for unsupported Apertus2"),
    )

    quant_config = object() if unsupported_mode == "quantized" else None
    dtype = torch.float16 if unsupported_mode == "float16" else torch.bfloat16

    with pytest.raises(ValueError, match="only unquantized BF16 inference"):
        apertus2.Apertus2Model(
            vllm_config=_model_vllm_config(
                quant_config=quant_config,
                dtype=dtype,
            ),
            prefix="model",
        )


def test_apertus2_model_rejects_sequence_parallel_before_building_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        apertus2,
        "get_pp_group",
        lambda: pytest.fail("PP state must not be queried for unsupported Apertus2"),
    )
    vllm_config = _model_vllm_config()
    vllm_config.parallel_config.use_sequence_parallel_moe = True

    with pytest.raises(ValueError, match="deferred to Phase 3"):
        apertus2.Apertus2Model(vllm_config=vllm_config, prefix="model")


def test_apertus2_model_pipeline_flow_matches_hf_embedding_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=False)

    class FakeEmbedding(nn.Module):
        def __init__(self, num_embeddings, embedding_dim, **kwargs) -> None:
            super().__init__()
            del num_embeddings, kwargs
            self.embedding_dim = embedding_dim

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.float().unsqueeze(-1).expand(-1, self.embedding_dim)

    class FakeLayer(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.prefix = kwargs["prefix"]

        def forward(self, positions, hidden_states):
            del positions
            return hidden_states + 1.0

    class FakeNorm(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs

        def forward(self, hidden_states):
            return hidden_states * 2.0

    def fake_make_layers(num_layers, layer_fn, prefix):
        split = num_layers // 2
        start_layer, end_layer = (
            (0, split) if pp_group.is_first_rank else (split, num_layers)
        )
        layers = nn.ModuleList(
            [
                layer_fn(f"{prefix}.{layer_idx}")
                if start_layer <= layer_idx < end_layer
                else nn.Identity()
                for layer_idx in range(num_layers)
            ]
        )
        return start_layer, end_layer, layers

    monkeypatch.setattr(apertus2, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(apertus2, "VocabParallelEmbedding", FakeEmbedding)
    monkeypatch.setattr(apertus2, "RMSNorm", FakeNorm)
    monkeypatch.setattr(apertus2, "make_layers", fake_make_layers)

    vllm_config = _model_vllm_config()
    first_stage = apertus2.Apertus2Model(
        vllm_config=vllm_config,
        prefix="model",
        layer_type=FakeLayer,
    )
    positions = torch.arange(2)
    input_ids = torch.tensor([1, 2])
    first_output = first_stage(input_ids, positions, None)

    assert isinstance(first_output, apertus2.IntermediateTensors)
    assert set(first_output.tensors) == {"hidden_states"}
    torch.testing.assert_close(
        first_output["hidden_states"],
        torch.tensor([[4.0] * 4, [7.0] * 4]),
    )
    assert first_stage.layers[0].prefix == "model.layers.0"

    inputs_embeds = torch.tensor([[2.0] * 4, [-1.0] * 4])
    inputs_embeds_before = inputs_embeds.clone()
    embedded_output = first_stage(None, positions, None, inputs_embeds)
    torch.testing.assert_close(
        embedded_output["hidden_states"],
        inputs_embeds * 3.0 + 1.0,
    )
    torch.testing.assert_close(inputs_embeds, inputs_embeds_before)

    empty = first_stage.make_empty_intermediate_tensors(
        batch_size=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert set(empty.tensors) == {"hidden_states"}
    assert empty["hidden_states"].shape == (2, 4)

    pp_group.is_first_rank = False
    pp_group.is_last_rank = True
    last_stage = apertus2.Apertus2Model(
        vllm_config=vllm_config,
        prefix="model",
        layer_type=FakeLayer,
    )
    assert last_stage.layers[1].prefix == "model.layers.1"
    final_output = last_stage(None, positions, first_output)

    torch.testing.assert_close(
        final_output,
        torch.tensor([[10.0] * 4, [16.0] * 4]),
    )


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_name", "expected_shard_id"),
    [
        (
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "q",
        ),
        (
            "model.layers.0.self_attn.k_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "k",
        ),
        (
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.qkv_proj.weight",
            "v",
        ),
        (
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        ),
        (
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        ),
        (
            "model.layers.1.mlp.shared_experts.gate_proj.weight",
            "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
            0,
        ),
        (
            "model.layers.1.mlp.shared_experts.up_proj.weight",
            "model.layers.1.mlp.shared_experts.gate_up_proj.weight",
            1,
        ),
        (
            "model.layers.1.mlp.experts.7.gate_proj.weight",
            "model.layers.1.mlp.experts.7.gate_proj.weight",
            None,
        ),
        (
            "model.layers.1.mlp.gate.qb_beta",
            "model.layers.1.mlp.gate.qb_beta",
            None,
        ),
        ("lm_head.weight", "lm_head.weight", None),
    ],
)
def test_apertus2_causal_lm_weight_mapper(
    checkpoint_name: str,
    expected_name: str,
    expected_shard_id: str | int | None,
) -> None:
    checkpoint_weight = torch.empty(1)

    [(mapped_name, mapped_weight)] = list(
        apertus2.Apertus2ForCausalLM.hf_to_vllm_mapper.apply(
            [(checkpoint_name, checkpoint_weight)]
        )
    )

    assert mapped_name == expected_name
    assert mapped_weight is checkpoint_weight
    assert getattr(mapped_weight, "shard_id", None) == expected_shard_id


def test_apertus2_causal_lm_builds_untied_head_and_delegates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    construction = SimpleNamespace()

    class FakeLayer(nn.Module):
        pass

    class FakeModel(nn.Module):
        def __init__(self, *, vllm_config, prefix, layer_type) -> None:
            super().__init__()
            construction.model = (vllm_config, prefix, layer_type)
            self.embed_tokens = nn.Embedding(2, 4)
            self.empty_factory = lambda *args, **kwargs: (args, kwargs)
            self.make_empty_intermediate_tensors = self.empty_factory
            self.forward_args = None

        def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids + 10

        def forward(
            self,
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        ):
            self.forward_args = (
                input_ids,
                positions,
                intermediate_tensors,
                inputs_embeds,
            )
            return inputs_embeds if inputs_embeds is not None else input_ids + 1

    class FakeLMHead(nn.Module):
        def __init__(
            self,
            vocab_size,
            hidden_size,
            *,
            quant_config,
            prefix,
        ) -> None:
            super().__init__()
            construction.lm_head = (
                vocab_size,
                hidden_size,
                quant_config,
                prefix,
            )
            self.weight = nn.Parameter(torch.empty(vocab_size, hidden_size))

    class FakeLogitsProcessor(nn.Module):
        def __init__(self, vocab_size) -> None:
            super().__init__()
            construction.logits_vocab_size = vocab_size
            self.call_args = None

        def forward(self, lm_head, hidden_states):
            self.call_args = (lm_head, hidden_states)
            return hidden_states + 2

    monkeypatch.setattr(apertus2, "get_pp_group", lambda: pp_group)
    monkeypatch.setattr(apertus2, "Apertus2Model", FakeModel)
    monkeypatch.setattr(apertus2, "ParallelLMHead", FakeLMHead)
    monkeypatch.setattr(apertus2, "LogitsProcessor", FakeLogitsProcessor)

    vllm_config = _model_vllm_config()
    causal_lm = apertus2.Apertus2ForCausalLM(
        vllm_config=vllm_config,
        prefix="root",
        layer_type=FakeLayer,
    )

    assert construction.model == (vllm_config, "root.model", FakeLayer)
    assert construction.lm_head == (16, 4, None, "root.lm_head")
    assert construction.logits_vocab_size == 16
    assert causal_lm.lm_head.weight is not causal_lm.model.embed_tokens.weight
    assert (
        causal_lm.make_empty_intermediate_tensors
        is causal_lm.model.make_empty_intermediate_tensors
    )

    input_ids = torch.tensor([1, 2])
    positions = torch.tensor([3, 4])
    inputs_embeds = torch.randn(2, 4)
    torch.testing.assert_close(causal_lm.embed_input_ids(input_ids), input_ids + 10)
    assert causal_lm(input_ids, positions, None, inputs_embeds) is inputs_embeds
    forward_input_ids, forward_positions, forward_intermediate, forward_embeds = (
        causal_lm.model.forward_args
    )
    assert forward_input_ids is input_ids
    assert forward_positions is positions
    assert forward_intermediate is None
    assert forward_embeds is inputs_embeds

    hidden_states = torch.randn(2, 4)
    logits = causal_lm.compute_logits(hidden_states)
    torch.testing.assert_close(logits, hidden_states + 2)
    logits_head, logits_hidden_states = causal_lm.logits_processor.call_args
    assert logits_head is causal_lm.lm_head
    assert logits_hidden_states is hidden_states

    pp_group.is_last_rank = False
    non_last_stage = apertus2.Apertus2ForCausalLM(
        vllm_config=vllm_config,
        layer_type=FakeLayer,
    )
    assert isinstance(non_last_stage.lm_head, apertus2.PPMissingLayer)
    assert not hasattr(non_last_stage, "logits_processor")


def test_apertus2_causal_lm_loads_with_its_weight_mapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = SimpleNamespace()

    class FakeAutoWeightsLoader:
        def __init__(self, module) -> None:
            captured.module = module

        def load_weights(self, weights, *, mapper):
            captured.weights = list(weights)
            captured.mapper = mapper
            return {"loaded.weight"}

    monkeypatch.setattr(apertus2, "AutoWeightsLoader", FakeAutoWeightsLoader)
    causal_lm = apertus2.Apertus2ForCausalLM.__new__(
        apertus2.Apertus2ForCausalLM
    )
    nn.Module.__init__(causal_lm)
    weights = [("lm_head.weight", torch.empty(1))]

    loaded = causal_lm.load_weights(iter(weights))

    assert loaded == {"loaded.weight"}
    assert captured.module is causal_lm
    assert captured.weights == weights
    assert captured.mapper is causal_lm.hf_to_vllm_mapper
