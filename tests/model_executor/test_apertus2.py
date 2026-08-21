# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import vllm.model_executor.models.apertus2 as apertus2
from vllm.config import CompilationMode
from vllm.model_executor.models.apertus2 import (
    Apertus2Attention,
    merge_branch_with_residual,
    quantile_balancing_routing,
    quantile_balancing_routing_native,
)


def _qb_reference(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
    sigmoid_selection: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = logits.float()
    qb_scores = torch.sigmoid(logits) if sigmoid_selection else logits
    ids = torch.topk(qb_scores - qb_beta.float(), top_k, dim=-1, sorted=False).indices
    weights = torch.sigmoid(logits).gather(1, ids)
    if renormalize:
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    return weights.float(), ids.int()


@pytest.mark.parametrize("sigmoid_selection", [True, False])
@pytest.mark.parametrize("renormalize", [False, True])
def test_quantile_balancing_routing_matches_reference(
    renormalize: bool, sigmoid_selection: bool
) -> None:
    logits = torch.tensor(
        [
            [3.0, -1.0, 0.5, 2.0, -4.0],
            [-2.0, 5.0, 0.0, 1.0, 4.0],
        ],
        dtype=torch.bfloat16,
    )
    qb_beta = torch.tensor([0.5, -0.25, 1.0, 0.0, -2.0])
    logits_before = logits.clone()
    beta_before = qb_beta.clone()

    actual = quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k=3,
        renormalize=renormalize,
        sigmoid_selection=sigmoid_selection,
    )
    expected = _qb_reference(
        logits,
        qb_beta,
        top_k=3,
        renormalize=renormalize,
        sigmoid_selection=sigmoid_selection,
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])
    assert actual[0].dtype == torch.float32
    assert actual[1].dtype == torch.int32
    torch.testing.assert_close(logits, logits_before)
    torch.testing.assert_close(qb_beta, beta_before)


@pytest.mark.parametrize(
    ("sigmoid_selection", "expected_id"),
    [(False, 0), (True, 1)],
)
def test_quantile_balancing_uses_unbiased_logits_for_weights(
    sigmoid_selection: bool, expected_id: int
) -> None:
    """The score space decides the winner; the weight stays bias-free sigmoid.

    Legacy raw-logit selection keeps expert 0 (10 - 8 = 2 still wins), while
    sigmoid selection saturates near 1 so the same beta pushes expert 0 below
    expert 1.
    """
    logits = torch.tensor([[10.0, 1.0, -5.0]])
    qb_beta = torch.tensor([8.0, 0.5, 0.0])

    weights, ids = quantile_balancing_routing_native(
        logits, qb_beta, top_k=1, renormalize=False, sigmoid_selection=sigmoid_selection
    )

    assert ids.item() == expected_id
    torch.testing.assert_close(
        weights, torch.sigmoid(logits[:, expected_id : expected_id + 1])
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("sigmoid_selection", [True, False])
@torch.inference_mode()
def test_quantile_balancing_compiled_matches_reference(sigmoid_selection: bool) -> None:
    torch._dynamo.reset()
    logits = torch.randn(7, 13, device="cuda")
    qb_beta = torch.randn(13, device="cuda")

    actual = quantile_balancing_routing(
        logits, qb_beta, top_k=4, renormalize=True, sigmoid_selection=sigmoid_selection
    )
    expected = _qb_reference(
        logits, qb_beta, top_k=4, renormalize=True, sigmoid_selection=sigmoid_selection
    )

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, "sigmoid"),
        ("sigmoid", "sigmoid"),
        ("average", "sigmoid"),
        ("histogram", "sigmoid"),
        ("legacy", "legacy"),
        ("legacy_average", "legacy"),
    ],
)
def test_quantile_balancing_method_resolution(stored, expected: str) -> None:
    config = SimpleNamespace()
    if stored is not None:
        config.moe_router_quantile_balancing_method = stored

    assert apertus2.resolve_quantile_balancing_method(config) == expected


def test_quantile_balancing_method_rejects_unknown_values() -> None:
    config = SimpleNamespace(moe_router_quantile_balancing_method="softmax")

    with pytest.raises(ValueError, match="'sigmoid' or 'legacy'"):
        apertus2.resolve_quantile_balancing_method(config)


def test_apertus2_moe_wires_qb_latent_and_shared_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {"projections": []}

    class FakeGate(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["gate"] = kwargs
            self.output_size = kwargs["output_size"]

        def forward(self, hidden_states: torch.Tensor):
            logits = hidden_states.new_zeros(
                (hidden_states.shape[0], self.output_size), dtype=torch.float32
            )
            return logits, None

    class FakeProjection(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["projections"].append(kwargs)

    class FakeSharedExpert(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["shared"] = kwargs

    class FakeExperts(nn.Module):
        def forward(self, hidden_states, router_logits):
            captured["expert_inputs"] = (hidden_states, router_logits)
            return hidden_states

    def make_experts(**kwargs):
        captured["experts"] = kwargs
        return FakeExperts()

    monkeypatch.setattr(apertus2, "GateLinear", FakeGate)
    monkeypatch.setattr(apertus2, "ReplicatedLinear", FakeProjection)
    monkeypatch.setattr(apertus2, "Apertus2MLP", FakeSharedExpert)
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
    moe = apertus2.Apertus2MoE(
        config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        prefix="model.layers.1.mlp",
    )
    gate_args = captured["gate"]
    expert_args = captured["experts"]

    assert gate_args["out_dtype"] == torch.float32
    assert gate_args["force_fp32_compute"] is True
    assert [args["output_size"] for args in captured["projections"]] == [2, 4]
    assert captured["shared"]["reduce_results"] is False
    assert captured["shared"]["is_sequence_parallel"] is False
    assert expert_args["is_sequence_parallel"] is False
    assert expert_args["hidden_size"] == 2
    assert expert_args["activation"] == "sssglu"
    assert expert_args["shared_experts"] is moe.shared_experts
    assert expert_args["routed_input_transform"] is moe.latent_down_proj
    assert expert_args["routed_output_transform"] is moe.latent_up_proj
    assert expert_args["routed_scaling_factor"] == 2.5
    assert expert_args["apply_routed_scale_to_output"] is True
    assert expert_args["router_logits_dtype"] == torch.float32
    assert moe.gate.qb_beta.dtype == torch.float32

    routing_call = {}

    def fake_routing(**kwargs):
        routing_call.update(kwargs)
        return torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32)

    monkeypatch.setattr(apertus2, "quantile_balancing_routing", fake_routing)
    router_logits = torch.randn(1, 5)
    expert_args["custom_routing_function"](
        hidden_states=torch.randn(1, 2),
        gating_output=router_logits,
        topk=2,
        renormalize=True,
    )
    assert routing_call == {
        "logits": router_logits,
        "qb_beta": moe.gate.qb_beta,
        "top_k": 2,
        "renormalize": True,
        "sigmoid_selection": True,
    }

    hidden_states = torch.randn(6, 4)
    assert moe(hidden_states).shape == hidden_states.shape
    expert_hidden, expert_logits = captured["expert_inputs"]
    assert expert_hidden.shape == (6, 4)
    assert expert_logits.shape == (6, 5)


def test_apertus2_moe_rejects_non_qb_routing() -> None:
    with pytest.raises(ValueError, match="only QB routing"):
        apertus2.Apertus2MoE(SimpleNamespace(use_quantile_balancing=False))


@pytest.mark.parametrize(
    ("qb_method", "expected"),
    [
        (None, True),
        ("histogram", True),
        ("legacy", False),
        ("legacy_average", False),
    ],
)
def test_apertus2_moe_forwards_selection_space_to_routing(
    monkeypatch: pytest.MonkeyPatch,
    qb_method: str | None,
    expected: bool,
) -> None:
    captured: dict[str, Any] = {}
    moe = _make_stubbed_moe(
        monkeypatch,
        sequence_parallel=False,
        captured=captured,
        qb_method=qb_method,
    )
    assert moe.qb_sigmoid_selection is expected

    routing_call: dict[str, Any] = {}

    def fake_routing(**kwargs):
        routing_call.update(kwargs)
        return torch.ones(1, 2), torch.zeros(1, 2, dtype=torch.int32)

    monkeypatch.setattr(apertus2, "quantile_balancing_routing", fake_routing)
    captured["experts"]["custom_routing_function"](
        hidden_states=torch.randn(1, 4),
        gating_output=torch.randn(1, 5),
        topk=2,
        renormalize=True,
    )
    assert routing_call["sigmoid_selection"] is expected


def _make_stubbed_moe(
    monkeypatch: pytest.MonkeyPatch,
    *,
    sequence_parallel: bool,
    captured: dict[str, Any],
    qb_method: str | None = None,
) -> apertus2.Apertus2MoE:
    """Build an Apertus2MoE whose gate/shared/experts are shape-preserving fakes."""

    class FakeGate(nn.Module):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            self.output_size = kwargs["output_size"]

        def forward(self, hidden_states: torch.Tensor):
            logits = hidden_states.new_zeros(
                (hidden_states.shape[0], self.output_size), dtype=torch.float32
            )
            return logits, None

    class FakeShared(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["shared"] = kwargs

    class FakeExperts(nn.Module):
        def forward(self, hidden_states, router_logits):
            captured["expert_tokens"] = hidden_states.shape[0]
            return hidden_states

    def make_experts(**kwargs):
        captured["experts"] = kwargs
        return FakeExperts()

    monkeypatch.setattr(apertus2, "GateLinear", FakeGate)
    monkeypatch.setattr(apertus2, "Apertus2MLP", FakeShared)
    monkeypatch.setattr(apertus2, "FusedMoEFactory", make_experts)

    config = SimpleNamespace(
        hidden_size=4,
        hidden_act="sssglu",
        n_routed_experts=5,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=3,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        use_quantile_balancing=True,
    )
    if qb_method is not None:
        config.moe_router_quantile_balancing_method = qb_method
    return apertus2.Apertus2MoE(
        config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=sequence_parallel),
        prefix="model.layers.1.mlp",
    )


def test_apertus2_moe_sequence_parallel_chunks_gathers_and_trims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SP forward must trim the gathered output to the pre-chunk token count."""
    captured: dict[str, Any] = {}
    moe = _make_stubbed_moe(monkeypatch, sequence_parallel=True, captured=captured)

    tp_size = 2

    def fake_chunk(hidden_states: torch.Tensor) -> torch.Tensor:
        captured["chunk_tokens"] = hidden_states.shape[0]
        pad_rows = -hidden_states.shape[0] % tp_size
        padded = torch.cat(
            [hidden_states, hidden_states.new_zeros(pad_rows, hidden_states.shape[1])]
        )
        return padded[: padded.shape[0] // tp_size]

    def fake_all_gather(tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 0
        captured["gathered_tokens"] = tensor.shape[0]
        return torch.cat([tensor, torch.full_like(tensor, 7.0)])

    monkeypatch.setattr(apertus2, "sequence_parallel_chunk", fake_chunk)
    monkeypatch.setattr(apertus2, "tensor_model_parallel_all_gather", fake_all_gather)

    hidden_states = torch.randn(5, 4)
    output = moe(hidden_states)

    assert captured["chunk_tokens"] == 5
    assert captured["expert_tokens"] == 3
    assert captured["gathered_tokens"] == 3
    assert output.shape == (5, 4)
    torch.testing.assert_close(output[:3], hidden_states[:3])
    torch.testing.assert_close(output[3:], torch.full((2, 4), 7.0))


def test_apertus2_moe_without_sequence_parallel_skips_collectives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}
    moe = _make_stubbed_moe(monkeypatch, sequence_parallel=False, captured=captured)

    def forbidden(*args, **kwargs):
        raise AssertionError("collective must not run without sequence parallel")

    monkeypatch.setattr(apertus2, "sequence_parallel_chunk", forbidden)
    monkeypatch.setattr(apertus2, "tensor_model_parallel_all_gather", forbidden)

    hidden_states = torch.randn(5, 4)
    torch.testing.assert_close(moe(hidden_states), hidden_states)


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_apertus2_moe_propagates_sequence_parallel_flag(
    monkeypatch: pytest.MonkeyPatch,
    sequence_parallel: bool,
) -> None:
    captured: dict[str, Any] = {}
    _make_stubbed_moe(
        monkeypatch, sequence_parallel=sequence_parallel, captured=captured
    )

    assert captured["shared"]["is_sequence_parallel"] is sequence_parallel
    assert captured["experts"]["is_sequence_parallel"] is sequence_parallel


@pytest.mark.parametrize(
    ("mlp_kwargs", "expected_disable_tp"),
    [
        ({}, False),
        ({"is_sequence_parallel": False}, False),
        ({"is_sequence_parallel": True}, True),
    ],
)
def test_apertus2_mlp_disable_tp_follows_sequence_parallel(
    default_vllm_config,
    monkeypatch: pytest.MonkeyPatch,
    mlp_kwargs: dict[str, Any],
    expected_disable_tp: bool,
) -> None:
    captured: list[dict[str, Any]] = []

    class FakeLinear(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured.append(kwargs)

    monkeypatch.setattr(apertus2, "MergedColumnParallelLinear", FakeLinear)
    monkeypatch.setattr(apertus2, "RowParallelLinear", FakeLinear)

    apertus2.Apertus2MLP(
        hidden_size=4,
        intermediate_size=3,
        hidden_act="sssglu",
        **mlp_kwargs,
    )

    assert [kwargs["disable_tp"] for kwargs in captured] == [expected_disable_tp] * 2


@pytest.mark.parametrize("sandwich", [False, True])
def test_merge_branch_with_residual_matches_reference(sandwich: bool) -> None:
    residual = torch.tensor([1.0, -2.0, 4.0])
    branch = torch.tensor([3.0, 5.0, -7.0])
    post_norm = (lambda value: value * 2.0 + 3.0) if sandwich else None

    actual = merge_branch_with_residual(
        residual, branch, residual_multiplier=0.25, post_norm=post_norm
    )
    normalized = post_norm(branch) if post_norm is not None else branch

    torch.testing.assert_close(actual, residual + 0.25 * normalized)


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
        values = torch.arange(self.output_size, dtype=hidden_states.dtype)
        return values.expand(hidden_states.shape[0], -1), None


class _OutputProjection(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def forward(self, hidden_states: torch.Tensor):
        return hidden_states, None


class _ZeroGateProjection(nn.Module):
    def __init__(self, input_size: int, output_size: int, **kwargs) -> None:
        super().__init__()
        del kwargs
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, hidden_states: torch.Tensor):
        gate = hidden_states.new_zeros(hidden_states.shape[0], self.output_size)
        return gate, None


class _RecordingNorm(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return hidden_states


class _RecordingRoPE(nn.Module):
    def __init__(self, is_neox_style: bool) -> None:
        super().__init__()
        self.is_neox_style = is_neox_style
        self.calls = 0

    def forward(self, positions, query, key):
        self.calls += 1
        return query, key


class _RecordingAttention(nn.Module):
    def __init__(
        self, *args, per_layer_sliding_window: int | None = None, **kwargs
    ) -> None:
        super().__init__()
        self.sliding_window = per_layer_sliding_window

    def forward(self, query, key, value):
        return query


@pytest.fixture
def stub_apertus2_attention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(apertus2, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(apertus2, "QKVParallelLinear", _QKVProjection)
    monkeypatch.setattr(apertus2, "ColumnParallelLinear", _ZeroGateProjection)
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
    [(0, True, 513), (1, False, 513), (2, True, None), (3, False, None)],
)
def test_apertus2_attention_schedules_are_independent(
    stub_apertus2_attention: None,
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

    assert attention.use_rope is expected_rope
    assert attention.attn.sliding_window == expected_window
    assert attention.rotary_emb.is_neox_style
    assert attention(torch.arange(2), torch.ones(2, 8)).shape == (2, 8)
    assert attention.rotary_emb.calls == int(expected_rope)


def test_apertus2_attention_output_gate_defaults_off(
    stub_apertus2_attention: None,
) -> None:
    config = SimpleNamespace(
        head_dim=4,
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

    assert attention.attention_output_gate is False
    assert not hasattr(attention, "g_proj")


def test_apertus2_attention_zero_gate_halves_pre_o_proj_output(
    stub_apertus2_attention: None,
) -> None:
    def build(attention_output_gate: bool) -> Apertus2Attention:
        config = SimpleNamespace(
            head_dim=4,
            rms_norm_eps=1e-5,
            rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
            attention_output_gate=attention_output_gate,
        )
        return Apertus2Attention(
            config=config,
            hidden_size=8,
            num_heads=2,
            num_kv_heads=1,
            prefix="model.layers.0.self_attn",
        )

    ungated = build(False)
    gated = build(True)
    assert gated.g_proj.input_size == 8
    assert gated.g_proj.output_size == 2 * 4

    positions = torch.arange(2)
    hidden_states = torch.ones(2, 8, dtype=torch.bfloat16)
    baseline = ungated(positions, hidden_states)
    gated_output = gated(positions, hidden_states)

    # sigmoid(0) = 0.5, and the stubbed o_proj is the identity.
    torch.testing.assert_close(gated_output, 0.5 * baseline)
    assert gated_output.dtype == hidden_states.dtype


def test_apertus2_decoder_layer_routes_parallel_config_only_to_moe(
    stub_apertus2_attention: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeMoE(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["moe"] = kwargs

    class FakeMLP(nn.Identity):
        def __init__(self, **kwargs) -> None:
            super().__init__()
            captured["mlp"] = kwargs

    monkeypatch.setattr(apertus2, "Apertus2MoE", FakeMoE)
    monkeypatch.setattr(apertus2, "Apertus2MLP", FakeMLP)

    parallel_config = SimpleNamespace(use_sequence_parallel_moe=True)
    config = SimpleNamespace(
        hidden_size=8,
        residual_multiplier=1.0,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=3,
        hidden_act="sssglu",
        rms_norm_eps=1e-5,
        sandwich_norm=False,
        head_dim=4,
        rope_parameters={"rope_type": "default", "rope_theta": 10_000.0},
        is_moe_layer=lambda layer_idx: layer_idx == 1,
    )

    for layer_idx in (0, 1):
        apertus2.Apertus2DecoderLayer(
            config,
            parallel_config=parallel_config,
            prefix=f"model.layers.{layer_idx}",
        )

    assert captured["moe"]["parallel_config"] is parallel_config
    assert "parallel_config" not in captured["mlp"]
    assert "is_sequence_parallel" not in captured["mlp"]


def _model_vllm_config(
    *,
    quant_config=None,
    dtype: torch.dtype = torch.bfloat16,
):
    hf_config = SimpleNamespace(
        vocab_size=16,
        hidden_size=4,
        num_hidden_layers=2,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
        embedding_multiplier=1.0,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(hf_config=hf_config, dtype=dtype),
        cache_config=None,
        quant_config=quant_config,
        parallel_config=SimpleNamespace(use_sequence_parallel_moe=False),
        compilation_config=SimpleNamespace(mode=CompilationMode.NONE),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (_model_vllm_config(quant_config=object()), "unquantized BF16"),
        (_model_vllm_config(dtype=torch.float16), "unquantized BF16"),
    ],
)
def test_apertus2_model_rejects_unsupported_modes(config, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        apertus2.Apertus2Model(vllm_config=config)


@pytest.mark.parametrize(
    ("checkpoint_name", "expected_name", "expected_shard_id"),
    [
        ("self_attn.q_proj.weight", "self_attn.qkv_proj.weight", "q"),
        ("self_attn.k_proj.weight", "self_attn.qkv_proj.weight", "k"),
        ("self_attn.v_proj.weight", "self_attn.qkv_proj.weight", "v"),
        ("self_attn.g_proj.weight", "self_attn.g_proj.weight", None),
        ("mlp.gate_proj.weight", "mlp.gate_up_proj.weight", 0),
        ("mlp.up_proj.weight", "mlp.gate_up_proj.weight", 1),
        (
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.gate_up_proj.weight",
            0,
        ),
        (
            "mlp.shared_experts.up_proj.weight",
            "mlp.shared_experts.gate_up_proj.weight",
            1,
        ),
        ("mlp.experts.7.gate_proj.weight", "mlp.experts.7.gate_proj.weight", None),
        ("mlp.gate.qb_beta", "mlp.gate.qb_beta", None),
    ],
)
def test_apertus2_weight_mapper(
    checkpoint_name: str,
    expected_name: str,
    expected_shard_id: str | int | None,
) -> None:
    checkpoint_weight = torch.empty(1)
    prefix = "model.layers.0."
    [(mapped_name, mapped_weight)] = list(
        apertus2.Apertus2ForCausalLM.hf_to_vllm_mapper.apply(
            [(prefix + checkpoint_name, checkpoint_weight)]
        )
    )

    assert mapped_name == prefix + expected_name
    assert mapped_weight is checkpoint_weight
    assert getattr(mapped_weight, "shard_id", None) == expected_shard_id
