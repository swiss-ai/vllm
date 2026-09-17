# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import re
from collections.abc import Callable, Iterable
from itertools import islice
from types import SimpleNamespace

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.parallel import ParallelConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SSSGLUAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoEFactory, GateLinear
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.kimi_gdn_linear_attn import (
    KimiGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import maybe_disable_graph_partition, set_weight_attrs
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.third_party.flash_linear_attention.ops.kda import FusedRMSNormGated
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    direct_register_custom_op,
)

from .interfaces import HasInnerState, IsHybrid, MambaStateShapes, SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
    sequence_parallel_chunk,
)

logger = init_logger(__name__)


# Canonical QB selection score spaces plus the Megatron estimator spellings
# that map onto them; mirrors hfconverter's normalization.
_QB_METHOD_ALIASES = {
    "average": "sigmoid",
    "histogram": "sigmoid",
    "legacy_average": "legacy",
}


def resolve_quantile_balancing_method(config: PretrainedConfig) -> str:
    """Return the canonical QB selection method ("sigmoid" or "legacy")."""
    raw = getattr(config, "moe_router_quantile_balancing_method", "sigmoid")
    method = _QB_METHOD_ALIASES.get(raw, raw)
    if method not in ("sigmoid", "legacy"):
        raise ValueError(
            "moe_router_quantile_balancing_method must be 'sigmoid' or "
            "'legacy' (Megatron spellings 'average', 'histogram', and "
            f"'legacy_average' are also accepted); got {raw!r}."
        )
    return method


def quantile_balancing_routing_native(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
    sigmoid_selection: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts with QB offsets and return unscaled unbiased weights.

    With ``sigmoid_selection`` experts are chosen from
    ``sigmoid(logits) - qb_beta``; otherwise from raw ``logits - qb_beta``
    (the legacy score space of early QB runs). Mixture weights always come
    from the unbiased sigmoid scores. Exact ties in the selection scores
    follow ``torch.topk`` semantics; expert membership and ordering at the
    tie boundary are otherwise unspecified.
    """
    logits = logits.float()
    gate_scores = torch.sigmoid(logits)
    qb_scores = gate_scores if sigmoid_selection else logits
    selection_scores = qb_scores - qb_beta.float()
    topk_ids = torch.topk(
        selection_scores,
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices

    topk_weights = gate_scores.gather(1, topk_ids)
    if renormalize:
        normalizer = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
        topk_weights = topk_weights / normalizer

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


@torch.compile(
    dynamic=True,
    fullgraph=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def quantile_balancing_routing(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
    sigmoid_selection: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run quantile-balancing routing as one compiled tensor region."""
    return quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k,
        renormalize,
        sigmoid_selection,
    )


def merge_branch_with_residual(
    residual_stream: torch.Tensor,
    branch_output: torch.Tensor,
    residual_multiplier: float,
    post_norm: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> torch.Tensor:
    """Apply an optional post-norm before merging a branch into its residual."""
    if post_norm is not None:
        branch_output = post_norm(branch_output)
    return residual_stream + residual_multiplier * branch_output


class Apertus2MLP(nn.Module):
    """Tensor-parallel dense SSSGLU feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        prefix: str = "",
        is_sequence_parallel: bool = False,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "sssglu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only sssglu is supported."
            )
        self.act_fn = SSSGLUAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Apertus2MoE(nn.Module):
    """Apertus 2 latent routed experts with a full-width shared expert."""

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not getattr(config, "use_quantile_balancing", False):
            raise ValueError("Apertus2MoE currently supports only QB routing.")
        self.qb_sigmoid_selection = (
            resolve_quantile_balancing_method(config) == "sigmoid"
        )

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        latent_size = getattr(config, "moe_latent_size", None)
        self.moe_hidden_size = latent_size or self.hidden_size

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        self.gate = GateLinear(
            input_size=self.hidden_size,
            output_size=self.n_routed_experts,
            bias=False,
            out_dtype=torch.float32,
            force_fp32_compute=True,
            prefix=f"{prefix}.gate",
        )
        self.gate.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )
        self.gate.register_buffer(
            "qb_beta",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

        if self.n_shared_experts:
            self.shared_experts = Apertus2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * self.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=False,
                reduce_results=False,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_experts",
            )
        else:
            self.shared_experts = None

        if latent_size:
            self.latent_down_proj = ReplicatedLinear(
                input_size=self.hidden_size,
                output_size=self.moe_hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.latent_down_proj",
            )
            self.latent_up_proj = ReplicatedLinear(
                input_size=self.moe_hidden_size,
                output_size=self.hidden_size,
                bias=False,
                quant_config=quant_config,
                prefix=f"{prefix}.latent_up_proj",
            )
        else:
            self.latent_down_proj = None
            self.latent_up_proj = None

        self.experts = FusedMoEFactory(
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=self.moe_hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            custom_routing_function=self._qb_routing,
            activation="sssglu",
            shared_experts=self.shared_experts,
            routed_input_transform=self.latent_down_proj,
            routed_output_transform=self.latent_up_proj,
            routed_scaling_factor=config.routed_scaling_factor,
            is_sequence_parallel=self.is_sequence_parallel,
            apply_routed_scale_to_output=True,
            router_logits_dtype=torch.float32,
            prefix=f"{prefix}.experts",
        )

    def _qb_routing(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del hidden_states
        return quantile_balancing_routing(
            logits=gating_output,
            qb_beta=self.gate.qb_beta,
            top_k=topk,
            renormalize=renormalize,
            sigmoid_selection=self.qb_sigmoid_selection,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        router_logits, _ = self.gate(hidden_states)
        output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if self.is_sequence_parallel:
            output = tensor_model_parallel_all_gather(output, 0)
            output = output[:num_tokens]

        return output.view(num_tokens, hidden_dim)


class Apertus2Attention(nn.Module):
    """Apertus 2 attention with independent RoPE and window schedules.

    With ``attention_output_gate``, an extra head-sharded projection of the
    block input gates the attention output channelwise before ``o_proj``.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        no_rope_layers = getattr(config, "no_rope_layers", None)
        self.use_rope = no_rope_layers is None or no_rope_layers[layer_idx] == 1

        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = self.hidden_size // self.total_num_heads
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias_o_proj,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Per-channel sigmoid gate on the attention output, projected from the
        # same normalized input as Q/K/V. Rows are in global head order, so
        # column-parallel sharding aligns each rank's gate channels with its
        # local query heads. Never q-normalized or rotated.
        self.attention_output_gate = getattr(config, "attention_output_gate", False)
        if self.attention_output_gate:
            self.g_proj = ColumnParallelLinear(
                input_size=hidden_size,
                output_size=self.total_num_heads * self.head_dim,
                bias=bias,
                gather_output=False,
                quant_config=quant_config,
                prefix=f"{prefix}.g_proj",
            )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=True,
        )

        sliding_window = None
        layer_types = getattr(config, "layer_types", None)
        if layer_types and layer_types[layer_idx] == "sliding_attention":
            sliding_window = config.sliding_window

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view_as(q)
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view_as(k)
        if self.use_rope:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        if self.attention_output_gate:
            gate, _ = self.g_proj(hidden_states)
            attn_output = (attn_output * torch.sigmoid(gate.float())).to(
                attn_output.dtype
            )
        output, _ = self.o_proj(attn_output)
        return output


def kda_layer_indices(config: PretrainedConfig) -> list[int]:
    """Return the 0-indexed layers whose ``layer_types`` entry is KDA."""
    layer_types = getattr(config, "layer_types", None) or []
    return [i for i, t in enumerate(layer_types) if t == "linear_attention"]


def kda_geometry(config: PretrainedConfig) -> tuple[int, int, int]:
    """Return ``(num_heads, head_dim, conv_kernel_size)`` of the KDA layers.

    hfconverter writes the geometry as flat Qwen3-Next-style fields and keeps
    key and value heads (and their head dims) equal, which is what the shared
    Kimi layer assumes. Raise instead of letting a missing field surface as
    ``TypeError: None`` from the hybrid cache-shape pre-pass.
    """
    if not kda_layer_indices(config):
        raise ValueError(
            "Apertus2KDAForCausalLM requires at least one 'linear_attention' "
            "entry in layer_types; pure-softmax checkpoints must keep "
            "architectures=['Apertus2ForCausalLM']."
        )
    fields = {
        name: getattr(config, name, None)
        for name in (
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
        )
    }
    missing = [name for name, value in fields.items() if value is None]
    if missing:
        raise ValueError(
            f"layer_types has 'linear_attention' layers but {missing} are unset."
        )
    if fields["linear_num_key_heads"] != fields["linear_num_value_heads"]:
        raise ValueError(
            "KDA needs linear_num_key_heads == linear_num_value_heads, got "
            f"{fields['linear_num_key_heads']} != {fields['linear_num_value_heads']}."
        )
    if fields["linear_key_head_dim"] != fields["linear_value_head_dim"]:
        raise ValueError(
            "KDA needs linear_key_head_dim == linear_value_head_dim, got "
            f"{fields['linear_key_head_dim']} != {fields['linear_value_head_dim']}."
        )
    return (
        fields["linear_num_value_heads"],
        fields["linear_value_head_dim"],
        fields["linear_conv_kernel_dim"],
    )


class Apertus2KDAAttention(KimiGatedDeltaNetAttention):
    """Kimi Delta Attention layer fed by the flat Apertus 2 config.

    The shared Kimi layer supplies projections, short convolutions, the KDA
    kernels, the gated output norm and the GDN state cache. This subclass only
    adapts the contract hfconverter freezes on disk:

    * geometry comes from the flat ``linear_*`` fields instead of Kimi's
      ``linear_attn_config`` dict; ``gate_lower_bound`` selects the bounded
      decay ``g = lb * sigmoid(exp(A_log) * (alpha + dt_bias))`` (``None``
      means the unbounded softplus decay);
    * the low-rank output gate ``g_b_proj`` carries a trained bias
      (``linear_attn_output_gate_bias``), which the shared layer omits;
    * the gated output norm uses the model's ``rms_norm_eps``;
    * the packed conv1d parameter learns to take its shard id from the tensor
      attribute that ``load_weights`` sets, because ``AutoWeightsLoader`` calls
      parameter loaders without a positional shard id.

    The KDA core runs as the ``vllm::apertus2_kda_attention_core`` custom op so
    ``Apertus2Model`` keeps its piecewise torch.compile path: the op is a
    splitting op, exactly like the Qwen3-Next GDN core. Checkpoint keys live
    under ``self_attn.`` with Kimi-Linear spellings; ``Apertus2KDAForCausalLM``
    routes them.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        vllm_config: VllmConfig,
        prefix: str = "",
    ) -> None:
        num_heads, head_dim, conv_kernel_size = kda_geometry(config)
        gate_lower_bound = getattr(config, "gate_lower_bound", None)
        kimi_view = SimpleNamespace(
            hidden_size=config.hidden_size,
            # The conv activation; Apertus2's ``hidden_act`` names the MLP.
            hidden_act="silu",
            rms_norm_eps=config.rms_norm_eps,
            linear_attn_config={
                "num_heads": num_heads,
                "head_dim": head_dim,
                "short_conv_kernel_size": conv_kernel_size,
                "gate_lower_bound": gate_lower_bound,
            },
        )
        super().__init__(kimi_view, vllm_config, prefix)

        self.output_gate_bias = bool(
            getattr(config, "linear_attn_output_gate_bias", True)
        )
        if self.output_gate_bias:
            self.g_b_proj = ColumnParallelLinear(
                self.head_dim,
                self.projection_size,
                bias=True,
                quant_config=self.quant_config,
                prefix=f"{prefix}.g_b_proj",
            )
        self.o_norm = FusedRMSNormGated(
            self.head_dim, eps=config.rms_norm_eps, activation="sigmoid"
        )

        fused_conv_loader = self.conv1d.weight.weight_loader

        def conv1d_weight_loader(
            param: torch.Tensor,
            loaded_weight: torch.Tensor,
            loaded_shard_id: int | None = None,
        ) -> None:
            if loaded_shard_id is None:
                loaded_shard_id = getattr(loaded_weight, "shard_id", None)
            if loaded_shard_id is None:
                raise ValueError(
                    f"{prefix}.conv1d expects a q/k/v shard id; got a weight "
                    "without one."
                )
            fused_conv_loader(param, loaded_weight, loaded_shard_id)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(self.conv1d.weight, {"weight_loader": conv1d_weight_loader})

    def forward(  # type: ignore[override]
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        del positions  # KDA layers carry no positional signal.
        num_tokens = hidden_states.size(0)
        projected, _ = self.in_proj_qkvgfab(hidden_states)
        mixed_qkv, beta, f_a = projected.split(
            [3 * self.local_projection_size, self.local_num_heads, self.head_dim],
            dim=-1,
        )
        g1 = self.f_b_proj(f_a)[0].view(
            1, num_tokens, self.local_num_heads, self.head_dim
        )
        g2 = self.g_b_proj(self.g_a_proj(hidden_states)[0])[0].view(
            num_tokens, self.local_num_heads, self.head_dim
        )
        beta = beta.unsqueeze(0)
        core_attn_out = torch.empty(
            (1, num_tokens, self.local_num_heads, self.head_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        torch.ops.vllm.apertus2_kda_attention_core(
            mixed_qkv,
            g1,
            g2,
            beta,
            core_attn_out,
            layer_name=_encode_layer_name(self.prefix),
        )
        output, _ = self.o_proj(
            core_attn_out.view(num_tokens, self.local_projection_size)
        )
        return output


def apertus2_kda_attention_core(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the shared Kimi KDA core (conv, chunk/recurrent KDA, gated norm).

    ``core_attn_out`` is written in place; the layer's conv and recurrent
    state caches are updated through the layer object itself.
    """
    layer_name = _resolve_layer_name(layer_name)
    forward_context: ForwardContext = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    self._forward(
        mixed_qkv=mixed_qkv,
        g1=g1,
        g2=g2,
        beta=beta,
        core_attn_out=core_attn_out,
    )


def apertus2_kda_attention_core_fake(
    mixed_qkv: torch.Tensor,
    g1: torch.Tensor,
    g2: torch.Tensor,
    beta: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    return


direct_register_custom_op(
    op_name="apertus2_kda_attention_core",
    op_func=apertus2_kda_attention_core,
    mutates_args=["core_attn_out"],
    fake_impl=apertus2_kda_attention_core_fake,
)


class Apertus2DecoderLayer(nn.Module):
    """Apertus 2 attention followed by a scheduled dense or MoE block."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        parallel_config: ParallelConfig | None = None,
        prefix: str = "",
        vllm_config: VllmConfig | None = None,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        # The module is ``self_attn`` for both kinds: hfconverter stores KDA
        # weights under the same parent as softmax attention.
        if layer_idx in kda_layer_indices(config):
            if vllm_config is None:
                raise ValueError(
                    f"{prefix} is a linear_attention layer and needs vllm_config."
                )
            self.self_attn = Apertus2KDAAttention(
                config=config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            self.self_attn = Apertus2Attention(
                config=config,
                hidden_size=self.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=getattr(
                    config, "num_key_value_heads", config.num_attention_heads
                ),
                max_position_embeddings=max_position_embeddings,
                quant_config=quant_config,
                bias=getattr(config, "attention_bias", False),
                bias_o_proj=False,
                cache_config=cache_config,
                prefix=f"{prefix}.self_attn",
            )

        if config.is_moe_layer(layer_idx):
            self.mlp = Apertus2MoE(
                config=config,
                quant_config=quant_config,
                parallel_config=parallel_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Apertus2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                bias=getattr(config, "mlp_bias", False),
                prefix=f"{prefix}.mlp",
            )

        self.attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.sandwich_norm:
            self.post_attention_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
            self.post_feedforward_layernorm = RMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        residual_stream = hidden_states
        attention_input = self.attention_layernorm(hidden_states)
        attention_output = self.self_attn(
            positions=positions,
            hidden_states=attention_input,
        )
        hidden_states = merge_branch_with_residual(
            residual_stream=residual_stream,
            branch_output=attention_output,
            residual_multiplier=self.residual_multiplier,
            post_norm=getattr(self, "post_attention_layernorm", None),
        )

        residual_stream = hidden_states
        feedforward_input = self.feedforward_layernorm(hidden_states)
        feedforward_output = self.mlp(feedforward_input)
        return merge_branch_with_residual(
            residual_stream=residual_stream,
            branch_output=feedforward_output,
            residual_multiplier=self.residual_multiplier,
            post_norm=getattr(self, "post_feedforward_layernorm", None),
        )


@support_torch_compile
class Apertus2Model(nn.Module):
    """Pipeline-parallel Apertus 2 decoder backbone."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = Apertus2DecoderLayer,
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        parallel_config = vllm_config.parallel_config
        if quant_config is not None:
            raise ValueError(
                "Apertus2 currently supports only unquantized BF16 inference."
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "Apertus2 currently supports only unquantized BF16 inference."
            )

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.embedding_multiplier = config.embedding_multiplier

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                parallel_config=parallel_config,
                prefix=prefix,
                vllm_config=vllm_config,
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states * self.embedding_multiplier
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states = layer(positions, hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        return self.norm(hidden_states)


class Apertus2ForCausalLM(nn.Module, SupportsPP):
    """Minimal native vLLM wrapper for Apertus 2 causal generation."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_experts.gate_proj": (
                ".shared_experts.gate_up_proj",
                0,
            ),
            ".shared_experts.up_proj": (
                ".shared_experts.gate_up_proj",
                1,
            ),
        }
    )

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = Apertus2DecoderLayer,
    ) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        if kda_layer_indices(config) and not getattr(type(self), "is_hybrid", False):
            # Only the hybrid subclass sizes the GDN state cache; building KDA
            # layers here would fail later with an opaque mamba_block_size
            # assertion. Exports made before hfconverter emitted the KDA
            # architecture name land here.
            raise ValueError(
                "This checkpoint has linear_attention (KDA) layers; set "
                "config.json architectures to ['Apertus2KDAForCausalLM'] "
                f"instead of ['{type(self).__name__}']."
            )

        self.model = self._init_model(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            layer_type=layer_type,
        )

        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            if config.tie_word_embeddings:
                self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
            self.logits_processor = LogitsProcessor(config.vocab_size)
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = Apertus2DecoderLayer,
    ) -> Apertus2Model:
        return Apertus2Model(
            vllm_config=vllm_config,
            prefix=prefix,
            layer_type=layer_type,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


# Checkpoint leaf -> (merged vLLM parameter, shard id) for KDA layers. The
# remaining KDA leaves (f_b_proj, g_a_proj, g_b_proj(+bias), A_log, dt_bias,
# o_norm, o_proj) already share their names with the vLLM modules.
_KDA_STACKED_LEAVES: dict[str, tuple[str, int]] = {
    "q_proj": ("in_proj_qkvgfab", 0),
    "k_proj": ("in_proj_qkvgfab", 1),
    "v_proj": ("in_proj_qkvgfab", 2),
    "b_proj": ("in_proj_qkvgfab", 3),
    "f_a_proj": ("in_proj_qkvgfab", 4),
    "q_conv1d": ("conv1d", 0),
    "k_conv1d": ("conv1d", 1),
    "v_conv1d": ("conv1d", 2),
}

_SELF_ATTN_LEAF = re.compile(
    r"^(?P<parent>(?:.*\.)?layers\.(?P<layer>\d+)\.self_attn\.)"
    r"(?P<leaf>\w+)\.(?P<kind>weight|bias)$"
)


def remap_kda_checkpoint_keys(
    weights: Iterable[tuple[str, torch.Tensor]],
    kda_layers: set[int],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Route KDA-layer ``self_attn`` leaves to their merged vLLM parameters.

    ``hf_to_vllm_mapper`` is name-based, so ``.q_proj`` in a KDA layer would
    otherwise be sent to the softmax ``qkv_proj``. Renaming here, before the
    mapper runs, keeps that mapper untouched: the new names contain none of
    its substrings, and the mapper only rewrites ``shard_id`` on its own
    matches. Merged-linear loaders read the shard id from the tensor.
    """
    for name, weight in weights:
        match = _SELF_ATTN_LEAF.match(name)
        if (
            match is not None
            and int(match["layer"]) in kda_layers
            and match["leaf"] in _KDA_STACKED_LEAVES
        ):
            target, shard_id = _KDA_STACKED_LEAVES[match["leaf"]]
            weight.shard_id = shard_id
            name = f"{match['parent']}{target}.{match['kind']}"
        yield name, weight


class Apertus2KDAForCausalLM(Apertus2ForCausalLM, HasInnerState, IsHybrid):
    """Apertus 2 with Kimi Delta Attention layers (``layer_types`` mixes
    ``linear_attention`` with ``full_attention``).

    Kept as a separate architecture so ``is_hybrid`` (a class property) never
    changes cache setup for the existing pure-softmax exports. hfconverter
    emits this name whenever ``layer_types`` contains ``linear_attention``.
    """

    packed_modules_mapping = {
        **Apertus2ForCausalLM.packed_modules_mapping,
        # q/k/v_proj appear in both qkv_proj and in_proj_qkvgfab; the two are
        # told apart by layer index in load_weights. Unquantized BF16 (all
        # this model supports) never consults this mapping.
        "in_proj_qkvgfab": ["q_proj", "k_proj", "v_proj", "b_proj", "f_a_proj"],
        "conv1d": ["q_conv1d", "k_conv1d", "v_conv1d"],
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = Apertus2DecoderLayer,
    ) -> None:
        config = vllm_config.model_config.hf_config
        num_heads, head_dim, conv_kernel_size = kda_geometry(config)
        super().__init__(vllm_config=vllm_config, prefix=prefix, layer_type=layer_type)
        gate_lower_bound = getattr(config, "gate_lower_bound", None)
        logger.info_once(
            "Apertus2 KDA: %d linear_attention layers, %d heads x %d, conv %d, "
            "decay gate %s, output gate bias %s",
            len(kda_layer_indices(config)),
            num_heads,
            head_dim,
            conv_kernel_size,
            (
                f"{gate_lower_bound} * sigmoid(exp(A_log) * (alpha + dt_bias))"
                if gate_lower_bound is not None
                else "-exp(A_log) * softplus(alpha + dt_bias) (gate_lower_bound unset)"
            ),
            bool(getattr(config, "linear_attn_output_gate_bias", True)),
        )

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.kda_state_dtype(
            vllm_config.model_config.dtype, vllm_config.cache_config.mamba_cache_dtype
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls,
        vllm_config: VllmConfig,
    ) -> MambaStateShapes:
        num_heads, head_dim, conv_kernel_size = kda_geometry(
            vllm_config.model_config.hf_config
        )
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.kda_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            num_heads,
            head_dim,
            conv_kernel_size=conv_kernel_size,
            num_spec=num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(
        cls,
    ) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.kda_state_copy_func()

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        weights = remap_kda_checkpoint_keys(
            weights, set(kda_layer_indices(self.config))
        )
        return super().load_weights(weights)
