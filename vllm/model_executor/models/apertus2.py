# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Iterable
from itertools import islice

import torch
from torch import nn
from transformers.configuration_utils import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
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
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsPP
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


def quantile_balancing_routing_native(
    logits: torch.Tensor,
    qb_beta: torch.Tensor,
    top_k: int,
    renormalize: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select experts with QB offsets and return unscaled unbiased weights.

    Exact ties in ``logits - qb_beta`` follow ``torch.topk`` semantics; expert
    membership and ordering at the tie boundary are otherwise unspecified.
    """
    logits = logits.float()
    selection_scores = logits - qb_beta.float()
    topk_ids = torch.topk(
        selection_scores,
        k=top_k,
        dim=-1,
        sorted=False,
    ).indices

    topk_weights = torch.sigmoid(logits).gather(1, topk_ids)
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
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run quantile-balancing routing as one compiled tensor region."""
    return quantile_balancing_routing_native(
        logits,
        qb_beta,
        top_k,
        renormalize,
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
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size, intermediate_size],
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
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
        prefix: str = "",
    ) -> None:
        super().__init__()
        if not getattr(config, "use_quantile_balancing", False):
            raise ValueError("Apertus2MoE currently supports only QB routing.")

        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        latent_size = getattr(config, "moe_latent_size", None)
        self.moe_hidden_size = latent_size or self.hidden_size

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
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits, _ = self.gate(hidden_states)
        output = self.experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        return output.reshape(original_shape)


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


class Apertus2DecoderLayer(nn.Module):
    """Apertus 2 attention followed by a scheduled dense or MoE block."""

    def __init__(
        self,
        config: PretrainedConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = config.hidden_size
        self.residual_multiplier = config.residual_multiplier
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

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
        if quant_config is not None:
            raise ValueError(
                "Apertus2 currently supports only unquantized BF16 inference."
            )
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError(
                "Apertus2 currently supports only unquantized BF16 inference."
            )
        if vllm_config.parallel_config.use_sequence_parallel_moe:
            raise ValueError(
                "Apertus2 sequence-parallel MoE support is deferred to Phase 3."
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
                prefix=prefix,
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
        loader = AutoWeightsLoader(self)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
