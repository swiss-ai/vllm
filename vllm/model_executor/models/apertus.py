# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2025 The Swiss AI Initiative.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate the architectural differences made by
# the Swiss AI Initiative that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Apertus model compatible with HuggingFace weights."""

from collections.abc import Iterable, Mapping, Sequence
from itertools import islice

import torch
from torch import nn
from transformers import ApertusConfig, BatchFeature

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.activation import XIELU
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
)
from vllm.multimodal.parse import ImageProcessorItems, MultiModalDataItems
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptReplacement,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.v1.attention.backend import AttentionType

from .apertus_utils import ApertusImageTokenizer
from .interfaces import (
    EagleModelMixin,
    MultiModalEmbeddings,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsMultiModal,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)


class ApertusProcessingInfo(BaseProcessingInfo):
    def get_hf_config(self) -> ApertusConfig:
        return self.ctx.get_hf_config(ApertusConfig)

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None}


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_images = mm_counts.get("image", 0)
        return ApertusImageTokenizer.DEFAULT_IMAGE_PLACEHOLDER * num_images

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        del seq_len
        num_images = mm_counts.get("image", 0)
        image_overrides = mm_options.get("image")
        max_side = int(ApertusImageTokenizer.DEFAULT_MAX_PIXELS**0.5)
        return {
            "image": self._get_dummy_images(
                width=max_side,
                height=max_side,
                num_images=num_images,
                overrides=image_overrides,
            )
        }


class ApertusMultiModalProcessor(BaseMultiModalProcessor[ApertusProcessingInfo]):
    def __init__(
        self,
        info: ApertusProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[ApertusProcessingInfo],
        *,
        cache: object | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        self.image_tokenizer = ApertusImageTokenizer()

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return {}

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        return []

    @staticmethod
    def _find_image_placeholders(prompt: str, aliases: Sequence[str]) -> list[str]:
        placeholders: list[tuple[int, str]] = []
        for alias in aliases:
            start = 0
            while True:
                idx = prompt.find(alias, start)
                if idx < 0:
                    break
                placeholders.append((idx, alias))
                start = idx + len(alias)

        placeholders.sort(key=lambda item: item[0])
        return [placeholder for _, placeholder in placeholders]

    def _validate_only_image_inputs(self, mm_items: MultiModalDataItems) -> None:
        unsupported = [modality for modality in mm_items if modality != "image"]
        if unsupported:
            raise ValueError(
                "Apertus multimodal preprocessing currently supports only "
                f"image inputs. Unsupported modalities: {unsupported}"
            )

    def _tokenize_text(
        self,
        text: str,
        tokenization_kwargs: Mapping[str, object],
    ) -> list[int]:
        tokenizer = self.info.get_tokenizer()
        token_ids = tokenizer.encode(text, **dict(tokenization_kwargs))
        return list(token_ids)

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ) -> MultiModalInput:
        self._validate_only_image_inputs(inputs.mm_data_items)

        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt
            if isinstance(inputs.prompt, str)
            else tokenizer.decode(inputs.prompt)
        )

        if inputs.mm_data_items.get_count("image", strict=False) == 0:
            with timing_ctx.record("tokenize"):
                prompt_token_ids = self._tokenize_text(
                    prompt_text,
                    inputs.tokenization_kwargs,
                )
            return mm_input(
                prompt_token_ids=prompt_token_ids,
                mm_kwargs=MultiModalKwargsItems({}),
                mm_hashes={},
                mm_placeholders={},
                prompt=prompt_text,
            )

        with timing_ctx.record("encode_apertus_images"):
            image_items = inputs.mm_data_items.get_items("image", ImageProcessorItems)
            images = [image_items[idx] for idx in range(len(image_items))]
            image_prompts = self.image_tokenizer.encode_images(
                images,
                tokenizer=tokenizer,
                mm_processor_kwargs=inputs.hf_processor_mm_kwargs,
            )

        aliases = self.image_tokenizer.placeholder_aliases(
            tokenizer, inputs.hf_processor_mm_kwargs
        )
        placeholders = self._find_image_placeholders(prompt_text, aliases)
        if len(placeholders) != len(image_prompts):
            raise ValueError(
                "Apertus image placeholder/input mismatch: found "
                f"{len(placeholders)} placeholder(s) in the prompt using aliases "
                f"{aliases}, but received {len(image_prompts)} image input(s)."
            )

        prompt_updates = self._bind_and_group_updates(
            [
                PromptReplacement(
                    modality="image",
                    target=lambda item_idx: placeholders[item_idx],
                    replacement=lambda item_idx: image_prompts[item_idx],
                )
            ],
            {"image": len(image_prompts)},
        )

        with timing_ctx.record("apply_prompt_updates"):
            merged_prompt, match_result = self._apply_text_matches(
                prompt_text, prompt_updates
            )

        if not all(
            update_idx is not None
            for update_idxs in match_result.values()
            for update_idx in update_idxs
        ):
            raise RuntimeError("Failed to replace all Apertus image placeholders.")

        with timing_ctx.record("tokenize"):
            prompt_token_ids = self._tokenize_text(
                merged_prompt,
                inputs.tokenization_kwargs,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems({}),
            mm_hashes={},
            mm_placeholders={},
            prompt=merged_prompt,
        )


class ApertusMLP(nn.Module):
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
        self.up_proj = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=intermediate_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "xielu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only xIELU is supported for now."
            )
        self.act_fn = XIELU()

    def forward(self, x):
        x, _ = self.up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class ApertusAttention(nn.Module):
    def __init__(
        self,
        config: ApertusConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position_embeddings: int = 8192,
        quant_config: QuantizationConfig | None = None,
        bias: bool = False,
        bias_o_proj: bool = False,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
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

        self._init_rotary_emb(config, quant_config=quant_config)

        sliding_window = None
        if layer_types := getattr(config, "layer_types", None):
            is_sliding = layer_types[layer_idx] == "sliding_attention"
            if is_sliding:
                sliding_window = config.sliding_window

        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )

        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            attn_type=attn_type,
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
        q = self.q_norm(q.contiguous().view(-1, self.head_dim)).view_as(q)
        k = self.k_norm(k.contiguous().view(-1, self.head_dim)).view_as(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output

    def _init_rotary_emb(
        self,
        config: ApertusConfig,
        quant_config: QuantizationConfig | None,
    ) -> None:
        is_neox_style = True

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            is_neox_style=is_neox_style,
        )


class ApertusDecoderLayer(nn.Module):
    def __init__(
        self,
        config: ApertusConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        bias_o_proj = attention_bias
        # support internlm/internlm3-8b with qkv_bias
        if hasattr(config, "qkv_bias"):
            attention_bias = config.qkv_bias

        # Apertus defaults to causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. parasail-ai/GritLM-7B-vllm)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = ApertusAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            bias_o_proj=bias_o_proj,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
        )
        self.mlp = ApertusMLP(
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

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.attention_layernorm(hidden_states)
        else:
            hidden_states, residual = self.attention_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

        # Fully Connected
        hidden_states, residual = self.feedforward_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


@support_torch_compile
class ApertusModel(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusDecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.config = config
        self.quant_config = quant_config

        self.vocab_size = config.vocab_size

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
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states
        return hidden_states


@MULTIMODAL_REGISTRY.register_processor(
    ApertusMultiModalProcessor,
    info=ApertusProcessingInfo,
    dummy_inputs=ApertusDummyInputsBuilder,
)
class ApertusForCausalLM(
    nn.Module,
    SupportsLoRA,
    SupportsPP,
    SupportsEagle,
    SupportsEagle3,
    SupportsMultiModal,
):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
        }
    )
    packed_modules_mapping = {"qkv_proj": ["q_proj", "k_proj", "v_proj"]}

    # LoRA specific attributes
    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusDecoderLayer,
    ):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        self.config = config

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

            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                config.vocab_size, scale=logit_scale
            )
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def _init_model(
        self,
        vllm_config: VllmConfig,
        prefix: str = "",
        layer_type: type[nn.Module] = ApertusDecoderLayer,
    ):
        return ApertusModel(
            vllm_config=vllm_config, prefix=prefix, layer_type=layer_type
        )

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality == "image":
            return ApertusImageTokenizer.DEFAULT_IMAGE_PLACEHOLDER

        raise ValueError(f"Unsupported modality: {modality}")

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        if kwargs:
            raise ValueError(
                "Apertus image inputs are serialized to token IDs during "
                "preprocessing and should not reach the model as multimodal "
                f"kwargs. Got keys: {sorted(kwargs)}"
            )
        return []

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if multimodal_embeddings is not None and len(multimodal_embeddings) > 0:
            raise ValueError(
                "Apertus does not merge multimodal embeddings in the model. "
                "Images must be serialized to token IDs by the processor."
            )
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        if kwargs:
            raise ValueError(
                "Unexpected multimodal kwargs for Apertus forward: "
                f"{sorted(kwargs)}"
            )
        model_output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return model_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
