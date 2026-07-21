# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multimodal Apertus 1.5 pipeline optimized for native vLLM asynchronous execution.

Architecture Contract:
1. Processor (CPU): HuggingFace-compatible media preprocessing and prompt
   expansion. Zero neural network inference is executed here.
2. Worker (GPU): Native Emu3.5/WavTokenizer execution and O(1) boolean mask
   ID substitution.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from math import isqrt
from typing import Any

import torch
import torch.nn.functional as F
from transformers import Apertus1p5VisionTokenizerModel, AutoConfig, AutoModel

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader import DefaultModelLoader
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (
    MultiModalFieldConfig,
    MultiModalKwargsItems,
    PlaceholderRange,
)
from vllm.multimodal.parse import (
    AudioProcessorItems,
    ImageProcessorItems,
    MultiModalDataParser,
)
from vllm.multimodal.processing import (
    BaseDummyInputsBuilder,
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    ProcessorInputs,
    PromptUpdate,
    TimingContext,
)
from vllm.sequence import IntermediateTensors
from vllm.utils.torch_utils import set_default_torch_dtype

from .apertus import ApertusForCausalLM
from .interfaces import MultiModalEmbeddings, SupportsMultiModal
from .utils import AutoWeightsLoader, WeightsMapper, maybe_prefix

_IMAGE_TOKEN_BUDGET_OVERHEAD = 512
_MAX_AUDIO_SECONDS = 300
_AUDIO_TOKEN_BUDGET_OVERHEAD = 4

_DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
_DEFAULT_BOI_TOKEN = "<|img_start|>"
_DEFAULT_IMG_TOKEN = "<|img_token_start|>"
_DEFAULT_EOL_TOKEN = "<|img_end_of_row|>"
_DEFAULT_EOI_TOKEN = "<|img_end|>"
_DEFAULT_AUDIO_PLACEHOLDER = "<|audio|>"
_DEFAULT_AUDIO_START_TOKEN = "<|audio_start|>"
_DEFAULT_AUDIO_END_TOKEN = "<|audio_end|>"

_DEFAULT_IMAGE_TOKEN_ID = 131079
_DEFAULT_AUDIO_TOKEN_ID = 131085
_DEFAULT_IMAGE_TOKEN_OFFSET = 131272
_DEFAULT_AUDIO_TOKEN_OFFSET = 262344
_DEFAULT_IMAGE_START_TOKEN_ID = 131073
_DEFAULT_IMAGE_END_TOKEN_ID = 131074
_DEFAULT_AUDIO_START_TOKEN_ID = 131080
_DEFAULT_AUDIO_END_TOKEN_ID = 131081


def _pad_logits_to_input_vocab(
    logits: torch.Tensor, input_vocab_size: int
) -> torch.Tensor:
    return F.pad(
        logits,
        (0, input_vocab_size - logits.shape[-1]),
        value=float("-inf"),
    )


def _init_component_model(
    component_config: Mapping[str, Any],
    model_cls: type[torch.nn.Module] | None = None,
) -> torch.nn.Module:
    config_dict = dict(component_config)
    config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
    return AutoModel.from_config(config) if model_cls is None else model_cls(config)


class Apertus1p5ImageTokenizer:
    def __init__(self, tokenizer: Any | None = None) -> None:
        self.image_placeholder = getattr(
            tokenizer, "image_token", _DEFAULT_IMAGE_PLACEHOLDER
        )
        self.boi_token = getattr(tokenizer, "boi_token", _DEFAULT_BOI_TOKEN)
        self.img_token = getattr(tokenizer, "image_wrapper_token", _DEFAULT_IMG_TOKEN)
        self.eol_token = getattr(tokenizer, "eol_token", _DEFAULT_EOL_TOKEN)
        self.eoi_token = getattr(tokenizer, "eoi_token", _DEFAULT_EOI_TOKEN)


class Apertus1p5AudioTokenizer:
    def __init__(self, tokenizer: Any | None = None) -> None:
        self.audio_placeholder = getattr(
            tokenizer, "audio_token", _DEFAULT_AUDIO_PLACEHOLDER
        )
        self.audio_start_token = getattr(
            tokenizer, "audio_start_token", _DEFAULT_AUDIO_START_TOKEN
        )
        self.audio_end_token = getattr(
            tokenizer, "audio_end_token", _DEFAULT_AUDIO_END_TOKEN
        )


class Apertus1p5ProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        feature_extractor = self.get_hf_processor().feature_extractor
        return MultiModalDataParser(
            target_sr=feature_extractor.sampling_rate,
            target_channels=1,
            expected_hidden_size=self._get_expected_hidden_size(),
        )

    def get_default_tok_params(self):
        """The Apertus chat template renders ``{{ bos_token }}`` itself, so the
        template owns BOS: when the tokenizer carries a chat template, default
        tokenization must not add special tokens, or every offline
        ``LLM.chat()`` prompt starts with a double BOS (``[1, 1, ...]``) --
        a sequence the model was not trained on (apertus-program #420).
        Base checkpoints (no chat template) keep the default so raw prompts
        still get their BOS. Same pattern as vllm-project/vllm#39842 (Gemma 4)
        and the ovis/ultravox/paligemma overrides.
        """
        tokenizer = self.ctx.get_tokenizer()
        has_chat_template = getattr(tokenizer, "chat_template", None) is not None

        params = super().get_default_tok_params()
        if has_chat_template:
            params = params.with_kwargs(add_special_tokens=False)
        return params

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "audio": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        del mm_counts
        processor = self.get_hf_processor()
        image_processor = processor.image_processor
        feature_extractor = processor.feature_extractor
        return {
            # Maximum image codes plus room for the image layout's wrapper
            # and row-separator tokens.
            "image": min(
                (image_processor.max_pixels // (image_processor.spatial_factor**2))
                + _IMAGE_TOKEN_BUDGET_OVERHEAD,
                seq_len,
            ),
            # Maximum audio codes for the configured duration plus special tokens.
            "audio": min(
                (
                    feature_extractor.get_num_audio_codes(
                        feature_extractor.sampling_rate
                    )
                    * _MAX_AUDIO_SECONDS
                )
                + _AUDIO_TOKEN_BUDGET_OVERHEAD,
                seq_len,
            ),
        }


class Apertus1p5DummyInputsBuilder(BaseDummyInputsBuilder[Apertus1p5ProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        tokenizer = self.info.get_tokenizer()
        image_placeholder = getattr(
            tokenizer, "image_token", _DEFAULT_IMAGE_PLACEHOLDER
        )
        audio_placeholder = getattr(
            tokenizer, "audio_token", _DEFAULT_AUDIO_PLACEHOLDER
        )
        return image_placeholder * mm_counts.get(
            "image", 0
        ) + audio_placeholder * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        image_overrides = mm_options.get("image")
        audio_overrides = mm_options.get("audio")
        processor = self.info.get_hf_processor()
        max_image_side = isqrt(processor.image_processor.max_pixels)

        return {
            "image": self._get_dummy_images(
                width=max_image_side,
                height=max_image_side,
                num_images=mm_counts.get("image", 0),
                overrides=image_overrides,
            ),
            "audio": self._get_dummy_audios(
                length=processor.feature_extractor.sampling_rate * _MAX_AUDIO_SECONDS,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_overrides,
            ),
        }


class Apertus1p5MultiModalProcessor(
    BaseMultiModalProcessor[Apertus1p5ProcessingInfo]
):
    """CPU-bound API Processor. Strict YAGNI Rule: NO heavy neural networks run here."""

    def __init__(
        self,
        info: Apertus1p5ProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder,
        *,
        cache: object | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        tokenizer = info.get_tokenizer()
        self.hf_processor = info.get_hf_processor()
        self.image_tokenizer = Apertus1p5ImageTokenizer(tokenizer)
        self.audio_tokenizer = Apertus1p5AudioTokenizer(tokenizer)
        self.image_token_id = getattr(
            tokenizer, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID
        )
        self.audio_token_id = getattr(
            tokenizer, "audio_token_id", _DEFAULT_AUDIO_TOKEN_ID
        )
        self.image_start_token_id = getattr(
            tokenizer, "boi_token_id", _DEFAULT_IMAGE_START_TOKEN_ID
        )
        self.image_end_token_id = getattr(
            tokenizer, "eoi_token_id", _DEFAULT_IMAGE_END_TOKEN_ID
        )
        self.audio_start_token_id = getattr(
            tokenizer, "audio_start_token_id", _DEFAULT_AUDIO_START_TOKEN_ID
        )
        self.audio_end_token_id = getattr(
            tokenizer, "audio_end_token_id", _DEFAULT_AUDIO_END_TOKEN_ID
        )

    def _get_mm_fields_config(
        self,
        hf_inputs: object,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        """Routes per-item tensors to the GPU Worker's embed_multimodal kwargs."""
        # The first argument of batched() is the MODALITY
        # ("image"/"audio"), not the field name -- the engine looks items up
        # by modality during profiling and scheduling.
        return {
            "pixel_values": MultiModalFieldConfig.batched("image"),
            "audio_values": MultiModalFieldConfig.batched("audio"),
        }

    def _get_prompt_updates(self, *args: Any, **kwargs: Any) -> Sequence[PromptUpdate]:
        return []

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt
            if isinstance(inputs.prompt, str)
            else tokenizer.decode(inputs.prompt)
        )

        tokenization_kwargs = dict(inputs.tokenization_kwargs)

        # A token-id prompt was already tokenized upstream (the renderer applied
        # add_special_tokens per the request), so the decoded text carries its
        # BOS as literal text. Re-encoding below must not add another one, or
        # every multimodal chat request starts ``[1, 1, ...]`` (double BOS,
        # apertus-program #420). Only Apertus does this decode/re-encode round
        # trip; the stock BaseMultiModalProcessor never re-tokenizes.
        if not isinstance(inputs.prompt, str):
            tokenization_kwargs.setdefault("add_special_tokens", False)

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)

        images = (
            inputs.mm_data_items.get_items("image", ImageProcessorItems).get_all()
            if num_images > 0
            else None
        )
        audios = (
            inputs.mm_data_items.get_items("audio", AudioProcessorItems).get_all()
            if num_audios > 0
            else None
        )

        mm_kwargs: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        mm_counts: dict[str, int] = {}

        with timing_ctx.record("preprocess_apertus"):
            hf_outputs = self.hf_processor(
                text=prompt_text,
                images=images,
                audio=audios,
                padding=True,
                return_tensors="pt",
                **tokenization_kwargs,
            )
            prompt_token_ids = hf_outputs["input_ids"][0].tolist()

        if num_images > 0:
            pixel_values = [
                image[:, : int(height), : int(width)].contiguous()
                for image, (height, width) in zip(
                    hf_outputs["pixel_values"],
                    hf_outputs["image_sizes"],
                )
            ]
            mm_kwargs["pixel_values"] = pixel_values
            mm_counts["image"] = len(pixel_values)

        if num_audios > 0:
            audio_values = [
                audio[0, : int(mask.sum())].contiguous()
                for audio, mask in zip(
                    hf_outputs["input_features"],
                    hf_outputs["feature_attention_mask"],
                )
            ]
            mm_kwargs["audio_values"] = audio_values
            mm_counts["audio"] = len(audio_values)

        # mm_input() requires mm_hashes and mm_placeholders
        # since upstream 08a8a4af. Placeholders are one PlaceholderRange per
        # item spanning its layout tokens, with is_embed marking the positions
        # the GPU worker overwrites.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        def _span_ranges(
            start_token: str,
            start_id: int,
            end_token: str,
            end_id: int,
            embed_token_id: int,
            count: int,
        ) -> list[PlaceholderRange]:
            # Anchor on the atomic start/end special tokens directly in the
            # prompt ids: unlike re-encoding the layout string standalone,
            # special tokens cannot merge with surrounding text under BPE.
            ranges: list[PlaceholderRange] = []
            pos = 0
            for _ in range(count):
                try:
                    s = prompt_token_ids.index(start_id, pos)
                    e = prompt_token_ids.index(end_id, s)
                except ValueError as exc:
                    raise ValueError(
                        f"Apertus MM: {start_token!r}/{end_token!r} pair not "
                        f"found in prompt (search from {pos})"
                    ) from exc
                span = prompt_token_ids[s : e + 1]
                is_embed = torch.tensor(
                    [tok == embed_token_id for tok in span], dtype=torch.bool
                )
                ranges.append(
                    PlaceholderRange(offset=s, length=e - s + 1, is_embed=is_embed)
                )
                pos = e + 1
            return ranges

        mm_placeholders: dict[str, list[PlaceholderRange]] = {}
        if num_images > 0:
            mm_placeholders["image"] = _span_ranges(
                self.image_tokenizer.boi_token,
                self.image_start_token_id,
                self.image_tokenizer.eoi_token,
                self.image_end_token_id,
                self.image_token_id,
                num_images,
            )
        if num_audios > 0:
            mm_placeholders["audio"] = _span_ranges(
                self.audio_tokenizer.audio_start_token,
                self.audio_start_token_id,
                self.audio_tokenizer.audio_end_token,
                self.audio_end_token_id,
                self.audio_token_id,
                num_audios,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems.from_hf_inputs(
                mm_kwargs, self._get_mm_fields_config(mm_kwargs, {})
            ),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
        )


@MULTIMODAL_REGISTRY.register_processor(
    Apertus1p5MultiModalProcessor,
    info=Apertus1p5ProcessingInfo,
    dummy_inputs=Apertus1p5DummyInputsBuilder,
)
class Apertus1p5ForConditionalGeneration(ApertusForCausalLM, SupportsMultiModal):
    hf_to_vllm_mapper = ApertusForCausalLM.hf_to_vllm_mapper | WeightsMapper(
        orig_to_new_prefix={
            "model.language_model.": "model.",
            "model.vision_tokenizer.": "vision_tower.",
            "model.audio_tokenizer.": "audio_tower.",
        }
    )
    allow_patterns_overrides = ["model-apertus-model-*.safetensors"]

    # Required by vLLM's chat serving to insert the
    # modality placeholder when flattening OpenAI content parts. Without it
    # image/audio parts silently vanish from the prompt.
    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> str | None:
        if modality.startswith("image"):
            return _DEFAULT_IMAGE_PLACEHOLDER
        if modality.startswith("audio"):
            return _DEFAULT_AUDIO_PLACEHOLDER
        raise ValueError(f"Unsupported modality: {modality}")

    """
    GPU Worker Domain.
    Heavy inference executes natively on GPU. Returns embeddings of substituted tokens.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config

        output_vocab_size = getattr(config, "output_vocab_size", config.vocab_size)
        if output_vocab_size > config.vocab_size:
            raise ValueError("Output vocabulary cannot exceed input vocabulary.")
        self._input_vocab_size = config.vocab_size
        self._should_pad_logits_to_input_vocab = False
        if (
            get_pp_group().is_last_rank
            and not config.tie_word_embeddings
            and output_vocab_size != config.vocab_size
        ):
            self.lm_head = ParallelLMHead(
                output_vocab_size,
                config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            logit_scale = getattr(config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                output_vocab_size, scale=logit_scale
            )
            self._should_pad_logits_to_input_vocab = True

        self.vision_tower: Any | None = None
        self.audio_tower: Any | None = None
        if get_pp_group().is_first_rank:
            self.secondary_weights = [
                DefaultModelLoader.Source(
                    model_or_path=vllm_config.model_config.model,
                    revision=vllm_config.model_config.revision,
                    allow_patterns_overrides=[
                        "model-vision_tokenizer-model.safetensors"
                    ],
                ),
                DefaultModelLoader.Source(
                    model_or_path=vllm_config.model_config.model,
                    revision=vllm_config.model_config.revision,
                    allow_patterns_overrides=["model-wavtokenizer-model.safetensors"],
                ),
            ]
            with set_default_torch_dtype(torch.float32):
                with self._mark_tower_model(vllm_config, "image"):
                    self.vision_tower = _init_component_model(
                        config.vision_tokenizer_config,
                        model_cls=Apertus1p5VisionTokenizerModel,
                    )
                with self._mark_tower_model(vllm_config, "audio"):
                    self.audio_tower = _init_component_model(
                        config.audio_tokenizer_config,
                    )
        else:
            self.secondary_weights = []

        self.image_token_offset = getattr(
            config, "image_token_offset", _DEFAULT_IMAGE_TOKEN_OFFSET
        )
        self.audio_token_offset = getattr(
            config, "audio_token_offset", _DEFAULT_AUDIO_TOKEN_OFFSET
        )

    def get_language_model(self):
        return self.model

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = super().compute_logits(hidden_states)
        if logits is None or not self._should_pad_logits_to_input_vocab:
            return logits
        return _pad_logits_to_input_vocab(logits, self._input_vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor | IntermediateTensors:
        # Absorb multimodal kwargs (e.g. pixel_values, etc.) which are
        # already processed in embed_multimodal
        return self.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds,
        )

    def load_weights(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> set[str]:
        skip_prefixes = ["lm_head."] if self.config.tie_word_embeddings else []
        if not get_pp_group().is_first_rank:
            skip_prefixes.extend(["vision_tower.", "audio_tower."])

        loader = AutoWeightsLoader(self, skip_prefixes=skip_prefixes)
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)

    def _get_module_device_dtype(
        self,
        module: torch.nn.Module,
    ) -> tuple[torch.device, torch.dtype]:
        parameter = next(module.parameters())
        return parameter.device, parameter.dtype

    def _encode_image_to_llm_ids(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        vision_tower = self.vision_tower
        assert vision_tower is not None
        target_device, target_dtype = self._get_module_device_dtype(vision_tower)
        image = image.unsqueeze(0).to(device=target_device, dtype=target_dtype)
        with torch.inference_mode():
            valid_codes = vision_tower.encode(image).flatten()
        return valid_codes.to(torch.long) + self.image_token_offset

    def _encode_audio_to_llm_ids(
        self,
        audio: torch.Tensor,
    ) -> torch.Tensor:
        audio_tower = self.audio_tower
        assert audio_tower is not None
        target_device, target_dtype = self._get_module_device_dtype(audio_tower)
        with torch.inference_mode():
            output = audio_tower.encode(
                audio.unsqueeze(0)
                .unsqueeze(0)
                .to(device=target_device, dtype=target_dtype)
            )
            valid_codes = output.audio_codes.squeeze(0).squeeze(0)

        return valid_codes.to(torch.long) + self.audio_token_offset

    def _process_modality_input(
        self,
        values: torch.Tensor | list[torch.Tensor],
        encode_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> list[torch.Tensor]:
        """Encodes inputs for a single modality and retrieves their embeddings."""
        items = list(values.unbind(0)) if isinstance(values, torch.Tensor) else values
        if not items:
            return []

        ids_per_item = [encode_fn(item) for item in items]
        lengths = [ids.shape[0] for ids in ids_per_item]

        all_ids = torch.cat(ids_per_item).to(device)
        all_embeds = super().embed_input_ids(all_ids)
        return list(all_embeds.split(lengths))

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Encode all modality batches into language embeddings."""
        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda")

        multimodal_embeddings: list[torch.Tensor] = []

        # Iterate over keys of kwargs to preserve modality order
        for input_key in kwargs:
            if input_key == "pixel_values":
                pixel_values = kwargs[input_key]
                if pixel_values is not None and self.vision_tower is not None:
                    image_embeds = self._process_modality_input(
                        pixel_values,  # type: ignore
                        self._encode_image_to_llm_ids,
                        device,
                    )
                    multimodal_embeddings.extend(image_embeds)
            elif input_key == "audio_values":
                audio_values = kwargs[input_key]
                if audio_values is not None and self.audio_tower is not None:
                    audio_embeds = self._process_modality_input(
                        audio_values,  # type: ignore
                        self._encode_audio_to_llm_ids,
                        device,
                    )
                    multimodal_embeddings.extend(audio_embeds)

        return multimodal_embeddings

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
        **kwargs: object,
    ) -> torch.Tensor:
        # Route to standard vLLM multi-modal merge.
        # This takes the text embeddings generated from input_ids and
        # overwrites the rows corresponding to the dummy placeholders (where
        # is_multimodal is True) with the high-fidelity visual/audio
        # embeddings generated in embed_multimodal.
        return SupportsMultiModal.embed_input_ids(
            self,
            input_ids,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )
