# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Multimodal Apertus Pipeline optimized for native vLLM asynchronous execution.

Architecture Contract:
1. Processor (CPU): Deterministic per-item tensor creation and dummy token
   injection. Zero neural network inference is executed here.
2. Worker (GPU): Native Emu3.5/WavTokenizer execution and O(1) boolean mask
   ID substitution.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Apertus1p5VisionTokenizerModel, AutoConfig, AutoModel

from vllm.config import VllmConfig
from vllm.config.multimodal import BaseDummyOptions
from vllm.distributed import get_pp_group
from vllm.inputs import MultiModalDataDict, MultiModalInput, mm_input
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
from .utils import AutoWeightsLoader, WeightsMapper


def _init_component_model(
    component_config: Mapping[str, Any],
    model_cls: type[torch.nn.Module] | None = None,
) -> torch.nn.Module:
    config_dict = dict(component_config)
    config = AutoConfig.for_model(config_dict.pop("model_type"), **config_dict)
    return AutoModel.from_config(config) if model_cls is None else model_cls(config)


class ApertusImageTokenizer:
    def __init__(self, vision_config: Mapping[str, Any] | None = None) -> None:
        config = vision_config or {}
        self.min_pixels = config.get("min_pixels", 256 * 256)
        self.max_pixels = config.get("max_pixels", 1400 * 1400)
        self.image_placeholder = config.get("image_placeholder", "<|image|>")
        self.ds_factor = config["ds_factor"]
        self.boi_token = config.get("boi_token", "<|img_start|>")
        self.img_token = config.get("img_token", "<|img_token_start|>")
        self.eol_token = config.get("eol_token", "<|img_end_of_row|>")
        self.eoi_token = config.get("eoi_token", "<|img_end|>")

    @staticmethod
    def smart_resize(image: Image.Image, area: int, ds_factor: int) -> Image.Image:
        width, height = image.size
        aspect_ratio = width / height
        new_height = int((area / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)
        new_height = ((new_height + ds_factor // 2) // ds_factor) * ds_factor
        new_width = ((new_width + ds_factor // 2) // ds_factor) * ds_factor
        return image.resize((new_width, new_height), Image.BICUBIC)

    @staticmethod
    def image_array_to_hwc(array: np.ndarray) -> np.ndarray:
        if array.ndim != 3:
            raise TypeError("Apertus image adapter expects 3D image arrays.")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            return np.transpose(array, (1, 2, 0))
        return array

    @staticmethod
    def image_array_to_uint8(array: np.ndarray) -> np.ndarray:
        if (
            np.issubdtype(array.dtype, np.floating)
            and array.size
            and np.nanmax(array) <= 1.0
        ):
            array = array * 255.0

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        return array

    @classmethod
    def coerce_pil_image(
        cls,
        image: Image.Image | np.ndarray | torch.Tensor,
    ) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if not isinstance(image, np.ndarray):
            raise TypeError(
                "Apertus image adapter expects PIL images, numpy arrays, or "
                "torch tensors in multi_modal_data['image'].",
            )

        array = cls.image_array_to_hwc(image)
        array = cls.image_array_to_uint8(array)
        return Image.fromarray(array).convert("RGB")

    def preprocess_image_to_tensor(
        self, raw_image: Image.Image | np.ndarray | torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        image = self.coerce_pil_image(raw_image)
        width, height = image.size
        target_area = max(min(self.max_pixels, width * height), self.min_pixels)
        resized = self.smart_resize(image, target_area, self.ds_factor)

        tensor = torch.tensor(
            (np.array(resized) / 127.5 - 1.0), dtype=torch.float32
        ).permute(2, 0, 1)
        height_tokens, width_tokens = (
            tensor.shape[1] // self.ds_factor,
            tensor.shape[2] // self.ds_factor,
        )
        return tensor, height_tokens, width_tokens


class ApertusAudioTokenizer:
    def __init__(self, audio_config: Mapping[str, Any] | None = None) -> None:
        config = audio_config or {}
        self.audio_placeholder = config.get("audio_placeholder", "<|audio|>")
        self.target_peak_dbfs = config.get("target_peak_dbfs", -3.0)
        self.audio_start_token = config.get("audio_start_token", "<|audio_start|>")
        self.audio_end_token = config.get("audio_end_token", "<|audio_end|>")

    def preprocess_audio_to_tensor(
        self, raw_audio: np.ndarray
    ) -> tuple[torch.Tensor, int]:
        if raw_audio.ndim == 0:
            raise ValueError("Audio waveform must have at least one dimension.")
        waveform = np.asarray(raw_audio, dtype=np.float32)
        tensor = torch.from_numpy(waveform).float()
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)

        peak = tensor.abs().max().clamp(min=1e-10)
        target = 10 ** (self.target_peak_dbfs / 20.0)
        tensor = tensor * (target / peak)

        num_tok = (tensor.shape[-1] + 600 - 1) // 600
        return tensor, num_tok


class ApertusProcessingInfo(BaseProcessingInfo):
    def get_data_parser(self) -> MultiModalDataParser:
        audio_config = self.get_hf_config().audio_tokenizer_config
        target_sr = audio_config.get("sampling_rate", 24000)
        return MultiModalDataParser(
            target_sr=target_sr,
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
        vision_config = self.get_hf_config().vision_tokenizer_config
        ds_factor = vision_config.get("ds_factor", 16)
        max_px = vision_config.get("max_pixels", 1400 * 1400)
        audio_config = self.get_hf_config().audio_tokenizer_config
        tokens_per_second = audio_config.get("tokens_per_second", 40)

        return {
            "image": min((max_px // (ds_factor * ds_factor)) + 512, seq_len),
            "audio": min((tokens_per_second * 300) + 4, seq_len),
        }


class ApertusDummyInputsBuilder(BaseDummyInputsBuilder[ApertusProcessingInfo]):
    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        vision_config = self.info.get_hf_config().vision_tokenizer_config
        image_placeholder = vision_config.get("image_placeholder", "<|image|>")
        audio_config = self.info.get_hf_config().audio_tokenizer_config
        audio_placeholder = audio_config.get("audio_placeholder", "<|audio|>")
        return image_placeholder * mm_counts.get(
            "image", 0
        ) + audio_placeholder * mm_counts.get("audio", 0)

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions],
    ) -> MultiModalDataDict:
        vision_config = self.info.get_hf_config().vision_tokenizer_config
        max_px = vision_config.get("max_pixels", 1400 * 1400)
        max_side = int(max_px**0.5)
        audio_config = self.info.get_hf_config().audio_tokenizer_config
        audio_length = audio_config.get("sampling_rate", 24000)
        image_overrides = mm_options.get("image")
        audio_overrides = mm_options.get("audio")

        return {
            "image": self._get_dummy_images(
                width=max_side,
                height=max_side,
                num_images=mm_counts.get("image", 0),
                overrides=image_overrides,
            ),
            "audio": self._get_dummy_audios(
                length=audio_length,
                num_audios=mm_counts.get("audio", 0),
                overrides=audio_overrides,
            ),
        }


class ApertusMultiModalProcessor(BaseMultiModalProcessor[ApertusProcessingInfo]):
    """CPU-bound API Processor. Strict YAGNI Rule: NO heavy neural networks run here."""

    def __init__(
        self,
        info: ApertusProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder,
        *,
        cache: object | None = None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        config = info.get_hf_config()
        vision_config = config.vision_tokenizer_config
        audio_config = config.audio_tokenizer_config
        self.image_tokenizer = ApertusImageTokenizer(vision_config)
        self.audio_tokenizer = ApertusAudioTokenizer(audio_config)
        self.dummy_image_token = "<|visual token 0|>"
        self.dummy_audio_token = "<|audio token 0|>"
        self.dummy_image_token_id = vision_config.get("image_token_offset")
        self.dummy_audio_token_id = audio_config.get("audio_token_offset")
        self.image_start_token_id = vision_config.get("image_start_token_id")
        self.image_end_token_id = vision_config.get("image_end_token_id")
        self.audio_start_token_id = audio_config.get("audio_start_token_id")
        self.audio_end_token_id = audio_config.get("audio_end_token_id")

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

    def _preprocess_image_item(
        self,
        image: Any,
        dummy_token: str,
    ) -> tuple[torch.Tensor, str]:
        tensor, h_tok, w_tok = self.image_tokenizer.preprocess_image_to_tensor(image)
        rows = [dummy_token * w_tok for _ in range(h_tok)]
        image_tokens = self.image_tokenizer.eol_token.join(rows)
        layout = (
            f"{self.image_tokenizer.boi_token}{h_tok}*{w_tok}"
            f"{self.image_tokenizer.img_token}{image_tokens}"
            f"{self.image_tokenizer.eoi_token}"
        )
        return tensor, layout

    def _preprocess_audio_item(
        self,
        audio: Any,
        dummy_token: str,
    ) -> tuple[torch.Tensor, str]:
        tensor, num_tokens = self.audio_tokenizer.preprocess_audio_to_tensor(audio)
        layout = (
            f"{self.audio_tokenizer.audio_start_token}"
            f"{dummy_token * num_tokens}"
            f"{self.audio_tokenizer.audio_end_token}"
        )
        return tensor.squeeze(0), layout

    def apply(
        self, inputs: ProcessorInputs, timing_ctx: TimingContext
    ) -> MultiModalInput:
        tokenizer = self.info.get_tokenizer()
        prompt_text = (
            inputs.prompt
            if isinstance(inputs.prompt, str)
            else tokenizer.decode(inputs.prompt)
        )
        
        # A token-id prompt was already tokenized upstream (the renderer applied
        # add_special_tokens per the request), so the decoded text carries its
        # BOS as literal text. Re-encoding below must not add another one, or
        # every multimodal chat request starts ``[1, 1, ...]`` (double BOS,
        # apertus-program #420). Only Apertus does this decode/re-encode round
        # trip; the stock BaseMultiModalProcessor never re-tokenizes.
        if not isinstance(inputs.prompt, str):
            tokenization_kwargs = dict(inputs.tokenization_kwargs)
            tokenization_kwargs.setdefault("add_special_tokens", False)
            inputs.tokenization_kwargs = tokenization_kwargs

        num_images = inputs.mm_data_items.get_count("image", strict=False)
        num_audios = inputs.mm_data_items.get_count("audio", strict=False)

        mm_kwargs: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        mm_counts: dict[str, int] = {}

        # 1. Vision Preprocessing (Mathematical Layout Only)
        if num_images > 0:
            with timing_ctx.record("preprocess_apertus_images"):
                images = inputs.mm_data_items.get_items(
                    "image", ImageProcessorItems
                ).get_all()
                pixel_values, image_layouts = [], []
                for image in images:
                    value, layout = self._preprocess_image_item(
                        image, self.dummy_image_token
                    )
                    pixel_values.append(value)
                    image_layouts.append(layout)

                mm_kwargs["pixel_values"] = pixel_values
                mm_counts["image"] = len(image_layouts)
                for layout in image_layouts:
                    prompt_text = prompt_text.replace(
                        self.image_tokenizer.image_placeholder, layout, 1
                    )

        # 2. Audio Preprocessing (Mathematical Layout Only)
        if num_audios > 0:
            with timing_ctx.record("preprocess_apertus_audios"):
                audios = inputs.mm_data_items.get_items(
                    "audio", AudioProcessorItems
                ).get_all()
                audio_values, audio_layouts = [], []
                for audio in audios:
                    value, layout = self._preprocess_audio_item(
                        audio, self.dummy_audio_token
                    )
                    audio_values.append(value)
                    audio_layouts.append(layout)

                mm_kwargs["audio_values"] = audio_values
                mm_counts["audio"] = len(audio_layouts)

                for layout in audio_layouts:
                    prompt_text = prompt_text.replace(
                        self.audio_tokenizer.audio_placeholder, layout, 1
                    )

        with timing_ctx.record("tokenize"):
            prompt_token_ids = list(
                tokenizer.encode(prompt_text, **dict(inputs.tokenization_kwargs))
            )

        # mm_input() requires mm_hashes and mm_placeholders
        # since upstream 08a8a4af. Placeholders are one PlaceholderRange per
        # item spanning its layout tokens, with is_embed marking the dummy
        # positions the GPU worker overwrites.
        with timing_ctx.record("get_mm_hashes"):
            mm_hashes = inputs.get_mm_hashes(self.info.model_id)

        def _span_ranges(
            start_token: str,
            start_id: int,
            end_token: str,
            end_id: int,
            dummy_token_id: int,
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
                    [tok == dummy_token_id for tok in span], dtype=torch.bool
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
                self.dummy_image_token_id,
                num_images,
            )
        if num_audios > 0:
            mm_placeholders["audio"] = _span_ranges(
                self.audio_tokenizer.audio_start_token,
                self.audio_start_token_id,
                self.audio_tokenizer.audio_end_token,
                self.audio_end_token_id,
                self.dummy_audio_token_id,
                num_audios,
            )

        return mm_input(
            prompt_token_ids=prompt_token_ids,
            mm_kwargs=MultiModalKwargsItems.from_hf_inputs(
                mm_kwargs, self._get_mm_fields_config(mm_kwargs, {})
            ),
            mm_hashes=mm_hashes,
            mm_placeholders=mm_placeholders,
            prompt=prompt_text,
        )


@MULTIMODAL_REGISTRY.register_processor(
    ApertusMultiModalProcessor,
    info=ApertusProcessingInfo,
    dummy_inputs=ApertusDummyInputsBuilder,
)
class ApertusForConditionalGeneration(ApertusForCausalLM, SupportsMultiModal):
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
            return "<|image|>"
        if modality.startswith("audio"):
            return "<|audio|>"
        raise ValueError(f"Unsupported modality: {modality}")

    """
    GPU Worker Domain.
    Heavy inference executes natively on GPU. Returns embeddings of substituted tokens.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = vllm_config.model_config.hf_config
        self.image_tokenizer = ApertusImageTokenizer(config.vision_tokenizer_config)
        self.audio_tokenizer = ApertusAudioTokenizer(config.audio_tokenizer_config)

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

        self.image_token_offset = config.vision_tokenizer_config.get(
            "image_token_offset"
        )
        self.audio_token_offset = config.audio_tokenizer_config.get(
            "audio_token_offset"
        )

    def get_language_model(self):
        return self.model

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
                audio.unsqueeze(0).unsqueeze(0).to(
                    device=target_device, dtype=target_dtype
                )
            )
            valid_codes = output.audio_codes.squeeze(0).squeeze(0)

        return valid_codes.to(torch.long) + self.audio_token_offset

    def _parse_mm_modality_and_encoder(
        self,
        **kwargs: object,
    ) -> tuple[
        torch.Tensor | list[torch.Tensor],
        Callable[[torch.Tensor], torch.Tensor],
    ] | None:
        pixel_values = kwargs.get("pixel_values")
        audio_values = kwargs.get("audio_values")

        # vLLM batches each encoder call by modality. Both fields cannot be set.
        if pixel_values is not None:
            if self.vision_tower is None:
                return None
            return pixel_values, self._encode_image_to_llm_ids
        if audio_values is not None:
            if self.audio_tower is None:
                return None
            return audio_values, self._encode_audio_to_llm_ids

        return None

    def embed_multimodal(
        self,
        **kwargs: object,
    ) -> MultiModalEmbeddings:
        """Encode one modality batch into language embeddings."""
        selected_encoder = self._parse_mm_modality_and_encoder(**kwargs)
        if selected_encoder is None:
            return []
        values, encode_to_llm_ids = selected_encoder

        try:
            device = next(self.parameters()).device
        except StopIteration:
            device = torch.device("cuda")

        items = list(values.unbind(0)) if isinstance(values, torch.Tensor) else values

        ids_per_item = [encode_to_llm_ids(item) for item in items]
        lengths = [ids.shape[0] for ids in ids_per_item]

        all_ids = torch.cat(ids_per_item).to(device)
        all_embeds = super().embed_input_ids(all_ids)
        return list(all_embeds.split(lengths))

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
