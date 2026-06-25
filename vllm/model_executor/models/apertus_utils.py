# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Apertus multimodal preprocessing helpers."""

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vllm.logger import init_logger

logger = init_logger(__name__)


def resolve_emu35_weights(
    hf_repo_id: str,
    *,
    cache_dir: str | None = None,
) -> str:
    import huggingface_hub

    logger.info("Resolving %s from Hugging Face cache (cache_dir=%s)",
                hf_repo_id, cache_dir)
    hf_folder = huggingface_hub.snapshot_download(
        repo_id=hf_repo_id,
        allow_patterns=["config.yaml", "model.ckpt"],
        cache_dir=cache_dir,
    )
    return str(Path(hf_folder).resolve())


@lru_cache(maxsize=4)
def load_emu35_build_vision_tokenizer() -> Any:
    try:
        from vision_tokenizer import build_vision_tokenizer
    except ImportError as exc:
        raise ImportError(
            "Apertus image preprocessing requires the Emu3.5 package to be "
            "installed. Install it with "
            "`uv pip install git+https://github.com/swiss-ai/Emu3.5.git` "
            "or install the package into the vLLM environment."
        ) from exc

    return build_vision_tokenizer


def build_emu35_vision_tokenizer(
    *,
    vq_hub: str,
    device: str,
    vq_type: str = "ibq",
    cache_dir: str | None = None,
    **kwargs: Any,
) -> Any:
    local_vq_path = resolve_emu35_weights(vq_hub, cache_dir=cache_dir)
    build_vision_tokenizer = load_emu35_build_vision_tokenizer()
    return build_vision_tokenizer(
        type=vq_type,
        model_path=local_vq_path,
        device=device,
        **kwargs,
    ).eval()


class ApertusImageTokenizer:
    DEFAULT_VQ_HUB = "BAAI/Emu3.5-VisionTokenizer"
    DEFAULT_MIN_PIXELS = 256 * 256
    DEFAULT_MAX_PIXELS = 1400 * 1400
    DEFAULT_IMAGE_PLACEHOLDER = "<|image|>"
    VISUAL_TEMPLATE = "<|visual token {token_id}|>"
    EMU35_DS_FACTOR = 16
    DEFAULT_BOI_TOKEN = "<|img_start|>"
    DEFAULT_IMG_TOKEN = "<|img_token_start|>"
    DEFAULT_EOL_TOKEN = "<|img_end_of_row|>"
    DEFAULT_EOI_TOKEN = "<|img_end|>"

    def __init__(self) -> None:
        self._vision_tokenizer_cache: dict[
            tuple[str, str, str, torch.dtype],
            tuple[Any, str, torch.dtype],
        ] = {}

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
    def extract_emu35_token_grid(
        encode_out: Any,
        token_height: int,
        token_width: int,
    ) -> torch.Tensor:
        token = encode_out[2][2]
        expected = token_height * token_width
        if token.numel() != expected:
            raise ValueError(
                "Apertus Emu3.5 token length mismatch: "
                f"got {token.numel()}, expected {expected}."
            )
        return token.view(token_height, token_width).to(dtype=torch.int64)

    @classmethod
    def placeholder_aliases(cls) -> list[str]:
        return [cls.DEFAULT_IMAGE_PLACEHOLDER]

    def load_vision_tokenizer(
        self,
        mm_processor_kwargs: Mapping[str, object],
    ) -> tuple[Any, str, torch.dtype]:
        vq_hub = self.DEFAULT_VQ_HUB
        vq_type = "ibq"
        vision_device = str(
            mm_processor_kwargs.get("apertus_vision_tokenizer_device", "cuda")
        )
        vision_dtype = torch.float32

        cache_key = (
            vq_hub,
            vq_type,
            vision_device,
            vision_dtype,
        )

        if cache_key in self._vision_tokenizer_cache:
            return self._vision_tokenizer_cache[cache_key]

        cache_dir = mm_processor_kwargs.get("apertus_vq_cache_dir")
        vision_tokenizer = build_emu35_vision_tokenizer(
            vq_hub=vq_hub,
            device=vision_device,
            vq_type=vq_type,
            cache_dir=cache_dir
            if isinstance(cache_dir, str)
            else None,
            dtype=vision_dtype,
        )

        resolved = (vision_tokenizer, vision_device, vision_dtype)
        self._vision_tokenizer_cache[cache_key] = resolved
        return resolved

    def build_apertus_image_prompt(
        self,
        image_tokens: torch.Tensor,
    ) -> str:
        if image_tokens.ndim != 2:
            raise ValueError(
                f"Apertus image tokens must be 2D, got "
                f"shape {tuple(image_tokens.shape)}"
            )

        height, width = image_tokens.shape
        rows = [
            "".join(
                self.VISUAL_TEMPLATE.format(token_id=int(token_id))
                for token_id in row
            )
            for row in image_tokens.detach().to("cpu").tolist()
        ]
        imgstr = self.DEFAULT_EOL_TOKEN.join(rows)

        return (
            f"{self.DEFAULT_BOI_TOKEN}{height}*{width}"
            f"{self.DEFAULT_IMG_TOKEN}{imgstr}{self.DEFAULT_EOI_TOKEN}"
        )

    def encode_images(
        self,
        images: Sequence[Image.Image],
        *,
        mm_processor_kwargs: Mapping[str, object],
    ) -> list[str]:
        if not images:
            return []

        vision_tokenizer, vision_device, vision_dtype = self.load_vision_tokenizer(
            mm_processor_kwargs
        )

        image_prompts: list[str] = []
        for image in images:
            image = image.convert("RGB")
            width, height = image.size
            current_area = width * height
            target_area = max(
                min(self.DEFAULT_MAX_PIXELS, current_area),
                self.DEFAULT_MIN_PIXELS,
            )
            resized_image = self.smart_resize(image, target_area, self.EMU35_DS_FACTOR)
            resized_w, resized_h = resized_image.size

            image_tensor = torch.tensor(
                (np.array(resized_image) / 127.5 - 1.0),
                device=vision_device,
                dtype=vision_dtype,
            ).permute(2, 0, 1)

            with torch.inference_mode():
                try:
                    encode_out = vision_tokenizer.encode(image_tensor[None])
                except TypeError:
                    try:
                        encode_out = vision_tokenizer.encode(
                            pixel_values=image_tensor[None]
                        )
                    except TypeError:
                        encode_out = vision_tokenizer.encode(images=image_tensor[None])

            token_h = resized_h // self.EMU35_DS_FACTOR
            token_w = resized_w // self.EMU35_DS_FACTOR
            image_token_grid = self.extract_emu35_token_grid(
                encode_out, token_h, token_w
            )
            image_prompts.append(self.build_apertus_image_prompt(image_token_grid))

        return image_prompts
