# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from PIL import Image

from vllm.model_executor.models.apertus import (
    ApertusForCausalLM,
    ApertusMultiModalProcessor,
)
from vllm.model_executor.models.apertus_utils import (
    ApertusImageTokenizer,
    resolve_emu35_codebase,
)
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import ProcessorInputs, TimingContext

pytestmark = pytest.mark.cpu_test


class DummyTokenizer:
    image_token = "<|image|>"

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        self.encoded_texts.append(text)
        return [ord(char) % 257 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class DummyInfo:
    model_id = "dummy-apertus"

    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer

    def get_tokenizer(self) -> DummyTokenizer:
        return self.tokenizer

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()


def build_processor(tokenizer: DummyTokenizer) -> ApertusMultiModalProcessor:
    return ApertusMultiModalProcessor(  # type: ignore[arg-type]
        DummyInfo(tokenizer),
        dummy_inputs=None,
    )


def parse_mm_images(num_images: int):
    images = [
        Image.new("RGB", (16, 16), color=(idx, 0, 0))
        for idx in range(num_images)
    ]
    return MultiModalDataParser().parse_mm_data({"image": images})


def test_apertus_image_prompt_serialization_uses_expected_tokens():
    tokenizer = DummyTokenizer()
    image_tokenizer = ApertusImageTokenizer()
    image_tokens = torch.tensor([[1, 2], [3, 4]])

    prompt = image_tokenizer.build_apertus_image_prompt(image_tokens, tokenizer)

    assert prompt == (
        "<|img_start|>2*2<|img_token_start|>"
        "<|visual token 1|><|visual token 2|>"
        "<|img_end_of_row|>"
        "<|visual token 3|><|visual token 4|>"
        "<|img_end|>"
    )


def test_apertus_image_prompt_honors_tokenizer_special_tokens():
    class CustomTokenizer(DummyTokenizer):
        boi_token = "<BOI>"
        img_token = "<IMG>"
        eol_token = "<EOL>"
        eoi_token = "<EOI>"

    prompt = ApertusImageTokenizer().build_apertus_image_prompt(
        torch.tensor([[8, 9]]),
        CustomTokenizer(),
    )

    assert prompt == "<BOI>1*2<IMG><|visual token 8|><|visual token 9|><EOI>"


def test_apertus_image_tokenizer_unwraps_media_with_bytes():
    image = Image.new("RGB", (16, 16), color=(3, 4, 5))
    wrapped = MediaWithBytes(image, b"raw-bytes")

    coerced = ApertusImageTokenizer.coerce_pil_image(wrapped)

    assert isinstance(coerced, Image.Image)
    assert coerced.size == image.size


def test_apertus_processor_replaces_placeholders_then_tokenizes():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    processor.image_tokenizer.encode_images = lambda *args, **kwargs: [
        "<IMG0>",
        "<IMG1>",
    ]

    result = processor.apply(
        ProcessorInputs(
            prompt="A <|image|> B <|image|> C",
            mm_data_items=parse_mm_images(2),
        ),
        TimingContext(enabled=False),
    )

    assert result["prompt"] == "A <IMG0> B <IMG1> C"
    assert tokenizer.encoded_texts == ["A <IMG0> B <IMG1> C"]
    assert result["prompt_token_ids"] == tokenizer.encode(result["prompt"])
    assert not result["mm_kwargs"]
    assert result["mm_hashes"] == {}
    assert result["mm_placeholders"] == {}


def test_apertus_processor_accepts_legacy_image_placeholder_alias():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    processor.image_tokenizer.encode_images = lambda *args, **kwargs: ["<IMG0>"]

    result = processor.apply(
        ProcessorInputs(
            prompt="A <image> C",
            mm_data_items=parse_mm_images(1),
        ),
        TimingContext(enabled=False),
    )

    assert result["prompt"] == "A <IMG0> C"


def test_apertus_processor_rejects_placeholder_image_mismatch():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    processor.image_tokenizer.encode_images = lambda *args, **kwargs: ["<IMG0>"]

    with pytest.raises(ValueError, match="placeholder/input mismatch"):
        processor.apply(
            ProcessorInputs(
                prompt="A C",
                mm_data_items=parse_mm_images(1),
            ),
            TimingContext(enabled=False),
        )


def test_apertus_processor_text_only_path():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    result = processor.apply(
        ProcessorInputs(
            prompt="plain text",
            mm_data_items=MultiModalDataParser().parse_mm_data({}),
        ),
        TimingContext(enabled=False),
    )

    assert result["prompt"] == "plain text"
    assert tokenizer.encoded_texts == ["plain text"]
    assert result["mm_placeholders"] == {}


def test_apertus_model_placeholder_str():
    assert ApertusForCausalLM.get_placeholder_str("image", 0) == "<|image|>"


def test_apertus_emu35_codebase_resolver_accepts_explicit_path(tmp_path):
    module_dir = tmp_path / "src" / "vision_tokenizer"
    module_dir.mkdir(parents=True)
    (module_dir / "__init__.py").write_text("", encoding="utf-8")

    assert resolve_emu35_codebase({"apertus_emu35_codebase": str(tmp_path)}) == tmp_path
