# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.model_executor.models.apertus import (
    ApertusAudioTokenizer,
    ApertusForCausalLM,
    ApertusMultiModalProcessor,
    load_wavtokenizer40_class,
)
from vllm.model_executor.models.apertus_utils import (
    ApertusImageTokenizer,
    load_emu35_build_vision_tokenizer,
)
from vllm.multimodal.parse import MultiModalDataParser
from vllm.multimodal.processing import ProcessorInputs, TimingContext

pytestmark = pytest.mark.cpu_test


class DummyTokenizer:
    image_token = "<|image|>"
    audio_token = "<|audio|>"
    unk_token_id = -1

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []
        self._token_to_id: dict[str, int] = {
            "<|image|>": 900001,
            "<image>": 900002,
            "<|audio|>": 900003,
            "<|audio_start|>": 900004,
            "<|audio_end|>": 900005,
            "<|stt_transcribe|>": 900006,
            "<|stt_continue|>": 900007,
            "<|tts_continue|>": 900008,
        }
        self._id_to_token = {
            token_id: token for token, token_id in self._token_to_id.items()
        }
        self._update_special_order()

    def _update_special_order(self) -> None:
        self._special_tokens = sorted(
            self._token_to_id.keys(),
            key=len,
            reverse=True,
        )

    def register_token(self, token: str, token_id: int) -> None:
        self._token_to_id[token] = token_id
        self._id_to_token[token_id] = token
        self._update_special_order()

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, list):
            return [self.convert_tokens_to_ids(token) for token in tokens]
        return self._token_to_id.get(tokens, self.unk_token_id)

    def convert_ids_to_tokens(self, token_ids):
        if isinstance(token_ids, list):
            return [self.convert_ids_to_tokens(token_id) for token_id in token_ids]
        return self._id_to_token.get(token_ids, f"<|tok:{token_ids}|>")

    def encode(self, text: str, **kwargs) -> list[int]:
        del kwargs
        self.encoded_texts.append(text)

        encoded: list[int] = []
        idx = 0
        while idx < len(text):
            matched = False
            for token in self._special_tokens:
                if text.startswith(token, idx):
                    encoded.append(self._token_to_id[token])
                    idx += len(token)
                    matched = True
                    break

            if matched:
                continue

            encoded.append(1000 + ord(text[idx]))
            idx += 1

        return encoded

    def decode(self, token_ids: list[int]) -> str:
        decoded: list[str] = []
        for token_id in token_ids:
            token = self._id_to_token.get(token_id)
            if token is not None:
                decoded.append(token)
                continue

            if token_id >= 1000:
                decoded.append(chr(token_id - 1000))
            else:
                decoded.append("?")

        return "".join(decoded)


class DummyInfo:
    model_id = "dummy-apertus"

    def __init__(self, tokenizer: DummyTokenizer) -> None:
        self.tokenizer = tokenizer
        self.ctx = type(
            "DummyCtx",
            (),
            {
                "get_merged_mm_kwargs": staticmethod(
                    lambda kwargs: dict(kwargs) if kwargs else {}
                )
            },
        )()

    def get_tokenizer(self) -> DummyTokenizer:
        return self.tokenizer

    def get_data_parser(self) -> MultiModalDataParser:
        return MultiModalDataParser()


def build_processor(tokenizer: DummyTokenizer) -> ApertusMultiModalProcessor:
    return ApertusMultiModalProcessor(  # type: ignore[arg-type]
        DummyInfo(tokenizer),
        dummy_inputs=None,
    )


def parse_mm_inputs(*, num_images: int = 0, num_audios: int = 0):
    parser = MultiModalDataParser()
    payload: dict[str, object] = {}

    if num_images > 0:
        payload["image"] = [
            Image.new("RGB", (16, 16), color=(idx, 0, 0))
            for idx in range(num_images)
        ]
    if num_audios > 0:
        payload["audio"] = [
            np.zeros((64,), dtype=np.float32) + idx
            for idx in range(num_audios)
        ]

    return parser.parse_mm_data(payload)


def install_fake_encoders(processor: ApertusMultiModalProcessor) -> None:
    processor.image_tokenizer.encode_images = lambda images, **kwargs: [  # type: ignore[method-assign]
        f"<IMG{idx}>"
        for idx in range(len(images))
    ]
    processor.audio_tokenizer.encode_audios = lambda audios, **kwargs: [  # type: ignore[method-assign]
        f"<AUD{idx}>"
        for idx in range(len(audios))
    ]


def run_processor(
    processor: ApertusMultiModalProcessor,
    *,
    prompt: str,
    num_images: int = 0,
    num_audios: int = 0,
    mm_processor_kwargs: dict[str, object] | None = None,
):
    return processor.apply(
        ProcessorInputs(
            prompt=prompt,
            mm_data_items=parse_mm_inputs(num_images=num_images, num_audios=num_audios),
            hf_processor_mm_kwargs=mm_processor_kwargs or {},
        ),
        TimingContext(enabled=False),
    )


def test_apertus_image_prompt_serialization_uses_expected_tokens():
    image_tokenizer = ApertusImageTokenizer()
    image_tokens = torch.tensor([[1, 2], [3, 4]])

    prompt = image_tokenizer.build_apertus_image_prompt(image_tokens)

    assert prompt == (
        "<|img_start|>2*2<|img_token_start|>"
        "<|visual token 1|><|visual token 2|>"
        "<|img_end_of_row|>"
        "<|visual token 3|><|visual token 4|>"
        "<|img_end|>"
    )


def test_apertus_extracts_emu35_ibq_token_grid():
    token_ids = torch.arange(6)
    encode_out = (
        torch.empty(1, 256, 2, 3),
        0.0,
        (None, None, token_ids),
    )

    grid = ApertusImageTokenizer.extract_emu35_token_grid(encode_out, 2, 3)

    assert grid.tolist() == [[0, 1, 2], [3, 4, 5]]
    assert grid.dtype is torch.int64


def test_apertus_processor_text_only():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(processor, prompt="plain text")

    assert result["prompt"] == "plain text"
    assert tokenizer.encoded_texts == ["plain text"]
    assert result["mm_placeholders"] == {}
    assert result["mm_kwargs"] == {}


def test_apertus_processor_image_only():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="<|image|>",
        num_images=1,
    )

    assert result["prompt"] == "<IMG0>"


def test_apertus_processor_audio_only():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="<|audio|>",
        num_audios=1,
    )

    assert result["prompt"] == "<AUD0>"


def test_apertus_processor_image_text():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A <|image|> B",
        num_images=1,
    )

    assert result["prompt"] == "A <IMG0> B"


def test_apertus_processor_audio_text():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A <|audio|> B",
        num_audios=1,
    )

    assert result["prompt"] == "A <AUD0> B"


def test_apertus_processor_image_audio():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="<|image|><|audio|>",
        num_images=1,
        num_audios=1,
    )

    assert result["prompt"] == "<IMG0><AUD0>"


def test_apertus_processor_image_audio_text():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="T<|image|>M<|audio|>E",
        num_images=1,
        num_audios=1,
    )

    assert result["prompt"] == "T<IMG0>M<AUD0>E"


def test_apertus_processor_multiple_audio_placeholders():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A<|audio|>B<|audio|>C",
        num_audios=2,
    )

    assert result["prompt"] == "A<AUD0>B<AUD1>C"


def test_apertus_processor_multiple_image_placeholders():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A<|image|>B<|image|>C",
        num_images=2,
    )

    assert result["prompt"] == "A<IMG0>B<IMG1>C"


def test_apertus_processor_mixed_multiple_ordered_replacement():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A<|image|>B<|audio|>C<|image|>D<|audio|>E",
        num_images=2,
        num_audios=2,
    )

    assert result["prompt"] == "A<IMG0>B<AUD0>C<IMG1>D<AUD1>E"


def test_apertus_processor_absent_modality_without_placeholder():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A <|image|> B",
        num_images=1,
        num_audios=0,
    )

    assert result["prompt"] == "A <IMG0> B"


def test_apertus_processor_placeholder_present_but_missing_input():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    with pytest.raises(ValueError, match="audio placeholder/input mismatch"):
        run_processor(
            processor,
            prompt="A <|audio|> B",
            num_audios=0,
        )


def test_apertus_processor_image_placeholder_present_but_missing_input():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A <|image|> B",
        num_images=0,
    )
    assert result["prompt"] == "A  B"


def test_apertus_processor_input_present_but_missing_placeholder():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    with pytest.raises(ValueError, match="audio placeholder/input mismatch"):
        run_processor(
            processor,
            prompt="A B",
            num_audios=1,
        )


def test_apertus_processor_image_input_present_but_missing_placeholder():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    with pytest.raises(
        ValueError,
        match="Received more images than placeholders",
    ):
        run_processor(
            processor,
            prompt="A B",
            num_images=1,
        )


def test_apertus_processor_more_placeholders_than_images_replaces_extra_with_empty():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="A<|image|>B<|image|>C<|image|>D",
        num_images=2,
    )

    assert result["prompt"] == "A<IMG0>B<IMG1>CD"


def test_apertus_processor_zero_images_removes_all_image_placeholders():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)

    result = run_processor(
        processor,
        prompt="<|image|>\nNo image provided\n<|image|>",
        num_images=0,
    )

    assert result["prompt"] == "\nNo image provided\n"


def test_apertus_audio_token_serialization_roundtrip():
    tokenizer = DummyTokenizer()
    tokenizer.register_token("<|audio_code_1|>", 262345)
    tokenizer.register_token("<|audio_code_2|>", 262346)

    token_ids = [
        tokenizer.convert_tokens_to_ids("<|audio_start|>"),
        tokenizer.convert_tokens_to_ids("<|audio_code_1|>"),
        tokenizer.convert_tokens_to_ids("<|audio_code_2|>"),
        tokenizer.convert_tokens_to_ids("<|audio_end|>"),
    ]
    serialized = ApertusAudioTokenizer().serialize_audio_token_ids(
        token_ids,
        tokenizer,
    )

    assert tokenizer.encode(serialized, add_special_tokens=False) == token_ids


def test_apertus_model_placeholder_str():
    assert ApertusForCausalLM.get_placeholder_str("image", 0) == "<|image|>"
    assert ApertusForCausalLM.get_placeholder_str("audio", 0) == "<|audio|>"


def test_apertus_image_placeholder_aliases_include_apertus_default():
    aliases = ApertusImageTokenizer.placeholder_aliases()

    assert aliases == ["<|image|>"]


def test_apertus_emu35_vision_tokenizer_loads_from_installed_package(monkeypatch):
    module = types.ModuleType("vision_tokenizer")

    def build_vision_tokenizer(**kwargs):
        return kwargs

    module.build_vision_tokenizer = build_vision_tokenizer

    load_emu35_build_vision_tokenizer.cache_clear()
    try:
        monkeypatch.setitem(sys.modules, "vision_tokenizer", module)

        assert load_emu35_build_vision_tokenizer() is build_vision_tokenizer
    finally:
        load_emu35_build_vision_tokenizer.cache_clear()


def test_apertus_vision_tokenizer_device_comes_from_mm_kwargs(monkeypatch):
    tokenizer = ApertusImageTokenizer()
    captured_kwargs: dict[str, object] = {}

    class FakeVisionTokenizer:
        pass

    def fake_build_emu35_vision_tokenizer(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeVisionTokenizer()

    monkeypatch.setattr(
        "vllm.model_executor.models.apertus_utils.build_emu35_vision_tokenizer",
        fake_build_emu35_vision_tokenizer,
    )

    _, device, dtype = tokenizer.load_vision_tokenizer(
        {"apertus_vision_tokenizer_device": "cuda:1"}
    )

    assert device == "cuda:1"
    assert dtype is torch.float32
    assert captured_kwargs["device"] == "cuda:1"
    assert captured_kwargs["dtype"] is torch.float32


def test_apertus_vision_tokenizer_cpu_uses_default_dtype(monkeypatch):
    tokenizer = ApertusImageTokenizer()
    captured_kwargs: dict[str, object] = {}

    class FakeVisionTokenizer:
        pass

    def fake_build_emu35_vision_tokenizer(**kwargs):
        captured_kwargs.update(kwargs)
        return FakeVisionTokenizer()

    monkeypatch.setattr(
        "vllm.model_executor.models.apertus_utils.build_emu35_vision_tokenizer",
        fake_build_emu35_vision_tokenizer,
    )

    _, device, dtype = tokenizer.load_vision_tokenizer(
        {"apertus_vision_tokenizer_device": "cpu"}
    )

    assert device == "cpu"
    assert dtype is torch.float32
    assert captured_kwargs["device"] == "cpu"
    assert captured_kwargs["dtype"] is torch.float32


def test_apertus_audio_tokenizer_loads_from_installed_package():
    load_wavtokenizer40_class.cache_clear()
    module = types.ModuleType("apertus_audio_tokenizer")

    class FakeWavTokenizer40:
        pass

    module.WavTokenizer40 = FakeWavTokenizer40
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "apertus_audio_tokenizer", module)
        assert load_wavtokenizer40_class() is FakeWavTokenizer40
    load_wavtokenizer40_class.cache_clear()


def test_apertus_audio_tokenizer_only_honors_operational_kwargs():
    load_wavtokenizer40_class.cache_clear()
    module = types.ModuleType("apertus_audio_tokenizer")
    init_kwargs: dict[str, object] = {}

    class FakeWavTokenizer40:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

    module.WavTokenizer40 = FakeWavTokenizer40
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "apertus_audio_tokenizer", module)
        audio_tokenizer = ApertusAudioTokenizer().get_audio_tokenizer(
            {
                "apertus_audio_tokenizer_type": "other",
                "apertus_audio_tokenizer_name": "OtherTokenizer",
                "apertus_audio_tokenizer_compile": False,
                "apertus_audio_tokenizer_device": "cpu",
            }
        )

    assert isinstance(audio_tokenizer, FakeWavTokenizer40)
    assert init_kwargs == {
        "device": "cpu",
        "torch_compile": False,
    }
    load_wavtokenizer40_class.cache_clear()


def test_apertus_audio_tokenizer_cache_is_keyed_by_device():
    load_wavtokenizer40_class.cache_clear()
    module = types.ModuleType("apertus_audio_tokenizer")
    init_kwargs: list[dict[str, object]] = []

    class FakeWavTokenizer40:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.append(dict(kwargs))

    module.WavTokenizer40 = FakeWavTokenizer40
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "apertus_audio_tokenizer", module)
        apertus_audio_tokenizer = ApertusAudioTokenizer()

        cpu_tokenizer = apertus_audio_tokenizer.get_audio_tokenizer(
            {"apertus_audio_tokenizer_device": "cpu"}
        )
        cuda0_tokenizer = apertus_audio_tokenizer.get_audio_tokenizer(
            {"apertus_audio_tokenizer_device": "cuda:0"}
        )
        cached_cpu_tokenizer = apertus_audio_tokenizer.get_audio_tokenizer(
            {"apertus_audio_tokenizer_device": "cpu"}
        )

    assert cpu_tokenizer is cached_cpu_tokenizer
    assert cpu_tokenizer is not cuda0_tokenizer
    assert init_kwargs == [
        {
            "device": "cpu",
            "torch_compile": True,
        },
        {
            "device": "cuda:0",
            "torch_compile": True,
        },
    ]
    load_wavtokenizer40_class.cache_clear()


def test_apertus_processor_forwards_audio_mm_processor_kwargs():
    tokenizer = DummyTokenizer()
    processor = build_processor(tokenizer)
    install_fake_encoders(processor)
    captured_kwargs: dict[str, object] = {}

    def encode_audios(audios, **kwargs):  # type: ignore[no-untyped-def]
        del audios
        captured_kwargs.update(kwargs["mm_processor_kwargs"])
        return ["<AUD0>"]

    processor.audio_tokenizer.encode_audios = encode_audios  # type: ignore[method-assign]

    result = run_processor(
        processor,
        prompt="A<|audio|>B",
        num_audios=1,
        mm_processor_kwargs={
            "apertus_audio_tokenizer_device": "cuda:1",
            "apertus_audio_tokenizer_compile": False,
        },
    )

    assert result["prompt"] == "A<AUD0>B"
    assert captured_kwargs == {
        "apertus_audio_tokenizer_device": "cuda:1",
        "apertus_audio_tokenizer_compile": False,
    }


def test_apertus_audio_tokenizer_uses_explicit_checkpoint_path():
    load_wavtokenizer40_class.cache_clear()
    module = types.ModuleType("apertus_audio_tokenizer")
    init_kwargs: dict[str, object] = {}

    class FakeWavTokenizer40:
        def __init__(self, **kwargs: object) -> None:
            init_kwargs.update(kwargs)

    module.WavTokenizer40 = FakeWavTokenizer40
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setitem(sys.modules, "apertus_audio_tokenizer", module)
        ApertusAudioTokenizer().get_audio_tokenizer(
            {"apertus_audio_tokenizer_path": "/tmp/wavtokenizer"}
        )

    assert init_kwargs["checkpoint"] == "/tmp/wavtokenizer"
    load_wavtokenizer40_class.cache_clear()
