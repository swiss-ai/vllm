# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline Apertus multimodal inference (text/image/audio).

Supported scenarios:
  - text_only
  - image_only
  - audio_only
  - image_text
  - audio_text
  - image_audio
  - image_audio_text
  - all

Example:
    python examples/offline_inference/apertus_multimodal.py \
      --scenario all \
      --image /path/to/image.jpg \
      --audio /path/to/audio.wav \
      --tensor-parallel-size 4
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

try:
    import soundfile as sf
except ImportError:
    sf = None

DEFAULT_MODEL = (
    "/capstor/store/cscs/swissai/infra01/MLLM/ablations/"
    "apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-"
    "seqlen8192-s2onlytxtloss/HF"
)
DEFAULT_TOKENIZER = (
    "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/"
    "apertus_emu3.5_instruct"
)
DEFAULT_AUDIO_TOKENIZER_PATH = "/capstor/store/cscs/swissai/infra01/MLLM/wavtokenizer"
IMAGE_PLACEHOLDER = "<|image|>"
AUDIO_PLACEHOLDER = "<|audio|>"

SCENARIOS = (
    "text_only",
    "image_only",
    "audio_only",
    "image_text",
    "audio_text",
    "image_audio",
    "image_audio_text",
    "all",
)


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--scenario", default="image_text", choices=SCENARIOS)
    parser.add_argument("--image", default=None, help="Optional image path.")
    parser.add_argument("--audio", default=None, help="Optional audio path.")
    parser.add_argument(
        "--text",
        default="Describe the provided content in one concise sentence.",
    )
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vision-tokenizer-device", default="cuda")
    parser.add_argument(
        "--emu35-codebase",
        default=None,
        help=(
            "Path to an Emu3.5 checkout. Defaults to the sibling "
            "vllm-omni/external/Emu3.5 or lmms-eval/external/Emu3.5 checkout."
        ),
    )
    parser.add_argument("--vq-hub", default="BAAI/Emu3.5-VisionTokenizer")
    parser.add_argument("--audio-tokenizer-path", default=DEFAULT_AUDIO_TOKENIZER_PATH)
    parser.add_argument("--audio-tokenizer-device", default="cuda")
    parser.add_argument(
        "--audio-tokenizer-codebase",
        default=None,
        help="Path to benchmark-audio-tokenizer checkout.",
    )
    parser.add_argument("--audio-target-sampling-rate", type=int, default=24000)
    parser.add_argument("--audio-default-sampling-rate", type=int, default=16000)
    parser.add_argument("--audio-token-offset", type=int, default=262344)
    parser.add_argument(
        "--expect-substring",
        default=None,
        help="Optional case-insensitive substring that must appear in answers.",
    )
    return parser.parse_args()


def build_mm_processor_kwargs(args) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "apertus_vision_tokenizer_device": args.vision_tokenizer_device,
        "apertus_vq_hub": args.vq_hub,
        "apertus_audio_tokenizer_path": args.audio_tokenizer_path,
        "apertus_audio_tokenizer_device": args.audio_tokenizer_device,
        "apertus_audio_target_sampling_rate": args.audio_target_sampling_rate,
        "apertus_audio_default_sampling_rate": args.audio_default_sampling_rate,
        "apertus_audio_token_offset": args.audio_token_offset,
    }
    if args.emu35_codebase:
        kwargs["apertus_emu35_codebase"] = args.emu35_codebase
    if args.audio_tokenizer_codebase:
        kwargs["apertus_audio_tokenizer_codebase"] = args.audio_tokenizer_codebase
    return kwargs


def load_or_create_image(image_path: str | None) -> Image.Image:
    if image_path is None:
        return Image.new("RGB", (96, 96), color=(64, 128, 192))

    path = Path(image_path).expanduser()
    return Image.open(path).convert("RGB")


def create_synthetic_audio() -> tuple[np.ndarray, int]:
    sr = 16000
    duration_s = 1.5
    frequency_hz = 440.0
    num_samples = max(1, int(sr * duration_s))
    t = np.arange(num_samples, dtype=np.float32) / float(sr)
    waveform = 0.2 * np.sin(2.0 * math.pi * frequency_hz * t)
    return waveform.astype(np.float32), sr


def load_or_create_audio(audio_path: str | None) -> tuple[np.ndarray, int]:
    if audio_path is None:
        return create_synthetic_audio()

    if sf is None:
        raise ImportError("soundfile is required to load audio files.")

    path = Path(audio_path).expanduser()
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    return np.asarray(audio, dtype=np.float32), int(sr)


def scenario_names(selected: str) -> list[str]:
    if selected == "all":
        return [
            "text_only",
            "image_only",
            "audio_only",
            "image_text",
            "audio_text",
            "image_audio",
            "image_audio_text",
        ]
    return [selected]


def infer_mm_limits(scenarios: list[str]) -> dict[str, int]:
    image_scenarios = {
        "image_only",
        "image_text",
        "image_audio",
        "image_audio_text",
    }
    audio_scenarios = {
        "audio_only",
        "audio_text",
        "image_audio",
        "image_audio_text",
    }
    needs_image = any(scenario in image_scenarios for scenario in scenarios)
    needs_audio = any(scenario in audio_scenarios for scenario in scenarios)
    return {"image": int(needs_image), "audio": int(needs_audio)}


def make_request(
    scenario: str,
    *,
    text: str,
    image: Image.Image,
    audio: tuple[np.ndarray, int],
) -> dict[str, Any]:
    if scenario == "text_only":
        return {"prompt": text}
    if scenario == "image_only":
        return {"prompt": IMAGE_PLACEHOLDER, "multi_modal_data": {"image": image}}
    if scenario == "audio_only":
        return {"prompt": AUDIO_PLACEHOLDER, "multi_modal_data": {"audio": audio}}
    if scenario == "image_text":
        return {
            "prompt": f"{text}\n{IMAGE_PLACEHOLDER}",
            "multi_modal_data": {"image": image},
        }
    if scenario == "audio_text":
        return {
            "prompt": f"{text}\n{AUDIO_PLACEHOLDER}",
            "multi_modal_data": {"audio": audio},
        }
    if scenario == "image_audio":
        return {
            "prompt": f"{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}",
            "multi_modal_data": {"image": image, "audio": audio},
        }
    if scenario == "image_audio_text":
        return {
            "prompt": f"{text}\n{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}",
            "multi_modal_data": {"image": image, "audio": audio},
        }

    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    image = load_or_create_image(args.image)
    audio = load_or_create_audio(args.audio)
    selected_scenarios = scenario_names(args.scenario)

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt=infer_mm_limits(selected_scenarios),
        mm_processor_kwargs=build_mm_processor_kwargs(args),
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    for scenario in selected_scenarios:
        request = make_request(
            scenario,
            text=args.text,
            image=image,
            audio=audio,
        )
        outputs = llm.generate([request], sampling_params=sampling_params)
        answer = outputs[0].outputs[0].text.strip()

        if not answer:
            raise RuntimeError(f"Apertus returned an empty answer for {scenario}.")
        if (
            args.expect_substring
            and args.expect_substring.lower() not in answer.lower()
        ):
            raise RuntimeError(
                "Apertus answer did not contain expected substring "
                f"{args.expect_substring!r}: {answer!r}"
            )

        print(f"=== Scenario: {scenario} ===")
        print(request["prompt"])
        print(answer)
        print()


if __name__ == "__main__":
    main()
