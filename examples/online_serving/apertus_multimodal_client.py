# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for served Apertus text/image/audio inference.

Launch a server first, for example:

    MODEL=/capstor/store/cscs/swissai/infra01/MLLM/ablations/...
    TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct
    vllm serve "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      --max-model-len 8192 \
      --tensor-parallel-size 4 \
      --limit-mm-per-prompt '{"image": 1, "audio": 1}' \
      --mm-processor-kwargs '{
        "apertus_vision_tokenizer_device": "cuda",
        "apertus_audio_tokenizer_path": "/capstor/.../wavtokenizer"
      }'

Then run:

    python examples/online_serving/apertus_multimodal_client.py \
      --scenario all \
      --image /path/to/image.jpg \
      --audio /path/to/audio.wav
"""

from __future__ import annotations

import base64
import math
import mimetypes
import wave
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_MODEL = (
    "/capstor/store/cscs/swissai/infra01/MLLM/ablations/"
    "apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-"
    "seqlen8192-s2onlytxtloss/HF"
)
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


def image_to_data_url(path: str) -> str:
    image_path = Path(path).expanduser()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def synthetic_audio_wav_base64() -> tuple[str, str]:
    sr = 16000
    duration_s = 1.5
    frequency_hz = 440.0
    num_samples = max(1, int(sr * duration_s))
    t = np.arange(num_samples, dtype=np.float32) / float(sr)
    waveform = 0.2 * np.sin(2.0 * math.pi * frequency_hz * t)
    pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)

    buf = BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sr)
        wav_file.writeframes(pcm16.tobytes())

    return base64.b64encode(buf.getvalue()).decode("utf-8"), "wav"


def audio_to_base64_and_format(path: str | None) -> tuple[str, str]:
    if path is None:
        return synthetic_audio_wav_base64()

    audio_path = Path(path).expanduser()
    audio_bytes = audio_path.read_bytes()
    ext = audio_path.suffix.lower().lstrip(".")
    audio_format = ext or "wav"
    return base64.b64encode(audio_bytes).decode("utf-8"), audio_format


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--scenario", default="image_text", choices=SCENARIOS)
    parser.add_argument("--text", default="Describe the provided content concisely.")
    parser.add_argument("--image", default=None, help="Optional image path.")
    parser.add_argument("--audio", default=None, help="Optional audio path.")
    parser.add_argument("--host", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--expect-substring",
        default=None,
        help="Optional case-insensitive substring that must appear in answers.",
    )
    return parser.parse_args()


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


def build_content_parts(
    scenario: str,
    *,
    text: str,
    image_path: str | None,
    audio_path: str | None,
) -> list[dict[str, Any]]:
    if scenario == "text_only":
        return [{"type": "text", "text": text}]

    if scenario == "image_only":
        if image_path is None:
            raise ValueError("Scenario image_only requires --image.")
        return [
            {"type": "text", "text": IMAGE_PLACEHOLDER},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ]

    if scenario == "audio_only":
        audio_b64, audio_format = audio_to_base64_and_format(audio_path)
        return [
            {"type": "text", "text": AUDIO_PLACEHOLDER},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": audio_format},
            },
        ]

    if scenario == "image_text":
        if image_path is None:
            raise ValueError("Scenario image_text requires --image.")
        return [
            {"type": "text", "text": f"{text}\n{IMAGE_PLACEHOLDER}"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
        ]

    if scenario == "audio_text":
        audio_b64, audio_format = audio_to_base64_and_format(audio_path)
        return [
            {"type": "text", "text": f"{text}\n{AUDIO_PLACEHOLDER}"},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": audio_format},
            },
        ]

    if scenario == "image_audio":
        if image_path is None:
            raise ValueError("Scenario image_audio requires --image.")
        audio_b64, audio_format = audio_to_base64_and_format(audio_path)
        return [
            {"type": "text", "text": f"{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}"},
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": audio_format},
            },
        ]

    if scenario == "image_audio_text":
        if image_path is None:
            raise ValueError("Scenario image_audio_text requires --image.")
        audio_b64, audio_format = audio_to_base64_and_format(audio_path)
        return [
            {
                "type": "text",
                "text": f"{text}\n{IMAGE_PLACEHOLDER}\n{AUDIO_PLACEHOLDER}",
            },
            {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
            {
                "type": "input_audio",
                "input_audio": {"data": audio_b64, "format": audio_format},
            },
        ]

    raise ValueError(f"Unsupported scenario: {scenario}")


def main() -> None:
    args = parse_args()
    client = OpenAI(api_key=args.api_key, base_url=args.host)

    for scenario in scenario_names(args.scenario):
        content = build_content_parts(
            scenario,
            text=args.text,
            image_path=args.image,
            audio_path=args.audio,
        )
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": content}],
            max_completion_tokens=args.max_completion_tokens,
            temperature=args.temperature,
        )
        answer = (response.choices[0].message.content or "").strip()

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
        print(answer)
        print()


if __name__ == "__main__":
    main()
