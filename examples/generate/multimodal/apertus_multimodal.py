# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline image+text inference for Apertus.

This example exercises the Apertus multimodal path:

raw image + prompt with <|image|>
  -> Emu3.5 image tokenizer
  -> serialized Apertus image-token text
  -> normal Apertus tokenizer
  -> normal Apertus LM forward

Example:
    python examples/offline_inference/apertus_multimodal.py \
      --image /path/to/image.jpg \
      --tensor-parallel-size 4
"""

from pathlib import Path

from PIL import Image

from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_MODEL = (
    "/capstor/store/cscs/swissai/infra01/MLLM/ablations/"
    "apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-"
    "seqlen8192-s2onlytxtloss/HF"
)
DEFAULT_TOKENIZER = (
    "/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/"
    "apertus_emu3.5_instruct"
)
DEFAULT_PROMPT = (
    "User: <|image|>\n"
    "Describe the image in one concise sentence.\n"
    "Assistant:"
)


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
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
    parser.add_argument(
        "--expect-substring",
        default=None,
        help="Optional case-insensitive substring that must appear in the answer.",
    )
    return parser.parse_args()


def build_mm_processor_kwargs(args) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "apertus_vision_tokenizer_device": args.vision_tokenizer_device,
        "apertus_vq_hub": args.vq_hub,
    }
    if args.emu35_codebase:
        kwargs["apertus_emu35_codebase"] = args.emu35_codebase
    return kwargs


def main() -> None:
    args = parse_args()
    num_image_placeholders = args.prompt.count("<|image|>")
    if num_image_placeholders != 1:
        raise ValueError(
            "This example sends exactly one image, so the prompt must contain "
            f"exactly one <|image|> placeholder. Found {num_image_placeholders}."
        )

    image_path = Path(args.image).expanduser()
    image = Image.open(image_path).convert("RGB")

    llm = LLM(
        model=args.model,
        tokenizer=args.tokenizer,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        limit_mm_per_prompt={"image": num_image_placeholders},
        mm_processor_kwargs=build_mm_processor_kwargs(args),
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    request = {
        "prompt": args.prompt,
        "multi_modal_data": {"image": image},
    }
    outputs = llm.generate([request], sampling_params=sampling_params)
    answer = outputs[0].outputs[0].text.strip()

    if not answer:
        raise RuntimeError("Apertus returned an empty answer.")
    if args.expect_substring and args.expect_substring.lower() not in answer.lower():
        raise RuntimeError(
            "Apertus answer did not contain expected substring "
            f"{args.expect_substring!r}: {answer!r}"
        )

    print(answer)


if __name__ == "__main__":
    main()
