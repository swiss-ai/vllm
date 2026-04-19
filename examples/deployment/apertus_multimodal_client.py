# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenAI-compatible client for served Apertus image+text inference.

Launch a server first, for example:

    MODEL=/capstor/store/cscs/swissai/infra01/MLLM/ablations/...
    TOKENIZER=/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_instruct
    vllm serve "$MODEL" \
      --tokenizer "$TOKENIZER" \
      --trust-remote-code \
      --max-model-len 8192 \
      --tensor-parallel-size 4 \
      --limit-mm-per-prompt '{"image": 1}' \
      --mm-processor-kwargs '{"apertus_vision_tokenizer_device": "cuda"}'

Then run:

    python examples/online_serving/apertus_multimodal_client.py \
      --image /path/to/image.jpg
"""

import base64
import mimetypes
from pathlib import Path

from openai import OpenAI

from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_MODEL = (
    "/capstor/store/cscs/swissai/infra01/MLLM/ablations/"
    "apertus-8b-img-SFT-32nodes-gbs512-mbs1-steps8030-img-text-"
    "seqlen8192-s2onlytxtloss/HF"
)
DEFAULT_PROMPT = "\nDescribe the image.\n<|image|>, is this a picture of github logo?"


def image_to_data_url(path: str) -> str:
    image_path = Path(path).expanduser()
    mime_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{image_base64}"


def parse_args():
    parser = FlexibleArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--image", required=True, help="Path to an input image.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--host", default="http://localhost:8000/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--max-completion-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--expect-substring",
        default=None,
        help="Optional case-insensitive substring that must appear in the answer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_image_placeholders = args.prompt.count("<|image|>")
    if num_image_placeholders != 1:
        raise ValueError(
            "This example sends exactly one image, so the prompt must contain "
            f"exactly one <|image|> placeholder. Found {num_image_placeholders}."
        )

    client = OpenAI(api_key=args.api_key, base_url=args.host)
    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_to_data_url(args.image)},
                    },
                ],
            }
        ],
        max_completion_tokens=args.max_completion_tokens,
        temperature=args.temperature,
    )
    answer = (response.choices[0].message.content or "").strip()

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
