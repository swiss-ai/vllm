# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from transformers import ApertusConfig


class Apertus1p5Config(ApertusConfig):
    model_type = "apertus1p5"

    def __init__(
        self,
        text_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        text_config = dict(text_config or {})
        text_config.pop("model_type", None)
        kwargs.update(text_config)
        super().__init__(**kwargs)
