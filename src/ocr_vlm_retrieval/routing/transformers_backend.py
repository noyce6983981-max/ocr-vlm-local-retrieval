"""Lazy local Transformers backend for the V19 model pilot."""

from __future__ import annotations

import importlib.util
from threading import Lock
from typing import Any

from ocr_vlm_retrieval.routing.prompt import build_intent_messages


class TransformersIntentBackend:
    """Keep one local instruction model warm and return its unedited text."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        max_new_tokens: int = 192,
        load_in_4bit: bool = False,
        local_files_only: bool = True,
        prompt_lookup_num_tokens: int | None = None,
    ) -> None:
        if not model_name_or_path.strip():
            raise ValueError("model_name_or_path must not be empty")
        if max_new_tokens < 32:
            raise ValueError("max_new_tokens must be at least 32")
        if prompt_lookup_num_tokens is not None and prompt_lookup_num_tokens < 1:
            raise ValueError("prompt_lookup_num_tokens must be positive")
        self._model_name_or_path = model_name_or_path
        self._max_new_tokens = max_new_tokens
        self._load_in_4bit = load_in_4bit
        self._local_files_only = local_files_only
        self._prompt_lookup_num_tokens = prompt_lookup_num_tokens
        self._tokenizer: Any = None
        self._model: Any = None
        self._load_lock = Lock()
        self._generation_lock = Lock()

    @property
    def name(self) -> str:
        quantization = "4bit" if self._load_in_4bit else "native"
        lookup = (
            f":prompt_lookup={self._prompt_lookup_num_tokens}"
            if self._prompt_lookup_num_tokens is not None
            else ""
        )
        return f"transformers:{self._model_name_or_path}:{quantization}{lookup}"

    def _ensure_loaded(self) -> None:
        already_loaded = self._model is not None
        if already_loaded:
            return
        with self._load_lock:
            already_loaded = self._model is not None
            if already_loaded:
                return
            try:
                from transformers import (
                    AutoModelForCausalLM,
                    AutoTokenizer,
                    BitsAndBytesConfig,
                )
            except ImportError as error:
                raise RuntimeError(
                    "The V19 Transformers backend requires transformers"
                ) from error

            model_kwargs: dict[str, Any] = {
                "device_map": "auto",
                "local_files_only": self._local_files_only,
                "torch_dtype": "auto",
            }
            if self._load_in_4bit:
                if importlib.util.find_spec("bitsandbytes") is None:
                    raise RuntimeError(
                        "4-bit loading requested but bitsandbytes is unavailable"
                    )
                import torch

                quantization_factory: Any = BitsAndBytesConfig
                model_kwargs["quantization_config"] = quantization_factory(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )

            tokenizer: Any = AutoTokenizer.from_pretrained(
                self._model_name_or_path,
                local_files_only=self._local_files_only,
                use_fast=True,
            )
            model: Any = AutoModelForCausalLM.from_pretrained(
                self._model_name_or_path,
                **model_kwargs,
            )
            model.eval()
            self._tokenizer = tokenizer
            self._model = model

    def generate_intent_json(self, query: str) -> str:
        """Generate once in Qwen non-thinking mode; schema parsing is external."""

        self._ensure_loaded()
        tokenizer = self._tokenizer
        model = self._model
        messages = build_intent_messages(query)
        assistant_prefill = "{"
        messages.append({"role": "assistant", "content": assistant_prefill})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            continue_final_message=True,
            enable_thinking=False,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = inputs.to(model.device)

        import torch

        with self._generation_lock, torch.inference_mode():
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self._max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.eos_token_id,
            }
            if self._prompt_lookup_num_tokens is not None:
                generation_kwargs["prompt_lookup_num_tokens"] = (
                    self._prompt_lookup_num_tokens
                )
            outputs = model.generate(
                **inputs,
                **generation_kwargs,
            )
        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_tokens = outputs[0][prompt_tokens:]
        continuation = str(
            tokenizer.decode(generated_tokens, skip_special_tokens=True)
        ).strip()
        return assistant_prefill + continuation
