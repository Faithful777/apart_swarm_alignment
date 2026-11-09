"""Local Unsloth Qwen policy with GPU-batched bee decisions.

The policy accepts detached local observations, performs one deterministic
batched generation pass, and validates every action before returning any of them.
No API calls are made.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch

from beehive_sim import SYSTEM_PROMPT, validate_action


class BeeLocalQwenPolicy:
    label = "local_qwen35_08b_lora_batched"

    def __init__(self, model, tokenizer, max_new_tokens: int = 160,
                 batch_size: int = 16):
        if max_new_tokens < 32:
            raise ValueError("max_new_tokens must be at least 32")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.last_model = "unsloth/Qwen3.5-0.8B + trained LoRA adapter"
        self.last_response_id = None
        self.last_usage: dict[str, int] = {}
        self.last_usage_by_decision: list[dict[str, int]] = []
        self.last_batch_size = 0
        self.last_sub_batch_sizes: list[int] = []
        self.model.config.use_cache = True
        # Decoder-only generation with unequal prompt lengths requires left padding.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @classmethod
    def from_adapter(cls, adapter_dir: str | Path, max_seq_length: int = 3072,
                     max_new_tokens: int = 160, batch_size: int = 16):
        try:
            from unsloth import FastLanguageModel
        except ImportError as exc:
            raise RuntimeError("Install Unsloth in this runtime to load the adapter.") from exc
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir),
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=False,
            load_in_16bit=True,
            full_finetuning=False,
        )
        FastLanguageModel.for_inference(model)
        return cls(model, tokenizer, max_new_tokens=max_new_tokens,
                   batch_size=batch_size)

    def _generate_chunk(self, observations: list[dict[str, Any]]):
        conversations = [
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(
                    observation, ensure_ascii=False, separators=(",", ":"),
                    allow_nan=False,
                )},
            ]
            for observation in observations
        ]
        encoded = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {
            key: value.to(self.model.device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        prompt_width = inputs["input_ids"].shape[-1]
        input_counts = (
            inputs["attention_mask"].sum(dim=-1).tolist()
            if "attention_mask" in inputs
            else [prompt_width] * len(observations)
        )
        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        texts = self.tokenizer.batch_decode(
            generated[:, prompt_width:], skip_special_tokens=True
        )
        output_ids = generated[:, prompt_width:]
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        output_counts = []
        for row in output_ids:
            token_ids = row.tolist()
            if eos_id is not None and eos_id in token_ids:
                token_count = token_ids.index(eos_id) + 1
            else:
                if pad_id is not None:
                    while token_ids and token_ids[-1] == pad_id:
                        token_ids.pop()
                token_count = len(token_ids)
            output_counts.append(token_count)
        return texts, input_counts, output_counts

    def _generate_with_oom_split(self, observations: list[dict[str, Any]]):
        try:
            result = self._generate_chunk(observations)
            self.last_sub_batch_sizes.append(len(observations))
            return result
        except RuntimeError as exc:
            message = str(exc).lower()
            is_oom = "out of memory" in message or exc.__class__.__name__ == "OutOfMemoryError"
            if not is_oom or len(observations) == 1:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            midpoint = len(observations) // 2
            left = self._generate_with_oom_split(observations[:midpoint])
            right = self._generate_with_oom_split(observations[midpoint:])
            return tuple(a + b for a, b in zip(left, right))

    @staticmethod
    def _parse_action(text: str) -> dict[str, Any]:
        decoder = json.JSONDecoder()
        try:
            action, end = decoder.raw_decode(text.lstrip())
            leading = len(text) - len(text.lstrip())
            end += leading
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Local bee model returned invalid JSON at line {exc.lineno}, "
                f"column {exc.colno}: {exc.msg}"
            ) from exc
        if text[end:].strip():
            raise ValueError("Local bee model returned extra text after its action JSON.")
        if not isinstance(action, dict):
            raise ValueError("Local bee model output must be one JSON object.")
        schema_error = validate_action(action)
        if schema_error:
            raise ValueError(f"Local bee model returned an invalid action: {schema_error}")
        return action

    def decide_batch(self, observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not observations:
            self.last_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
            self.last_usage_by_decision = []
            self.last_batch_size = 0
            self.last_sub_batch_sizes = []
            return []
        self.last_response_id = None
        self.last_batch_size = len(observations)
        self.last_sub_batch_sizes = []
        texts: list[str] = []
        input_counts: list[int] = []
        output_counts: list[int] = []
        for start in range(0, len(observations), self.batch_size):
            chunk = observations[start:start + self.batch_size]
            generated = self._generate_with_oom_split(chunk)
            texts.extend(generated[0])
            input_counts.extend(int(n) for n in generated[1])
            output_counts.extend(int(n) for n in generated[2])

        # Validate the whole batch before the simulator applies any action.
        actions = []
        for index, text in enumerate(texts):
            try:
                actions.append(self._parse_action(text.strip()))
            except ValueError as exc:
                raise ValueError(f"Decision {index} in local inference batch failed: {exc}") from exc

        self.last_usage_by_decision = [
            {"input_tokens": input_count, "output_tokens": output_count,
             "total_tokens": input_count + output_count}
            for input_count, output_count in zip(input_counts, output_counts)
        ]
        self.last_usage = {
            key: sum(item[key] for item in self.last_usage_by_decision)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        }
        return actions

    def decide(self, observation: dict[str, Any]) -> dict[str, Any]:
        return self.decide_batch([observation])[0]
