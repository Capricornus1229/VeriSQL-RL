"""Asynchronous client for the local vLLM OpenAI-compatible server."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import httpx


class ModelUnavailableError(RuntimeError):
    """The configured vLLM service or tokenizer cannot be used."""


class PromptTooLongError(ValueError):
    """The rendered prompt plus generation budget exceeds model context."""


@dataclass(frozen=True)
class RenderedPrompt:
    text: str
    input_tokens: int


class VLLMClient:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.http: httpx.AsyncClient | None = None
        self.tokenizer: Any = None

    async def startup(self) -> None:
        from transformers import AutoTokenizer

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config["model"]["tokenizer_path"],
                use_fast=True,
            )
        except Exception as error:
            raise ModelUnavailableError(
                "The configured Qwen3 tokenizer could not be loaded."
            ) from error
        self.http = httpx.AsyncClient(
            timeout=self.config["model"]["request_timeout_seconds"],
            trust_env=False,
        )

    async def close(self) -> None:
        if self.http is not None:
            await self.http.aclose()
            self.http = None

    async def health(self) -> bool:
        if self.http is None:
            return False
        try:
            response = await self.http.get(
                self.config["model"]["vllm_health_url"]
            )
            response.raise_for_status()
            body = response.json()
            data = body.get("data", []) if isinstance(body, dict) else []
            model_ids = {
                item.get("id") for item in data if isinstance(item, dict)
            }
            return self.config["model"]["served_adapter_name"] in model_ids
        except (httpx.HTTPError, TypeError, ValueError):
            return False

    def render_prompt(self, messages: list[dict[str, str]]) -> RenderedPrompt:
        if self.tokenizer is None:
            raise ModelUnavailableError("The Qwen3 tokenizer is not loaded.")
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.config["generation"]["enable_thinking"],
        )
        input_tokens = len(
            self.tokenizer(text, add_special_tokens=False)["input_ids"]
        )
        max_new_tokens = self.config["generation"]["max_new_tokens"]
        if input_tokens + max_new_tokens > self.config["model"]["max_model_len"]:
            raise PromptTooLongError(
                f"Rendered prompt ({input_tokens} tokens) plus max_new_tokens "
                f"({max_new_tokens}) exceeds max_model_len "
                f"({self.config['model']['max_model_len']})."
            )
        return RenderedPrompt(text=text, input_tokens=input_tokens)

    async def _generate(
        self,
        prompt: RenderedPrompt,
        *,
        count: int,
        temperature: float,
        top_p: float,
        top_k: int,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        if self.http is None or self.tokenizer is None:
            raise ModelUnavailableError("The vLLM client has not been started.")

        payload: dict[str, Any] = {
            "model": self.config["model"]["served_adapter_name"],
            "prompt": prompt.text,
            "n": count,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": self.config["generation"]["max_new_tokens"],
            "logprobs": 1,
            "add_special_tokens": False,
        }
        if seed is not None:
            payload["seed"] = seed

        endpoint = (
            self.config["model"]["vllm_base_url"].rstrip("/")
            + "/completions"
        )
        try:
            response = await self.http.post(endpoint, json=payload)
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            raise ModelUnavailableError(f"vLLM request failed: {error}") from error

        if not isinstance(body, dict):
            raise ModelUnavailableError("vLLM response was not a JSON object.")
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelUnavailableError("vLLM response did not contain choices.")

        generated: list[dict[str, Any]] = []
        if any(not isinstance(choice, dict) for choice in choices):
            raise ModelUnavailableError("vLLM response contained an invalid choice.")
        try:
            ordered_choices = sorted(
                choices,
                key=lambda item: int(item.get("index", 0)),
            )
        except (TypeError, ValueError) as error:
            raise ModelUnavailableError(
                "vLLM response contained an invalid choice index."
            ) from error
        for choice in ordered_choices:
            raw_output = str(choice.get("text", ""))
            logprobs = choice.get("logprobs") or {}
            if not isinstance(logprobs, dict):
                raise ModelUnavailableError(
                    "vLLM response contained invalid token logprobs."
                )
            values = logprobs.get("token_logprobs") or []
            if not isinstance(values, list):
                raise ModelUnavailableError(
                    "vLLM response contained invalid token logprobs."
                )
            token_logprobs = []
            for value in values:
                if value is None:
                    continue
                try:
                    score = float(value)
                except (TypeError, ValueError) as error:
                    raise ModelUnavailableError(
                        "vLLM response contained invalid token logprobs."
                    ) from error
                if math.isfinite(score):
                    token_logprobs.append(score)
            tokens = logprobs.get("tokens") or []
            if not isinstance(tokens, list):
                raise ModelUnavailableError(
                    "vLLM response contained an invalid token list."
                )
            completion_tokens = (
                len(tokens)
                if tokens
                else len(
                    self.tokenizer(raw_output, add_special_tokens=False)["input_ids"]
                )
            )
            generated.append(
                {
                    "raw_output": raw_output,
                    "completion_tokens": completion_tokens,
                    "mean_logprob": (
                        sum(token_logprobs) / len(token_logprobs)
                        if token_logprobs
                        else None
                    ),
                }
            )

        if len(generated) != count:
            raise ModelUnavailableError(
                f"vLLM returned {len(generated)} choices; expected {count}."
            )
        return generated

    async def generate_greedy(
        self, prompt: RenderedPrompt
    ) -> list[dict[str, Any]]:
        settings = self.config["generation"]["fast"]
        return await self._generate(
            prompt,
            count=settings["candidates"],
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
        )

    async def generate_sampled(
        self,
        prompt: RenderedPrompt,
        count: int,
        seed: int,
    ) -> list[dict[str, Any]]:
        settings = self.config["generation"]["accurate"]
        return await self._generate(
            prompt,
            count=count,
            temperature=settings["temperature"],
            top_p=settings["top_p"],
            top_k=settings["top_k"],
            seed=seed,
        )
