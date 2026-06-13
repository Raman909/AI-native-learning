from __future__ import annotations

import logging
import time
from typing import Any

import requests

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dependency path
    class _DummyGenerativeModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def generate_content(self, *args, **kwargs):
            raise RuntimeError("google.generativeai is not installed")

    class _DummyGenAI:
        @staticmethod
        def configure(*args, **kwargs):
            return None

        GenerativeModel = _DummyGenerativeModel

    genai = _DummyGenAI()

try:
    from google.api_core.exceptions import ResourceExhausted
except Exception:  # pragma: no cover - optional dependency path
    ResourceExhausted = Exception


class LLMRouter:
    def __init__(self, gemini_api_key: str, hf_api_key: str, gemini_model: str, hf_model: str):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.gemini_api_key = gemini_api_key or ""
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)
        self.hf_api_key = hf_api_key or ""
        self.gemini_model = self._normalize_gemini_model(gemini_model)
        self.hf_model = hf_model
        self.active_provider = "gemini"
        self.gemini_quota_reset_time: float | None = None

    def _normalize_gemini_model(self, model_name: str) -> str:
        normalized = (model_name or "").strip()
        deprecated_aliases = {
            "gemini-1.5-flash": "gemini-2.5-flash",
            "gemini-1.5-flash-8b": "gemini-2.5-flash-lite",
            "gemini-1.5-pro": "gemini-2.5-pro",
            "gemini-2.0-flash": "gemini-2.5-flash",
            "gemini-2.0-flash-001": "gemini-2.5-flash",
            "gemini-2.0-flash-lite": "gemini-2.5-flash-lite",
            "gemini-2.0-flash-lite-001": "gemini-2.5-flash-lite",
        }
        return deprecated_aliases.get(normalized, normalized or "gemini-2.5-flash")

    def _gemini_candidates(self) -> list[str]:
        candidates = [
            self._normalize_gemini_model(self.gemini_model),
            "gemini-2.5-flash",
            "gemini-2.5-flash-lite",
        ]
        deduped: list[str] = []
        for candidate in candidates:
            if candidate not in deduped:
                deduped.append(candidate)
        return deduped

    def _call_gemini(self, prompt: str) -> str:
        generation_config = {"temperature": 0.2, "max_output_tokens": 1024}
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        last_exception: Exception | None = None

        for candidate in self._gemini_candidates():
            try:
                model = genai.GenerativeModel(candidate)
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                )
                self.gemini_model = candidate
                return response.text
            except Exception as exception:
                last_exception = exception
                if "not found" not in str(exception).lower():
                    raise

        if last_exception is not None:
            raise last_exception
        raise RuntimeError("No Gemini model candidates available")

    def _call_huggingface(self, prompt: str) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
        headers = {"Authorization": f"Bearer {self.hf_api_key}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 1024,
                "temperature": 0.2,
                "return_full_text": False,
                "do_sample": True,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        if response.status_code == 503:
            time.sleep(20)
            response = requests.post(url, headers=headers, json=payload, timeout=60)

        response.raise_for_status()
        response_data: Any = response.json()
        if isinstance(response_data, list) and response_data:
            first_item = response_data[0]
            if isinstance(first_item, dict) and "generated_text" in first_item:
                return first_item["generated_text"]
        if isinstance(response_data, dict) and "generated_text" in response_data:
            return response_data["generated_text"]
        raise ValueError("Unexpected HuggingFace response payload")

    def _is_quota_error(self, exception: Exception) -> bool:
        error_text = str(exception).lower()
        keywords = ["quota", "429", "resourceexhausted", "rate limit", "resource_exhausted"]
        return any(keyword.lower() in error_text for keyword in keywords)

    def generate(self, prompt: str) -> tuple[str, str]:
        can_retry_gemini = self.gemini_quota_reset_time is None or time.time() >= self.gemini_quota_reset_time
        if self.active_provider == "gemini" or can_retry_gemini:
            try:
                text = self._call_gemini(prompt)
                self.active_provider = "gemini"
                self.gemini_quota_reset_time = None
                return text, "gemini"
            except ResourceExhausted as exception:
                if self._is_quota_error(exception):
                    self.logger.warning("Gemini quota hit, switching to HuggingFace fallback")
                    self.active_provider = "huggingface"
                    self.gemini_quota_reset_time = time.time() + 3600
                else:
                    self.logger.error("Gemini failed with a non-quota error", exc_info=True)
            except Exception:
                self.logger.error("Gemini failed, falling back to HuggingFace", exc_info=True)

        try:
            text = self._call_huggingface(prompt)
            self.active_provider = "huggingface"
            return text, "huggingface"
        except Exception as exception:
            self.logger.error("HuggingFace fallback failed", exc_info=True)
            raise Exception("Both LLM providers failed") from exception

    def get_status(self) -> dict:
        return {
            "active_provider": self.active_provider,
            "gemini_model": self.gemini_model,
            "hf_model": self.hf_model,
            "gemini_retry_after": self.gemini_quota_reset_time,
        }
