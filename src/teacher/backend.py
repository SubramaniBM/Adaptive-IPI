"""
src/teacher/backend.py

Inference backend abstraction for the teacher model.

Separates the inference mechanism (vLLM / Transformers / API) from
the annotation logic (Change #2). This allows swapping the teacher
model or backend without touching the annotation pipeline.

Each backend implements the same interface: accept a list of messages
(chat format) and return generated text.
"""

import abc
from typing import Any, Optional

from src.core.enums import TeacherBackend
from src.utils.logging import get_logger

logger = get_logger(__name__)


class BaseTeacherBackend(abc.ABC):
    """Abstract base class for teacher inference backends.

    All backends must implement the ``generate`` method, which accepts
    a list of chat-formatted messages and returns the model's response.
    """

    @abc.abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a response from the teacher model.

        Args:
            messages: Chat-formatted messages (list of {"role": ..., "content": ...}).
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        ...

    @abc.abstractmethod
    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Generate responses for a batch of inputs.

        Default implementation falls back to sequential generation.
        Backends with native batch support should override this.

        Args:
            batch_messages: List of chat-formatted message lists.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            List of generated text strings.
        """
        ...


class VLLMBackend(BaseTeacherBackend):
    """Teacher inference via vLLM offline batch processing.

    Requires: pip install vllm
    Requires: GPU with sufficient VRAM for the teacher model.
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialise the vLLM backend.

        Args:
            model_id: HuggingFace model identifier.
            **kwargs: Additional arguments passed to vLLM LLM constructor
                (e.g., tensor_parallel_size, gpu_memory_utilization).
        """
        try:
            from vllm import LLM  # type: ignore
        except ImportError:
            raise ImportError(
                "vLLM is required for VLLMBackend. Install with:\n"
                "    pip install vllm"
            )

        self.model_id = model_id
        logger.info(f"Initialising vLLM backend with model: {model_id}")
        self.llm = LLM(model=model_id, **kwargs)
        logger.info("vLLM backend ready")

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single response using vLLM."""
        results = self.generate_batch([messages], temperature=temperature, max_tokens=max_tokens)
        return results[0]

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Generate responses for a batch using vLLM's native batching."""
        from vllm import SamplingParams  # type: ignore

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        outputs = self.llm.chat(
            messages=batch_messages,
            sampling_params=sampling_params,
        )

        return [output.outputs[0].text for output in outputs]


class TransformersBackend(BaseTeacherBackend):
    """Teacher inference via HuggingFace Transformers.

    Slower than vLLM but works on any hardware (including CPU, though
    CPU inference with a 32B model is impractical).
    """

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        """Initialise the Transformers backend.

        Args:
            model_id: HuggingFace model identifier.
            **kwargs: Additional arguments passed to pipeline constructor.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        logger.info(f"Initialising Transformers backend with model: {model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
            **kwargs,
        )
        self.model.eval()
        logger.info("Transformers backend ready")

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Generate a single response using Transformers."""
        import torch

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None,
                do_sample=temperature > 0,
            )

        # Decode only the generated portion
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> list[str]:
        """Generate responses sequentially (Transformers lacks native batching for chat)."""
        return [
            self.generate(messages, temperature=temperature, max_tokens=max_tokens)
            for messages in batch_messages
        ]


class OllamaBackend(BaseTeacherBackend):
    """Teacher inference via local Ollama API."""

    def __init__(self, model_id: str, **kwargs: Any) -> None:
        self.model_id = model_id
        self.base_url = kwargs.get("base_url", "http://localhost:11434/api/chat")

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        import requests
        payload = {
            "model": self.model_id,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        import concurrent.futures
        
        def _make_request():
            response = requests.post(self.base_url, json=payload, timeout=120)
            response.raise_for_status()
            return response.json()["message"]["content"]
            
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_make_request)
        try:
            return future.result(timeout=130)
        except concurrent.futures.TimeoutError:
            # Do NOT shutdown the executor because it will block forever
            raise TimeoutError("Ollama API completely hung (strict wall-clock timeout exceeded).")

    def generate_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> list[str]:
        return [
            self.generate(messages, temperature=temperature, max_tokens=max_tokens)
            for messages in batch_messages
        ]


def create_backend(
    backend_type: str,
    model_id: str,
    **kwargs: Any,
) -> BaseTeacherBackend:
    """Factory function to create a teacher inference backend.

    Args:
        backend_type: One of "vllm", "transformers", "api", "ollama".
        model_id: HuggingFace model identifier.
        **kwargs: Backend-specific arguments.

    Returns:
        An initialised backend instance.

    Raises:
        ValueError: If the backend type is not supported.
    """
    backend_type = backend_type.lower()

    if backend_type == TeacherBackend.VLLM.value:
        return VLLMBackend(model_id, **kwargs)
    elif backend_type == TeacherBackend.TRANSFORMERS.value:
        return TransformersBackend(model_id, **kwargs)
    elif backend_type == "ollama":
        return OllamaBackend(model_id, **kwargs)
    elif backend_type == TeacherBackend.API.value:
        raise NotImplementedError(
            "API backend is not yet implemented. "
            "Implement a subclass of BaseTeacherBackend for your API provider."
        )
    else:
        raise ValueError(
            f"Unsupported backend type: {backend_type!r}. "
            f"Supported: {[b.value for b in TeacherBackend]} + ['ollama']"
        )
