"""Unified LLM client supporting OpenAI and Anthropic backends."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


@dataclass
class LLMRequest:
    """Request to an LLM."""

    system_prompt: str
    user_prompt: str
    developer_prompt: str = ""  # Task instructions, appended to system for some backends
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4000
    response_format: Literal["json", "text"] = "json"

    def prompt_hash(self) -> str:
        """SHA256 hash of combined prompts for reproducibility tracking."""
        combined = f"{self.system_prompt}|{self.developer_prompt}|{self.user_prompt}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


@dataclass
class LLMResponse:
    """Response from an LLM."""

    content: str | dict  # Parsed JSON or raw text
    model_id: str
    prompt_hash: str
    timestamp: str  # ISO 8601
    usage: dict = field(default_factory=dict)  # Token counts
    raw_response: str = ""  # For debugging


class LLMClient:
    """Unified client for OpenAI and Anthropic APIs."""

    def __init__(
        self,
        backend: Literal["openai", "anthropic"] = "openai",
        api_key: str | None = None,
    ):
        """Initialize client with specified backend.

        Args:
            backend: Which API to use ("openai" or "anthropic")
            api_key: API key (defaults to env var OPENAI_API_KEY or ANTHROPIC_API_KEY)
        """
        self.backend = backend
        self._client: Any = None
        self._api_key = api_key

    def _get_client(self) -> Any:
        """Lazy-load the appropriate client."""
        if self._client is not None:
            return self._client

        if self.backend == "openai":
            try:
                import openai
            except ImportError:
                raise ImportError(
                    "openai package not installed. Install with: pip install squiggle-core[llm]"
                )

            api_key = self._api_key or os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set and no api_key provided"
                )
            self._client = openai.OpenAI(api_key=api_key)

        elif self.backend == "anthropic":
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package not installed. Install with: pip install squiggle-core[llm]"
                )

            api_key = self._api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set and no api_key provided"
                )
            self._client = anthropic.Anthropic(api_key=api_key)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

        return self._client

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send request to LLM and return response.

        Args:
            request: LLMRequest with prompts and parameters

        Returns:
            LLMResponse with content and metadata
        """
        client = self._get_client()
        timestamp = datetime.now(timezone.utc).isoformat()
        prompt_hash = request.prompt_hash()

        if self.backend == "openai":
            return self._complete_openai(client, request, timestamp, prompt_hash)
        else:
            return self._complete_anthropic(client, request, timestamp, prompt_hash)

    def _complete_openai(
        self,
        client: Any,
        request: LLMRequest,
        timestamp: str,
        prompt_hash: str,
    ) -> LLMResponse:
        """Complete request using OpenAI API."""
        # Build messages
        messages = [
            {"role": "system", "content": request.system_prompt},
        ]

        # Add developer prompt as a separate system message if provided
        if request.developer_prompt:
            messages.append({"role": "system", "content": request.developer_prompt})

        messages.append({"role": "user", "content": request.user_prompt})

        # Build API call kwargs
        kwargs: dict[str, Any] = {
            "model": request.model,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "messages": messages,
        }

        # Request JSON response format if specified
        if request.response_format == "json":
            kwargs["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**kwargs)

        raw_content = response.choices[0].message.content or ""

        # Parse JSON if requested
        if request.response_format == "json":
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                # Return raw if JSON parsing fails
                content = raw_content
        else:
            content = raw_content

        return LLMResponse(
            content=content,
            model_id=response.model,
            prompt_hash=prompt_hash,
            timestamp=timestamp,
            usage={
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0,
            },
            raw_response=raw_content,
        )

    def _complete_anthropic(
        self,
        client: Any,
        request: LLMRequest,
        timestamp: str,
        prompt_hash: str,
    ) -> LLMResponse:
        """Complete request using Anthropic API."""
        # Anthropic uses system as a separate parameter
        system = request.system_prompt
        if request.developer_prompt:
            system = f"{system}\n\n{request.developer_prompt}"

        # Build user message with JSON instruction if needed
        user_content = request.user_prompt
        if request.response_format == "json":
            user_content = f"{user_content}\n\nRespond with valid JSON only."

        response = client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_content}],
        )

        raw_content = response.content[0].text if response.content else ""

        # Parse JSON if requested
        if request.response_format == "json":
            try:
                content = json.loads(raw_content)
            except json.JSONDecodeError:
                content = raw_content
        else:
            content = raw_content

        return LLMResponse(
            content=content,
            model_id=response.model,
            prompt_hash=prompt_hash,
            timestamp=timestamp,
            usage={
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
            },
            raw_response=raw_content,
        )
