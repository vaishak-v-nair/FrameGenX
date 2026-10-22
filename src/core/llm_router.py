import os
from dotenv import load_dotenv
load_dotenv()
import json
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging

from openai import OpenAI, RateLimitError

logger = logging.getLogger(__name__)

# Per-request timeout for every provider call. 15s was measured too tight for
# NIM cold starts (verified 2026-08-23: request timed out, fell to Groq).
LLM_TIMEOUT_S = float(os.getenv("LLM_TIMEOUT_SECONDS", "30.0"))

class ProviderConfig(BaseModel):
    name: str
    api_key_env_var: Optional[str] = None
    base_url: str
    model: str
    vision_model: Optional[str] = None
    priority: int
    cooldown_until: float = 0.0

class LLMRouter:
    # Model IDs verified live against each provider's catalog on 2026-08-23
    # (the previous hardcoded IDs — llama-3.1-405b, llama-3.3-70b-versatile,
    # gemini-1.5-pro — were all retired and returned 404 model_not_found).
    # Every ID is overridable via env so future deprecations are a .env edit,
    # not a code change.
    def __init__(self):
        self.last_failures = []
        # Priority by MEASURED latency (2026-08-24): NIM llama-3.3-70b timed
        # out at 15s AND 30s in every run today (cold starts), while Groq
        # gpt-oss-120b answered in <5s every time. Groq first; NIM second
        # chance after Groq cooldowns. Env-overridable models as before.
        self.providers = [
            ProviderConfig(
                name="Groq",
                api_key_env_var="GROQ_API_KEY",
                base_url="https://api.groq.com/openai/v1",
                # Groq retired every llama-* id (404) as of 2026-08;
                # gpt-oss-120b is their current flagship text model.
                # gpt-oss has no vision variant -> vision falls through to others.
                model=os.getenv("LLM_MODEL_GROQ", "openai/gpt-oss-120b"),
                vision_model=os.getenv("LLM_VISION_GROQ") or None,
                priority=1,
                requires_base64_vision=True
            ),
            ProviderConfig(
                name="Nvidia",
                api_key_env_var="NVIDIA_NIM_KEY",
                base_url="https://integrate.api.nvidia.com/v1",
                model=os.getenv("LLM_MODEL_NVIDIA", "meta/llama-3.3-70b-instruct"),
                vision_model=os.getenv(
                    "LLM_VISION_NVIDIA", "meta/llama-3.2-90b-vision-instruct"
                ),
                priority=2
            ),
            ProviderConfig(
                name="Gemini",
                api_key_env_var="GOOGLE_AI_STUDIO_API_KEY",
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                # gemini-flash-latest is Google's stable alias for the current
                # Flash generation (vision-capable, json_object-capable).
                model=os.getenv("LLM_MODEL_GEMINI", "gemini-flash-latest"),
                vision_model=os.getenv("LLM_VISION_GEMINI", "gemini-flash-latest"),
                priority=4
            ),
            ProviderConfig(
                name="DeepSeek",
                api_key_env_var="DEEPSEEK_API_KEY",
                base_url="https://api.deepseek.com",
                model=os.getenv("LLM_MODEL_DEEPSEEK", "deepseek-chat"),
                vision_model=os.getenv("LLM_VISION_DEEPSEEK", None),
                priority=2
            ),
            ProviderConfig(
                name="Ollama",
                api_key_env_var=None,  # Local, no key required
                base_url="http://localhost:11434/v1",
                model=os.getenv("LLM_MODEL_OLLAMA", "llama3"),
                vision_model=os.getenv("LLM_VISION_OLLAMA", "llava"),
                priority=0  # Highest priority to try local first!
            )
        ]
        
        # Sort by priority (1 is highest)
        self.providers.sort(key=lambda x: x.priority)
        
        # Verify at least one key is available
        self.available_providers = []
        for p in self.providers:
            if not p.api_key_env_var or os.getenv(p.api_key_env_var):
                self.available_providers.append(p)
                
        if not self.available_providers:
            raise ValueError("No LLM API keys found. Please set NVIDIA_NIM_KEY, GROQ_API_KEY, or GOOGLE_AI_STUDIO_API_KEY.")

    def _active_providers(self, require_vision: bool = False) -> List[ProviderConfig]:
        """Snapshot eligible providers; try each at most once per request."""
        now = time.time()
        return [p for p in self.available_providers
                if (not require_vision or p.vision_model)
                and now >= p.cooldown_until]

    def _apply_cooldown(self, provider_name: str, seconds: float = 60.0):
        for p in self.providers:
            if p.name == provider_name:
                p.cooldown_until = time.time() + seconds
                print(f"[llm_router] {provider_name} cooling down for {seconds}s.")
                break

    def _record_failure(self, provider, error):
        """SDK error bodies may echo API-key fragments; retain safe fields only."""
        status = getattr(error, "status_code", None)
        status = status if type(status) is int and 100 <= status <= 599 else None
        failure = {"provider": provider.name, "error_type": type(error).__name__, "http_status": status}
        self.last_failures.append(failure)
        return failure["error_type"] + (f" (HTTP {status})" if status else "")

    def generate_json(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> Dict[str, Any]:
        """Attempt to generate JSON, falling back through providers on RateLimit."""
        self.last_failures = []
        for provider in self._active_providers():
            try:
                client = OpenAI(
                    base_url=provider.base_url,
                    api_key=os.getenv(provider.api_key_env_var) if provider.api_key_env_var else "ollama",
                    timeout=LLM_TIMEOUT_S,
                    max_retries=0,
                )
                print(f"[llm_router] Routing request to {provider.name} ({provider.model})...")
                
                # Note: Some models support json_object natively. 
                # If the provider is Gemini via OpenAI API, they support response_format={"type": "json_object"} natively now too.
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                # Fix common markdown wrapping if model ignores json_object format
                if content.startswith("```json"):
                    content = content[7:-3]
                elif content.startswith("```"):
                    content = content[3:-3]
                
                return json.loads(content)
                
            except RateLimitError as e:
                self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} hit rate limit (429). Falling back...")
                self._apply_cooldown(provider.name, 60.0)
                continue
            except Exception as e:
                reason = self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} failed: {reason}. Falling back...")
                self._apply_cooldown(provider.name, 30.0)
                continue
                
        raise RuntimeError("All LLM providers failed to generate a response.")
        
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> str:
        """Attempt to generate text, falling back through providers on RateLimit."""
        self.last_failures = []
        for provider in self._active_providers():
            try:
                client = OpenAI(
                    base_url=provider.base_url,
                    api_key=os.getenv(provider.api_key_env_var) if provider.api_key_env_var else "ollama",
                    timeout=LLM_TIMEOUT_S,
                    max_retries=0,
                )
                print(f"[llm_router] Routing request to {provider.name} ({provider.model})...")
                
                response = client.chat.completions.create(
                    model=provider.model,
                    messages=messages,
                    temperature=temperature
                )
                
                return response.choices[0].message.content
                
            except RateLimitError as e:
                self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} hit rate limit (429). Falling back...")
                self._apply_cooldown(provider.name, 60.0)
                continue
            except Exception as e:
                reason = self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} failed: {reason}. Falling back...")
                self._apply_cooldown(provider.name, 30.0)
                continue
                
        raise RuntimeError("All LLM providers failed to generate a response.")

    def generate_vision(self, messages: List[Dict[str, Any]], temperature: float = 0.0) -> str:
        """Attempt to generate a response from a vision model."""
        self.last_failures = []
        for provider in self._active_providers(require_vision=True):
            try:
                client = OpenAI(
                    base_url=provider.base_url,
                    api_key=os.getenv(provider.api_key_env_var) if provider.api_key_env_var else "ollama",
                    timeout=LLM_TIMEOUT_S,
                    max_retries=0,
                )
                print(f"[llm_router] Routing vision request to {provider.name} ({provider.vision_model})...")
                
                response = client.chat.completions.create(
                    model=provider.vision_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1000
                )
                
                return response.choices[0].message.content
                
            except RateLimitError as e:
                self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} hit rate limit (429). Falling back...")
                self._apply_cooldown(provider.name, 60.0)
                continue
            except Exception as e:
                reason = self._record_failure(provider, e)
                print(f"[llm_router] {provider.name} failed: {reason}. Falling back...")
                self._apply_cooldown(provider.name, 30.0)
                continue
                
        raise RuntimeError("All vision LLM providers failed to generate a response.")
