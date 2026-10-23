import os
import json
import httpx
from typing import List, Dict, Any, Optional, Tuple
from src.common.config import settings

class LLMProvider:
    """
    Unified LLM provider supporting Groq, OpenAI, and local offline execution.
    Automatically detects and prioritizes GROQ_API_KEY for ultra-low latency tool calling.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        # 1. Detect Groq API Key
        self.groq_api_key = api_key or os.environ.get("GROQ_API_KEY") or settings.GROQ_API_KEY
        self.openai_api_key = os.environ.get("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        
        if self.groq_api_key:
            self.provider_type = "groq"
            self.api_key = self.groq_api_key
            self.base_url = base_url or settings.GROQ_BASE_URL
            self.model = model or settings.GROQ_MODEL
        elif self.openai_api_key:
            self.provider_type = "openai"
            self.api_key = self.openai_api_key
            self.base_url = base_url or "https://api.openai.com/v1"
            self.model = model or settings.LLM_MODEL
        else:
            self.provider_type = "local"
            self.api_key = None
            self.base_url = None
            self.model = "local-deterministic"

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> Dict[str, Any]:
        """
        Invokes LLM with optional tool specs. Returns raw response containing content or tool_calls.
        """
        if self.provider_type in ["groq", "openai"]:
            url = f"{self.base_url}/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            async with httpx.AsyncClient(timeout=30.0) as client:
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        message = data["choices"][0]["message"]
                        return {
                            "status": "success",
                            "provider": self.provider_type,
                            "content": message.get("content") or message.get("reasoning"),
                            "reasoning": message.get("reasoning"),
                            "tool_calls": message.get("tool_calls", []),
                            "usage": data.get("usage", {})
                        }
                    else:
                        return {
                            "status": "error",
                            "error": f"API Error ({resp.status_code}): {resp.text}"
                        }
                except Exception as e:
                    return {
                        "status": "error",
                        "error": f"Network exception: {str(e)}"
                    }

        # Local Offline Fallback
        return {
            "status": "local",
            "provider": "local",
            "content": None,
            "tool_calls": []
        }
