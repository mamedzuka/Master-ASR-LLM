"""
Обёртки над API-клиентами трёх LLM:
- OpenAI (gpt-4o-mini)
- DeepSeek R1
- Gemini 2.5 Flash
"""
from __future__ import annotations

import os
import requests
from typing import List, Dict, Any

import openai


def _system_message(content: str) -> Dict[str, str]:
    return {"role": "system", "content": content}


def _user_message(content: str) -> Dict[str, str]:
    return {"role": "user", "content": content}


class OpenAIClient:
    def __init__(self) -> None:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.client = openai.OpenAI()

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        resp = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 1024),
        )
        return resp.choices[0].message.content.strip()


class DeepSeekClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        self.url = "https://api.deepseek.com/v1/chat/completions"

    def chat(self, messages: List[Dict[str, str]], model: str = "deepseek-r1", **kwargs) -> str:
        payload: Dict[str, Any] = {"model": model, "messages": messages}
        payload.update(kwargs)
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = requests.post(self.url, json=payload, headers=headers, timeout=60)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.base = "https://generativelanguage.googleapis.com/v1beta/models"

    def chat(self, messages: List[Dict[str, str]], model: str = "gemini-2.5-flash", **kwargs) -> str:
        url = f"{self.base}/{model}:generateContent?key={self.api_key}"
        # Gemini требует немного другого формата сообщений
        payload = {
            "contents": [
                {"role": m["role"], "parts": [{"text": m["content"]}]} for m in messages
            ]
        }
        payload.update(kwargs)
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        return (
            r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        )

