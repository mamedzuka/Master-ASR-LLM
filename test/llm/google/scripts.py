from typing import Optional

from google import genai
from google.genai import types

from config import config
from llm.constants import SYSTEM_PROMPT

client = genai.Client(
    api_key=config.google_ai_api_key,
)


def summarize_transcript(transcript: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    """Возращает json строку, представляющую результат саммаризации транскрипта"""
    model = "gemini-2.5-pro" if model is None else model
    temperature = 0.3 if temperature is None else temperature

    response = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=temperature,
        ),
        contents=transcript,
    )
    return response.text.strip("```json\n")  # type: ignore
