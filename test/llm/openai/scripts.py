from typing import Optional

from openai import OpenAI

from llm.constants import SYSTEM_PROMPT

client = OpenAI()


def summarize_transcript(transcript: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    """Возращает json строку, представляющую результат саммаризации транскрипта"""
    model = "gpt-4.1" if model is None else model
    temperature = 0.3 if temperature is None else temperature

    response = client.responses.create(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": transcript,
            },
        ],
    )

    return response.output_text.strip("```json\n")
