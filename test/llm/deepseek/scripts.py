from typing import Optional

from openai import OpenAI

from config import config
from llm.constants import SYSTEM_PROMPT


client = OpenAI(
    api_key=config.deepseek_api_key,
    base_url="https://api.deepseek.com",
)


def summarize_transcript(
    transcript: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    model = "deepseek-chat" if model is None else model
    temperature = 1.0 if temperature is None else temperature

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": transcript,
            },
        ],
        stream=False,
    )

    return response.choices[0].message.content.strip("```json\n")  # type: ignore
