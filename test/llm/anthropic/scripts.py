from typing import Optional

import anthropic

from llm.constants import SYSTEM_PROMPT

client = anthropic.Anthropic()


def summarize_transcript(
    transcript: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    model = "claude-sonnet-4-0" if model is None else model
    temperature = 0.3 if temperature is None else temperature

    message = client.messages.create(
        model=model,
        temperature=temperature,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": transcript},
        ],
    )

    return message.content[0].text.replace("\n", "").strip("```json")  # type: ignore
