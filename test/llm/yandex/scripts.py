from typing import Optional

from openai import OpenAI

from config import config
from llm.constants import SYSTEM_PROMPT


client = OpenAI(
    api_key=config.yndx_foundation_models_api_key,
    base_url="https://llm.api.cloud.yandex.net/v1",
)


def summarize_transcript(transcript: str, model: Optional[str] = None, temperature: Optional[float] = None) -> str:
    model = "yandexgpt" if model is None else model
    model_path = f"gpt://{config.yndx_cloud_folder_id}/{model}/latest"
    temperature = 0.3 if temperature is None else temperature

    response = client.chat.completions.create(
        model=model_path,
        temperature=temperature,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )

    return response.choices[0].message.content.strip("```\n")  # type: ignore
