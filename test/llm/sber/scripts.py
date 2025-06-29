from typing import Optional

from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole

from config import config
from llm.constants import SYSTEM_PROMPT


def summarize_transcript(
    transcript: str,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """Возращает json строку, представляющую результат саммаризации транскрипта"""
    model = "GigaChat-2" if model is None else model
    temperature = 0.3 if temperature is None else temperature

    with GigaChat(credentials=config.giga_chat_auth_key, verify_ssl_certs=False) as giga:
        payload = Chat(
            model=model,
            temperature=temperature,
            messages=[
                Messages(
                    role=MessagesRole.SYSTEM,
                    content=SYSTEM_PROMPT,
                ),
                Messages(
                    role=MessagesRole.USER,
                    content=transcript,
                ),
            ],
        )

        response = giga.chat(payload)
        return response.choices[0].message.content
