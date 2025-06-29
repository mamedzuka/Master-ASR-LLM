from pathlib import Path
import argparse

from dotenv import load_dotenv  
from speech import transcribe_mp3, correct_text
from llm_clients import (
    OpenAIClient, DeepSeekClient, GeminiClient,
    _system_message, _user_message,
)

PROMPT_PATH = Path(__file__).parent.parent / "prompt.conf"

def load_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def summarize_ru(text: str, sys_prompt: str) -> str:
    msgs = [_system_message(sys_prompt), _user_message(text)]
    gpt = OpenAIClient().chat(msgs)                 
    deep = DeepSeekClient().chat(msgs)        
    gem = GeminiClient().chat(msgs)                
    fusion = "\n\n".join([gpt, deep, gem])
    final = DeepSeekClient().chat(
        [
            _system_message(sys_prompt + "\n\nСоберите единый реферат из трёх версий."),
            _user_message(fusion),
        ],
        max_tokens=1024,
    )
    return final

def cli() -> None:
    parser = argparse.ArgumentParser(
        prog="autoref",
        description="Автоматическое реферирование русских аудиоконференций",
    )
    parser.add_argument("audio", type=Path, help="MP3-файл совещания")
    parser.add_argument("-o", "--output", type=Path, help="Файл для сохранения реферата")
    parser.add_argument("--no-correct", action="store_true", help="Пропустить орфокоррекцию")
    args = parser.parse_args()

    load_dotenv()  # переменные окружения
    text = transcribe_mp3(args.audio)["text"]
    if not args.no_correct:
        text = correct_text(text)
    summary = summarize_ru(text, load_prompt())

    if args.output:
        args.output.write_text(summary, encoding="utf-8")
        print(f"Реферат сохранён: {args.output}")
    else:
        print("\n=== Итоговый реферат ===\n")
        print(summary)


if __name__ == "__main__":
    cli()

