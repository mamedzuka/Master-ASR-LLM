import logging
import json
import time
from random import randint
from pathlib import Path
from datetime import datetime

from config import config
from utils.logs import create_log_extra
from asr.sber.client import Client, RecognizeOptions, Task

logger = logging.getLogger(__name__)

client = Client(config.salute_speech_auth_key)


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует аудиофайл (только mp3) и сохраняет результат в json файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    extra = create_log_extra(current_file=file_path)

    file_id = client.upload_audio_file(file_path=file_path)
    logger.info("file was uploaded to cloud storage with name '%s'", file_id, extra=extra)

    options = RecognizeOptions(
        model="general",
        audio_encoding="MP3",
        language="ru-RU",
    )
    task = client.create_task_recognition(request_file_id=file_id, options=options)

    task = _wait_for_task_finish(task.id)
    elapsed_time_sec = (
        datetime.fromisoformat(task.updated_at) - datetime.fromisoformat(task.created_at)
    ).total_seconds()
    logger.info(
        "recognition task with id: '%s' was finished, response_file_id: '%s'",
        task.id,
        task.response_file_id,
        extra=create_log_extra(current_file=file_path, elapsed_time_sec=elapsed_time_sec),
    )

    result = client.download_recognition_result(response_file_id=task.response_file_id)  # type: ignore
    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
    with output_path.open("w") as f:
        f.write(result)

    return str(output_path)


def _wait_for_task_finish(task_id: str) -> Task:
    extra = create_log_extra(write_to_file=False)
    while True:
        task = client.get_task_recognition(task_id=task_id)
        if task.status in ["DONE", "ERROR"]:
            if task.status == "ERROR":
                raise Exception(f"task {task_id} failed with error")

            return task

        time_to_sleep = 15 + randint(0, 5)
        logger.info("task with id %s is still in progress, waiting %ds", task_id, time_to_sleep, extra=extra)
        time.sleep(time_to_sleep)


def extract_text(file_path: str) -> str:
    """Извлекает текст из json файла с результатом распознавания."""
    with open(file_path, "r") as f:
        json_object = json.load(f)

        result = []
        for part in json_object:
            result.append(part["results"][0]["normalized_text"])

        return "".join(result)
