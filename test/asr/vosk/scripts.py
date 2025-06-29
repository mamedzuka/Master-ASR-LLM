import subprocess
import json
from functools import lru_cache
from pathlib import Path

from vosk import Model, KaldiRecognizer, SetLogLevel

from config import config

SAMPLE_RATE = 16000
SetLogLevel(0)


@lru_cache(maxsize=1)
def get_model() -> Model:
    return Model(config.vosk_model_path)


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует аудиофайл и сохраняет результат в txt файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    recognizer = KaldiRecognizer(get_model(), SAMPLE_RATE)

    _recognize_audio(recognizer, file_path)
    jres = json.loads(recognizer.FinalResult())
    processed_result = jres["text"]

    output_path = Path(output_dir) / f"{Path(file_path).stem}.txt"
    with open(output_path, "w") as f:
        f.write(processed_result)

    return str(output_path)


def _recognize_audio(recognizer: KaldiRecognizer, audio_file_path: str):
    with subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "quiet",
            "-i",
            audio_file_path,
            "-ar",
            str(SAMPLE_RATE),
            "-ac",
            "1",
            "-f",
            "s16le",
            "-",
        ],
        stdout=subprocess.PIPE,
    ) as process:
        while True:
            data = process.stdout.read(8000)  # type: ignore
            if len(data) == 0:
                break
            recognizer.AcceptWaveform(data)
