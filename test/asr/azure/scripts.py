import logging
import time
import json
from pathlib import Path

import azure.cognitiveservices.speech as speech_sdk

from config import config
from utils.logs import create_log_extra

logger = logging.getLogger(__name__)


speech_config = speech_sdk.SpeechConfig(
    subscription=config.azure_speech_service_key,
    region=config.azure_speech_service_region,
    speech_recognition_language="ru-RU",
)
speech_config.output_format = speech_sdk.OutputFormat.Detailed


class BinaryFileReaderCallback(speech_sdk.audio.PullAudioInputStreamCallback):
    def __init__(self, filename: str):
        super().__init__()
        self._file_h = open(filename, "rb")

    def read(self, buffer: memoryview) -> int:
        size = buffer.nbytes
        frames = self._file_h.read(size)

        buffer[: len(frames)] = frames

        return len(frames)

    def close(self) -> None:
        self._file_h.close()


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует mp3 аудиофайл и сохраняет результат в txt файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    compressed_format = speech_sdk.audio.AudioStreamFormat(
        compressed_stream_format=speech_sdk.AudioStreamContainerFormat.MP3
    )

    callback = BinaryFileReaderCallback(file_path)
    stream = speech_sdk.audio.PullAudioInputStream(stream_format=compressed_format, pull_stream_callback=callback)
    audio_config = speech_sdk.audio.AudioConfig(stream=stream)

    speech_recognizer = speech_sdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

    done = False
    elapsed_time_sec = 0
    result = []

    def stop_cb(evt):
        """callback that signals to stop continuous recognition upon receiving an event"""
        nonlocal done
        done = True

    def recognized_cb(evt: speech_sdk.SpeechRecognitionEventArgs):
        """callback that is called when a recognition result is received"""
        if evt.result.reason == speech_sdk.ResultReason.RecognizedSpeech:
            nonlocal elapsed_time_sec
            elapsed_time_sec += evt.result.duration
            result.append(evt.result.json)

    # Connect callbacks to the events fired by the speech recognizer
    speech_recognizer.recognized.connect(recognized_cb)

    speech_recognizer.session_started.connect(
        lambda evt: logger.info("SESSION STARTED {}".format(evt), extra=create_log_extra(write_to_file=False))
    )
    speech_recognizer.session_stopped.connect(
        lambda evt: logger.info("SESSION STOPPED {}".format(evt), extra=create_log_extra(write_to_file=False))
    )
    speech_recognizer.canceled.connect(
        lambda evt: logger.info("CANCELED {}".format(evt), extra=create_log_extra(write_to_file=False))
    )

    # stop continuous recognition on either session stopped or canceled events
    speech_recognizer.session_stopped.connect(stop_cb)
    speech_recognizer.canceled.connect(stop_cb)

    # start continuous speech recognition
    speech_recognizer.start_continuous_recognition()
    while not done:
        time.sleep(0.5)
    speech_recognizer.stop_continuous_recognition()

    logger.info(
        "transcription completed was finished",
        extra=create_log_extra(current_file=file_path, elapsed_time_sec=elapsed_time_sec * 1e-7),
    )

    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
    with open(output_path, "w") as f:
        f.write("\n".join(result))

    return str(output_path)


def extract_text(file_path: str) -> str:
    """Извлекает текст из json файла, полученного в результате транскрипции."""
    with open(file_path, "r", encoding="utf-8") as f:
        result = []
        for line in f:
            json_object = json.loads(line)
            result.append(json_object["DisplayText"])

        return " ".join(result)
