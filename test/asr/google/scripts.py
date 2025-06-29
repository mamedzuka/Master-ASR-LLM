import logging
import json
from pathlib import Path
from uuid import uuid4

from google.cloud.storage import Client as StorageClient
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import (
    RecognitionConfig,
    RecognitionFeatures,
    AutoDetectDecodingConfig,
    BatchRecognizeFileMetadata,
    BatchRecognizeRequest,
    RecognitionOutputConfig,
    InlineOutputConfig,
    BatchRecognizeResponse,
)

from config import config
from utils.logs import create_log_extra

logger = logging.getLogger(__name__)

speech_client = SpeechClient()
storage_client = StorageClient()
storage_bucket = storage_client.bucket(config.google_storage_bucket_name)


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует mp3 аудиофайл и сохраняет результат в json файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    extra = create_log_extra(current_file=file_path)

    audio_uri = _upload_file(file_path)
    logger.info("file was uploaded to cloud storage with uri '%s'", audio_uri, extra=extra)

    recognition_config = RecognitionConfig(
        model="long",
        language_codes=["ru-RU"],
        features=RecognitionFeatures(
            enable_word_time_offsets=True,
            enable_automatic_punctuation=True,
        ),
        auto_decoding_config=AutoDetectDecodingConfig(),
    )
    file_metadata = BatchRecognizeFileMetadata(uri=audio_uri)
    request = BatchRecognizeRequest(
        recognizer=f"projects/{config.google_project_id}/locations/global/recognizers/_",
        config=recognition_config,
        files=[file_metadata],
        recognition_output_config=RecognitionOutputConfig(
            inline_response_config=InlineOutputConfig(),
        ),
    )

    operation = speech_client.batch_recognize(request)
    logger.info("recognition operation was created with id '%s'", operation.operation.name, extra=extra)  # type: ignore

    response = operation.result(None)
    elapsed_time = (operation.metadata.update_time - operation.metadata.create_time).total_seconds()  # type: ignore
    logger.info(
        "recognition opeation with id '%s' was finished",
        operation.operation.name,  # type: ignore
        extra=create_log_extra(current_file=file_path, elapsed_time_sec=elapsed_time),
    )

    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(BatchRecognizeResponse.to_dict(response), f, ensure_ascii=False, indent=2)

    return str(output_path)


def _upload_file(file_path: str) -> str:
    """Загружает файл в Google Cloud Storage и возвращает ссылку на него."""
    object_name = "_".join([str(uuid4()).replace("-", "_"), Path(file_path).name])
    blob = storage_bucket.blob(object_name, chunk_size=1024 * 1024 * 5)  # 5 MB chunk size
    blob.upload_from_filename(file_path)
    return f"gs://{config.google_storage_bucket_name}/{blob.name}"


def extract_text(file_path: str) -> str:
    """Извлекает текст из json файла, полученного в результате транскрипции."""
    with open(file_path, "r", encoding="utf-8") as f:
        json_object = json.load(f)

        result = []
        gs_key = list(json_object["results"].keys())[0]
        parts = json_object["results"][gs_key]["transcript"]["results"]
        for part in parts:
            if len(part["alternatives"]) > 0:
                result.append(part["alternatives"][0]["transcript"])

        return "".join(result)
