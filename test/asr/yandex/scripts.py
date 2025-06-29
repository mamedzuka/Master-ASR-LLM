import logging
import time
import json
from uuid import uuid4
from random import randint
from pathlib import Path
from datetime import datetime

import boto3
from botocore.config import Config

from config import config
from utils.logs import create_log_extra
from asr.yandex.client import (
    Client,
    Operation,
    RecognizationOptions,
    RecognitionModelOptions,
    AudioFormatOptions,
    ContainerAudio,
    TextNormalizationOptions,
    LanguageRestrictionOptions,
)

logger = logging.getLogger(__name__)

speech_client = Client(api_key=config.speechkit_api_key)
storage_client = boto3.client(
    service_name="s3",
    region_name=config.yndx_storage_region,
    endpoint_url=config.yndx_storage_endpoint_url,
    aws_access_key_id=config.yndx_storage_access_key,
    aws_secret_access_key=config.yndx_storage_secret_access_key,
    config=Config(
        s3={
            "addressing_style": "virtual",
        },
        retries={
            "mode": "standard",
        },
    ),
)


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """Транскрибирует mp3 аудиофайл и сохраняет результат в json файл c именем из file_path."""
    extra = create_log_extra(current_file=file_path)

    object_name = _upload_file(file_path)
    logger.info("file was uploaded to cloud storage with name '%s'", object_name, extra=extra)

    audio_uri = _create_presigned_url(object_name)
    options = RecognizationOptions(
        recognitionModel=RecognitionModelOptions(
            model="general",
            audioFormat=AudioFormatOptions(
                containerAudio=ContainerAudio(containerAudioType="MP3"),
            ),
            textNormalization=TextNormalizationOptions(
                textNormalization="TEXT_NORMALIZATION_ENABLED",
                profanityFilter=False,
                literatureText=False,
                phoneFormattingMode="PHONE_FORMATTING_MODE_DISABLED",
            ),
            languageRestriction=LanguageRestrictionOptions(
                restrictionType="WHITELIST",
                languageCode=["ru-RU"],
            ),
            audioProcessingType="FULL_DATA",
        ),
    )

    operation = speech_client.create_operation_recognition(
        audio_uri=audio_uri,
        recognization_options=options,
    )
    logger.info("recognition operation was created with id '%s'", operation.id, extra=extra)

    operation = _wait_for_operation_finish(operation.id)
    elapsed_time = (
        datetime.fromisoformat(operation.modifiedAt) - datetime.fromisoformat(operation.createdAt)
    ).total_seconds()
    logger.info(
        "recognition opeation with id '%s' was finished",
        operation.id,
        extra=create_log_extra(current_file=file_path, elapsed_time_sec=elapsed_time),
    )

    result = speech_client.download_recognition_result(operation_id=operation.id)
    output_path = Path(output_dir) / f"{Path(file_path).stem}.json"
    with output_path.open("w") as f:
        f.write(result)

    return str(output_path)


def _upload_file(file_path: str) -> str:
    """Загружает файл в Yandex Object Storage и возвращает ссылку на него."""
    object_name = "_".join([str(uuid4()).replace("-", "_"), Path(file_path).name])
    storage_client.upload_file(
        Filename=file_path,
        Bucket=config.yndx_storage_bucket_name,
        Key=object_name,
    )
    return object_name


def _create_presigned_url(object_name: str) -> str:
    """Возвращает предзагруженную ссылку на объект в Yandex Object Storage."""
    return storage_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": config.yndx_storage_bucket_name,
            "Key": object_name,
        },
        ExpiresIn=3600,
    )


def _wait_for_operation_finish(operation_id: str) -> Operation:
    extra = create_log_extra(write_to_file=False)
    while True:
        operation = speech_client.get_operation_recognition(operation_id=operation_id)
        if operation.done or operation.error is not None:
            if operation.error is not None:
                raise Exception(f"operation with id {operation_id} failed with error: {str(operation.error)}")

            return operation

        time_to_sleep = 10 + randint(0, 5)
        logger.info("operation with id %s is still in progress, waiting %ds", operation_id, time_to_sleep, extra=extra)
        time.sleep(time_to_sleep)


def extract_text(file_path: str) -> str:
    """Извлекает текст из json файла с результатом распознавания."""
    with open(file_path, "r", encoding="utf-8") as f:
        result = []
        for line in f:
            json_object = json.loads(line)

            try:
                result.append(json_object["result"]["finalRefinement"]["normalizedText"]["alternatives"][0]["text"])
            except (KeyError, IndexError):
                pass

        return " ".join(result)
