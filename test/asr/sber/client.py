import time
import threading
import mimetypes
from typing import Dict, Literal, List, Optional
from dataclasses import dataclass, asdict
from uuid import uuid4

from requests import Request
from requests.auth import AuthBase

from utils.session import RetrySession

TASK_STATUSES = Literal["NEW", "RUNNING", "CANCELED", "DONE", "ERROR"]


@dataclass
class Task:
    id: str
    created_at: str
    updated_at: str
    status: TASK_STATUSES
    response_file_id: Optional[str] = None  # когда статус DONE id отличен от None


@dataclass
class Hints:
    words: List[str]
    enable_letters: bool = False
    eou_timeout: float = 1  # Диапазон: 0.5-5


@dataclass
class SpeakerSeparationOptions:
    count: int  # Диапазон: 1-10
    enable: bool = False
    enable_only_main_speaker: bool = False


@dataclass
class RecognizeOptions:
    """https://developers.sber.ru/docs/ru/salutespeech/rest/post-async-speech-recognition."""

    model: Literal["general", "callcenter"] = "general"
    audio_encoding: Literal["PCM_S16LE", "OPUS", "MP3", "FLAC", "ALAW", "MULAW"] = "MP3"
    sample_rate: Optional[int] = None  # https://developers.sber.ru/docs/ru/salutespeech/guides/recognition/encodings
    language: Literal["ru-RU", "en-US", "kk-KZ"] = "ru-RU"
    enable_profanity_filter: bool = False
    hypotheses_count: int = 1  # Диапазон: 0–10
    # no_speech_timeout: int = 7  # Диапазон: 2–20 секунд
    # max_speech_timeout: float = 20  # Диапазон: 0.5–20.0 секунд
    hints: Optional[Hints] = None
    channels_count: int = (
        1  # Диапазон: 1–10, https://developers.sber.ru/docs/ru/salutespeech/guides/recognition/encodings
    )
    speaker_separation_options: Optional[SpeakerSeparationOptions] = None
    insight_models: Optional[List[Literal["csi", "call_features", "is_solved"]]] = None


class TokenAuth(AuthBase):
    def __init__(self, auth_key: str):
        self._auth_key = auth_key
        self._lock = threading.Lock()
        self._access_token = ""
        self._expires_at = 0

    def _token_expired(self) -> bool:
        return time.time() >= self._expires_at - 300

    def _refresh_access_token(self):
        # только для физ.лиц
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": str(uuid4()),
            "Authorization": f"Basic {self._auth_key}",
        }

        body = {
            "scope": "SALUTE_SPEECH_PERS",
        }

        with RetrySession() as session:
            result = session.post(
                url="https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
                headers=headers,
                data=body,
            )
            result.raise_for_status()

            jresponse = result.json()
            self._access_token = jresponse["access_token"]
            self._expires_at = int(jresponse["expires_at"])

    def __call__(self, request: Request) -> Request:
        with self._lock:
            if self._token_expired():
                self._refresh_access_token()

            request.headers["Authorization"] = f"Bearer {self._access_token}"
            return request


class Client:
    """SberClient потокобезопасный клиент для salutespeech от сбера."""

    def __init__(
        self,
        auth_key: str,
    ):
        self.auth = TokenAuth(auth_key)

    def _get_default_headers(self, with_request_id: bool = False) -> Dict[str, str]:
        headers = {
            "Accept": "application/json",
        }

        if with_request_id:
            headers["X-Request-ID"] = str(uuid4())

        return headers

    def upload_audio_file(
        self,
        *,
        file_path: str,
    ) -> str:
        """Возвращает id загруженного файла"""
        headers = self._get_default_headers(with_request_id=True)
        headers["Content-Type"] = get_mimetype(file_path)

        with RetrySession() as session:
            with open(file_path, "rb") as f:
                result = session.post(
                    "https://smartspeech.sber.ru/rest/v1/data:upload",
                    headers=headers,
                    data=f,
                    auth=self.auth,
                )
                result.raise_for_status()
                return result.json()["result"]["request_file_id"]

    def create_task_recognition(
        self,
        *,
        request_file_id: str,
        options: RecognizeOptions,
    ) -> Task:
        """Возвращает id созданной задачи."""
        headers = self._get_default_headers(with_request_id=True)

        body = {
            "request_file_id": request_file_id,
            "options": asdict(options),
        }

        with RetrySession() as session:
            result = session.post(
                "https://smartspeech.sber.ru/rest/v1/speech:async_recognize",
                headers=headers,
                json=body,
                auth=self.auth,
            )
            result.raise_for_status()

            jresponse = result.json()["result"]
            return Task(**jresponse)

    def get_task_recognition(self, *, task_id: str) -> Task:
        headers = self._get_default_headers()

        params = {
            "id": task_id,
        }

        with RetrySession() as session:
            response = session.get(
                "https://smartspeech.sber.ru/rest/v1/task:get",
                headers=headers,
                params=params,
                auth=self.auth,
            )
            response.raise_for_status()

            jresponse = response.json()["result"]
            return Task(**jresponse)

    def download_recognition_result(self, *, response_file_id: str) -> str:
        """Возвращает json строку с результатом распознавания."""
        headers = self._get_default_headers()

        params = {
            "response_file_id": response_file_id,
        }

        with RetrySession() as session:
            response = session.get(
                "https://smartspeech.sber.ru/rest/v1/data:download",
                headers=headers,
                params=params,
                auth=self.auth,
            )
            response.raise_for_status()

            return response.text


def get_mimetype(file_path: str) -> str:
    # todo: чуть сложнее логика, возможно испльзовать ffmeg (для mp3 и flac пойдет)
    mimetype = mimetypes.guess_type(file_path)
    if mimetype[0] is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")
    return mimetype[0]
