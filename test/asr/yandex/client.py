from typing import Dict, List, Literal, Any, Optional
from dataclasses import dataclass, asdict
from uuid import uuid4

from utils.session import RetrySession


@dataclass
class Operation:
    """
    params:
        id - ID of the operation.
        description - Description of the operation. 0-256 characters long.
        createdAt - Creation date of the operation in RFC3339 format.
        createdBy - ID of the user who created the operation.
        modifiedAt - Last modification date of the operation in RFC3339 format.
        done - if false - the operation is still in progress, if true - the operation is completed or error is not none.
        metadata - Service-specific metadata associated with the operation.
        error - The error result of the operation in case of failure or cancellation.
    """

    id: str
    description: str
    createdAt: str
    createdBy: str
    modifiedAt: str
    done: bool
    metadata: Optional[Dict[str, Any]] = None
    error: Optional["ErrorStatus"] = None
    response: Optional[Dict[str, Any]] = None


@dataclass
class ErrorStatus:
    """
    params:
        code - Error code (https://github.com/googleapis/googleapis/blob/master/google/rpc/code.proto)
        message - An error message
        details - A list of messages that carry the error details.
    """

    code: int
    message: str
    details: List[Dict[str, Any]]


@dataclass
class RecognizationOptions:
    recognitionModel: "RecognitionModelOptions"
    speechAnalysis: Optional["SpeechAnalysisOptions"] = None
    speakerLabeling: Optional["SpeakerLabelingOptions"] = None
    recognitionClassifier: Optional["RecognitionClassifierOptions"] = None


@dataclass
class RecognitionModelOptions:
    """
    params:
        model - Sets the recognition model for the cloud version of SpeechKit.
        audioFormat - Specified input audio.
        textNormalization - Text normalization options.
        languageRestriction - Possible languages in audio.
        audioProcessingType - How to deal with audio data (in real time, after all data is received, etc). Default is REAL_TIME.
            REAL_TIME: Process audio in mode optimized for real-time recognition, i.e. send partials and final responses as soon as possible.
            FULL_DATA: Process audio after all data was received.
    """

    audioFormat: "AudioFormatOptions"
    textNormalization: "TextNormalizationOptions"
    languageRestriction: "LanguageRestrictionOptions"
    model: Literal["general", "general:rc", "general:deprecated"] = "general"
    audioProcessingType: Literal["AUDIO_PROCESSING_TYPE_UNSPECIFIED", "REAL_TIME", "FULL_DATA"] = "REAL_TIME"


@dataclass
class AudioFormatOptions:
    """
    params:
        rawAudio - Audio without container. Includes only one of the fields rawAudio, containerAudio.
        containerAudio - Audio is wrapped in container. Includes only one of the fields rawAudio, containerAudio.
    """

    rawAudio: Optional["RawAudio"] = None
    containerAudio: Optional["ContainerAudio"] = None


@dataclass
class RawAudio:
    """
    params:
        audioEncoding - Type of audio encoding
        sampleRateHertz - PCM sample rate
        audioChannelCount - PCM channel count. Currently only single channel audio is supported in real-time recognition.
    """

    audioEncoding: Literal["AUDIO_ENCODING_UNSPECIFIED", "LINEAR16_PCM"]
    sampleRateHertz: str
    audioChannelCount: str


@dataclass
class ContainerAudio:
    """
    params:
        containerAudioType - Type of audio container.
    """

    containerAudioType: Literal["CONTAINER_AUDIO_TYPE_UNSPECIFIED", "WAV", "OGG_OPUS", "MP3"]


@dataclass
class TextNormalizationOptions:
    """
    params:
        textNormalization - enum.
        profanityFilter - Profanity filter (default: false).
        literatureText - Rewrite text in literature style (default: false).
        phoneFormattingMode - Define phone formatting mode.
    """

    textNormalization: Literal[
        "TEXT_NORMALIZATION_UNSPECIFIED", "TEXT_NORMALIZATION_ENABLED", "TEXT_NORMALIZATION_DISABLED"
    ]
    profanityFilter: bool = False
    literatureText: bool = False
    phoneFormattingMode: Literal["PHONE_FORMATTING_MODE_UNSPECIFIED", "PHONE_FORMATTING_MODE_DISABLED"] = (
        "PHONE_FORMATTING_MODE_DISABLED"
    )


@dataclass
class LanguageRestrictionOptions:
    """
    params:
        restrictionType - Language restriction type.
            WHITELIST: The allowing list. The incoming audio can contain only the listed languages.
            BLACKLIST: The forbidding list. The incoming audio cannot contain the listed languages.
        languageCode - The list of language codes to restrict recognition in the case of an auto model
            https://yandex.cloud/en/docs/speechkit/stt/models
    """

    restrictionType: Literal["LANGUAGE_RESTRICTION_TYPE_UNSPECIFIED", "WHITELIST", "BLACKLIST"]
    languageCode: List[str]


@dataclass
class RecognitionClassifierOptions:
    classifiers: List["RecognitionClassifier"]


@dataclass
class RecognitionClassifier:
    """
    params:
        classifier -  Classifier name
        triggers - Describes the types of responses to which the classification results will come
    """

    classifier: str
    triggers: List[Literal["TRIGGER_TYPE_UNSPECIFIED", "ON_UTTERANCE", "ON_FINAL", "ON_PARTIAL"]]


@dataclass
class SpeechAnalysisOptions:
    """
    params:
        enableSpeakerAnalysis - Analyse speech for every speaker
        enableConversationAnalysis - Analyse conversation of two speakers
        descriptiveStatisticsQuantiles - Quantile levels in range (0, 1) for descriptive statistics
    """

    enableSpeakerAnalysis: bool
    enableConversationAnalysis: bool
    descriptiveStatisticsQuantiles: List[str]


@dataclass
class SpeakerLabelingOptions:
    """
    params:
        speakerLabeling - Specifies the execution of speaker labeling. Default is SPEAKER_LABELING_DISABLED.
    """

    speakerLabeling: Literal[
        "SPEAKER_LABELING_UNSPECIFIED", "SPEAKER_LABELING_ENABLED", "SPEAKER_LABELING_DISABLED"
    ] = "SPEAKER_LABELING_DISABLED"


class Client:
    """YandexClient потокобезопасный клиент для speechkit от яндекса"""

    def __init__(
        self,
        *,
        api_key: str,
    ):
        self.api_key = api_key

    def _get_default_headers(self, with_request_id: bool = False) -> Dict[str, str]:
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Accept": "application/json",
        }

        if with_request_id:
            headers["x-request-id"] = str(uuid4())

        return headers

    def create_operation_recognition(
        self,
        *,
        audio_uri: str,
        recognization_options: RecognizationOptions,
    ) -> Operation:
        headers = self._get_default_headers(with_request_id=True)

        body = asdict(recognization_options)
        body["uri"] = audio_uri

        with RetrySession() as session:
            response = session.post(
                "https://stt.api.cloud.yandex.net/stt/v3/recognizeFileAsync",
                headers=headers,
                json=body,
            )
            response.raise_for_status()

            jresponse = response.json()
            return Operation(**jresponse)

    def get_operation_recognition(self, *, operation_id: str) -> Operation:
        headers = self._get_default_headers()

        with RetrySession() as session:
            response = session.get(
                f"https://operation.api.cloud.yandex.net/operations/{operation_id}",
                headers=headers,
            )
            response.raise_for_status()

            jresponse = response.json()
            return Operation(**jresponse)

    def download_recognition_result(
        self,
        *,
        operation_id: str,
    ) -> str:
        """Возвращает json строку с результатами распознавания."""
        headers = self._get_default_headers()

        params = {
            "operationId": operation_id,
        }

        with RetrySession() as s:
            response = s.get(
                "https://stt.api.cloud.yandex.net/stt/v3/getRecognition",
                headers=headers,
                params=params,
            )
            response.raise_for_status()

            return response.text
