import os


class Config:
    def __init__(self):
        # sber variables
        self.salute_speech_auth_key = os.environ["SALUTE_SPEECH_AUTH_KEY"]
        self.giga_chat_auth_key = os.environ["GIGACHAT_AUTH_KEY"]

        # yandex variables
        self.speechkit_api_key = os.environ["SPEECHKIT_API_KEY"]
        self.yndx_cloud_folder_id = os.environ["YANDEX_CLOUD_FOLDER_ID"]
        self.yndx_storage_region = os.environ["YANDEX_STORAGE_REGION"]
        self.yndx_storage_endpoint_url = os.environ["YANDEX_STORAGE_ENDPOINT_URL"]
        self.yndx_storage_access_key = os.environ["YANDEX_OBJECT_STORAGE_ACCESS_KEY"]
        self.yndx_storage_secret_access_key = os.environ["YANDEX_STORAGE_SECRET_ACCESS_KEY"]
        self.yndx_storage_bucket_name = os.environ["YANDEX_STORAGE_BUCKET_NAME"]
        self.yndx_foundation_models_api_key = os.environ["YANDEX_FOUNDATION_MODELS_API_KEY"]

        # google variables
        self.google_storage_bucket_name = os.environ["GOOGLE_STORAGE_BUCKET_NAME"]
        self.google_project_id = os.environ["GOOGLE_PROJECT_ID"]
        self.google_ai_api_key = os.environ["GOOGLE_AI_API_KEY"]

        # azure variables
        self.azure_speech_service_key = os.environ["AZURE_SPEECH_SERVICE_KEY"]
        self.azure_speech_service_region = os.environ["AZURE_SPEECH_SERVICE_REGION"]
        self.azure_openai_api_key = os.environ["AZURE_OPENAI_API_KEY"]
        self.azure_openai_endpoint = os.environ["AZURE_OPENAI_ENDPOINT"]

        # deepseek variables
        self.deepseek_api_key = os.environ["DEEPSEEK_API_KEY"]

        # vosk variables
        self.vosk_model_path = os.environ["VOSK_MODEL_PATH"]


config = Config()
