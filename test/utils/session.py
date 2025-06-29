import logging

from urllib3.util import Retry
from requests import Session
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)


retries = Retry(
    total=3,
    backoff_factor=1,
    backoff_jitter=1,
    status_forcelist=(500, 502, 504),
    raise_on_status=False,
)


class RetrySession(Session):
    """
    A custom session that retries requests on failure.
    """

    tracked_headers = ["X-Request-ID", "x-request-id", "x-client-request-id"]

    def __init__(self):
        super().__init__()
        self.mount("http://", HTTPAdapter(max_retries=retries))
        self.mount("https://", HTTPAdapter(max_retries=retries))

    def request(self, method, url, **kwargs):
        response = super().request(method, url, **kwargs)

        request = response.request
        log_data = {
            "request": {
                "method": request.method,
                "url": request.url,
                "headers": {
                    header: request.headers[header] for header in self.tracked_headers if header in request.headers
                },
            },
            "response": {
                "status_code": response.status_code,
                "headers": {
                    header: response.headers[header] for header in self.tracked_headers if header in response.headers
                },
            },
        }

        logger.info("HTTP details: %s", str(log_data))
        return response
