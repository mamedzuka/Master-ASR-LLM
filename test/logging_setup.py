import logging

from utils.logs import LogExtraNames


class Formatter(logging.Formatter):
    extra_names = (LogExtraNames.CURRENT_FILE, LogExtraNames.ELAPSED_TIME_SEC)

    def format(self, record: logging.LogRecord) -> str:
        extra_parts = []

        for name in self.extra_names:
            value = getattr(record, name, None)
            if value is not None:
                extra_parts.append(f"{name}: {value}")

        msg = super().format(record)
        if len(extra_parts) > 0:
            msg += f" | extra_variables - {', '.join(extra_parts)}"

        return msg


class FileWriterFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return bool(getattr(record, LogExtraNames.WRITE_TO_FILE, True))


def configure_logging(file_name: str):
    """
    Configures the logging settings for the application.
    This function sets the logging format, level, and handlers.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    formatter = Formatter(
        "%(levelname)s, thread_id: %(thread)d, logger_name: %(name)s, func_name: %(funcName)s - %(message)s"
    )

    file_handler = logging.FileHandler(file_name)
    file_handler.addFilter(FileWriterFilter())
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    # urlib3 logger
    urlib3_logger = logging.getLogger("urllib3")
    urlib3_logger.setLevel(logging.ERROR)
    urlib3_logger.propagate = False

    # google service logger
    google_logger = logging.getLogger("google")
    google_logger.addHandler(logging.StreamHandler())
    google_logger.setLevel(logging.DEBUG)

    pymorhy3_logger = logging.getLogger("pymorphy3")
    pymorhy3_logger.setLevel(logging.ERROR)
    pymorhy3_logger.propagate = False
