from dataclasses import dataclass
from typing import Optional, Dict, Any
from enum import StrEnum


@dataclass
class LogData:
    file_name: str
    total_elapsed_time_sec: float
    operation_elapsed_time_sec: (
        float  # time execution async operation in service, for local is equal total_elapsed_time_sec
    )


class LogExtraNames(StrEnum):
    """
    Extra variables for logging.
    """

    CURRENT_FILE = "current_file"
    ELAPSED_TIME_SEC = "elapsed_time_sec"
    WRITE_TO_FILE = "write_to_file"


def create_log_extra(
    *,
    current_file: Optional[str] = None,
    elapsed_time_sec: Optional[float] = None,
    write_to_file: bool = True,
) -> Dict[str, Any]:
    """Creates a dictionary with extra variables for logging."""
    return {
        LogExtraNames.CURRENT_FILE: current_file,
        LogExtraNames.ELAPSED_TIME_SEC: elapsed_time_sec,
        LogExtraNames.WRITE_TO_FILE: write_to_file,
    }
