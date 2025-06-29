import re
import csv
import logging
import time
import concurrent.futures
from pathlib import Path
from typing import Literal, Dict, Callable, List

import jiwer
import jiwer.transforms as tr

from utils.fs import find_files
from utils.logs import create_log_extra, LogData
from asr.google import scripts as google_scripts
from asr.azure import scripts as azure_scripts
from asr.yandex import scripts as yandex_scripts
from asr.sber import scripts as sber_scripts
from asr.whisper import scripts as whisper_scripts
from asr.vosk import scripts as vosk_scripts
from asr.wav2vec2 import scripts as wav2vec2_scripts

logger = logging.getLogger(__name__)

exist_transcriber_systems = Literal["cloud_s2t", "azure", "speechkit", "salute", "whisper", "vosk", "wav2vec2"]

transcribers: Dict[exist_transcriber_systems, tuple] = {
    "cloud_s2t": (google_scripts.transcribe_audio, True),
    "azure": (azure_scripts.transcribe_audio, True),
    "speechkit": (yandex_scripts.transcribe_audio, True),
    "salute": (sber_scripts.transcribe_audio, True),
    "whisper": (whisper_scripts.transcribe_audio, False),
    "vosk": (vosk_scripts.transcribe_audio, False),
    "wav2vec2": (wav2vec2_scripts.transcribe_audio, False),
}

exist_parser_systems = Literal["cloud_s2t", "azure", "speechkit", "salute"]

parsers: Dict[exist_parser_systems, Callable[[str], str]] = {
    "cloud_s2t": google_scripts.extract_text,
    "azure": azure_scripts.extract_text,
    "speechkit": yandex_scripts.extract_text,
    "salute": sber_scripts.extract_text,
}

wer_transform = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.ReduceToListOfListOfWords(),
    ]
)

cer_transform = tr.Compose(
    [
        tr.ToLowerCase(),
        tr.RemoveWhiteSpace(replace_by_space=True),
        tr.RemoveMultipleSpaces(),
        tr.RemovePunctuation(),
        tr.Strip(),
        tr.ReduceToListOfListOfChars(),
    ]
)


class ASRProcessor:
    @staticmethod
    def transcribe(dataset_dir: str, output_dir: str, system: exist_transcriber_systems, max_threads: int = 1):
        transcriber, supports_multithreading = transcribers[system]
        transcriber = _transcriber_wrapper(transcriber)

        mp3_files = list(find_files(dataset_dir, ".mp3"))
        count_proccessed_file = 0

        output_dir_path = Path(output_dir) / system
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info("found %d mp3 files in dataset directory: '%s', start process", len(mp3_files), dataset_dir)

        if supports_multithreading and max_threads > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
                futures = {
                    executor.submit(transcriber, str(file_path), str(output_dir_path)): file_path
                    for file_path in mp3_files
                }
                for future in concurrent.futures.as_completed(futures):
                    count_proccessed_file += 1 if future.result() is not None else 0
        else:
            for file_path in mp3_files:
                result = transcriber(str(file_path), str(output_dir_path))
                count_proccessed_file += 1 if result is not None else 0

        logger.info(
            "transcription completed for %d files out of %d",
            count_proccessed_file,
            len(mp3_files),
        )

    @staticmethod
    def parse(input_dir: str, system: exist_parser_systems):
        parser = parsers[system]
        json_files = list(find_files(input_dir, ".json"))

        for file_path in json_files:
            result = parser(str(file_path))

            output_path = file_path.with_name(file_path.stem + ".txt")
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result)

    @staticmethod
    def metrics(reference_dir: str, hypotheses_dir: str, log_path: str):
        extra = create_log_extra(write_to_file=False)

        hypotheses_files = find_files(hypotheses_dir, ".txt")
        name_to_reference_file = {file_path.name: file_path for file_path in find_files(reference_dir, ".txt")}

        logs_data = _parse_log_file(log_path)
        file_name_to_log_data = {log_data.file_name: log_data for log_data in logs_data}

        result = []
        for hypotheses_file in hypotheses_files:
            file_name = hypotheses_file.name
            if file_name not in name_to_reference_file:
                logger.warning("no reference file found for hypothesis file: '%s'", file_name, extra=extra)
                continue

            reference_file = name_to_reference_file[file_name]

            with open(reference_file) as rf, open(hypotheses_file) as hf:
                rf_text = rf.read()
                hf_text = hf.read()

                wer = jiwer.wer(rf_text, hf_text, wer_transform, wer_transform)
                mer = jiwer.mer(rf_text, hf_text, wer_transform, wer_transform)
                wil = jiwer.wil(rf_text, hf_text, wer_transform, wer_transform)
                wip = jiwer.wip(rf_text, hf_text, wer_transform, wer_transform)
                cer = jiwer.cer(rf_text, hf_text, cer_transform, cer_transform)

                log_data = file_name_to_log_data.get(hypotheses_file.stem, None)
                total_sec = log_data.total_elapsed_time_sec if log_data else 0
                operation_sec = log_data.operation_elapsed_time_sec if log_data else 0

                result.append(
                    (
                        file_name,
                        f"{wer:.3f}".replace(".", ","),
                        f"{mer:.3f}".replace(".", ","),
                        f"{wil:.3f}".replace(".", ","),
                        f"{wip:.3f}".replace(".", ","),
                        f"{cer:.3f}".replace(".", ","),
                        f"{total_sec:.2f}".replace(".", ","),
                        f"{operation_sec:.2f}".replace(".", ","),
                    )
                )

            response_path = Path(hypotheses_dir) / f"metrics.csv"
            with open(response_path, "w", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
                writer.writerow(
                    [
                        "file_name",
                        "wer",
                        "mer",
                        "wil",
                        "wip",
                        "cer",
                        "total_elapsed_time_sec",
                        "operation_elapsed_time_sec",
                    ]
                )
                writer.writerows(result)


def _transcriber_wrapper(transcriber):
    def transcriber_wrapper(file_path: str, output_dir: str):
        try:
            logger.info("starting transcription", extra=create_log_extra(current_file=file_path))
            start_time = time.time()
            output_path = transcriber(file_path, output_dir)
            elapsed_time = time.time() - start_time
            logger.info(
                "transcription completed, output saved to: '%s'",
                output_path,
                extra=create_log_extra(current_file=file_path, elapsed_time_sec=elapsed_time),
            )
            return output_path
        except Exception:
            logger.exception("error while trancribing", extra=create_log_extra(current_file=file_path))
            return None

    return transcriber_wrapper


def _parse_log_file(log_path: str) -> List[LogData]:
    re_header = re.compile(
        r"(?P<level>[A-Z]+), thread_id: (?P<thread_id>\d+), logger_name: (?P<logger_name>[\w\.]+), func_name: (?P<func_name>\w+) - (?P<message>.+)"
    )

    re_complete = re.compile(
        r"transcription completed, output saved to: .+ \| extra_variables - current_file: (?P<file_path>.+), elapsed_time_sec: (?P<elapsed>[\d.]+)"
    )

    re_operation_complete = re.compile(
        r"extra_variables - current_file: (?P<file_path>.+), elapsed_time_sec: (?P<elapsed>[\d.]+)"
    )

    events = {}
    with open(log_path, "r") as f:
        for line in f:
            header_match = re_header.match(line)
            if not header_match:
                continue

            m_complete = re_complete.search(line)
            if m_complete:
                file_path = m_complete.group("file_path")
                elapsed_time = float(m_complete.group("elapsed"))
                events.setdefault(file_path, {})["total_elapsed_time"] = elapsed_time
                continue

            header = header_match.groupdict()
            func_name = header["func_name"]
            message = header["message"]

            if func_name == "transcribe_audio":
                m_op_complete = re_operation_complete.search(message)
                if m_op_complete:
                    file_path = m_op_complete.group("file_path")
                    elapsed_time = float(m_op_complete.group("elapsed"))
                    events.setdefault(file_path, {})["operation_elapsed_time"] = elapsed_time
                    continue

    result = []
    for file_path in events:
        total_elapsed_time = events[file_path].get("total_elapsed_time", None)
        if total_elapsed_time is None:
            continue

        operation_elapsed_time = events[file_path].get("operation_elapsed_time", None)
        file_name = Path(file_path).stem

        result.append(
            LogData(
                file_name=file_name,
                total_elapsed_time_sec=total_elapsed_time,
                operation_elapsed_time_sec=(
                    operation_elapsed_time if operation_elapsed_time is not None else total_elapsed_time
                ),
            )
        )

    return result
