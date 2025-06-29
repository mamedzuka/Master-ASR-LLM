import argparse
from typing import get_args
from pathlib import Path

from logging_setup import configure_logging
from asr.processor import ASRProcessor, exist_transcriber_systems, exist_parser_systems
from llm.processor import LLMProcessor, exist_summarizer_systems

command_processors = {
    "asr": ASRProcessor,
    "llm": LLMProcessor,
}


def main():
    parser = argparse.ArgumentParser(
        description="A tool for running experiments with different ASR or LLM systems.",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands", required=True)

    asr_parser = subparsers.add_parser(
        "asr",
        help="run ASR",
    )

    asr_subparsers = asr_parser.add_subparsers(
        dest="sub_command",
        help="Available ASR commands",
        required=True,
    )

    asr_trancribe_parser = asr_subparsers.add_parser(
        "transcribe",
        help="Transcribe audio files using a specified ASR system.",
        description="Transcribe audio files using a specified ASR system. All mp3 files in dataset dir and its",
    )
    asr_trancribe_parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset directory containing audio files. Find all mp3 files in this directory and this subdirs.",
    )
    asr_trancribe_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Path to the output directory where results will be saved. Repeats the structure of dataset dir and names of audio files. All outputs files have txt extension.",
    )
    asr_trancribe_parser.add_argument(
        "--system",
        "-s",
        type=str,
        choices=get_args(exist_transcriber_systems),
        required=True,
        help="ASR system to use for transcription.",
    )
    asr_trancribe_parser.add_argument(
        "--max-threads",
        "-t",
        type=int,
        default=1,
        help="Max threads used for transcribe audios. Default value is 1",
    )

    asr_parse_parser = asr_subparsers.add_parser(
        "parse",
        help="Parse JSON transcribed files using a specified parser system.",
        description="Parse JSON files using a specified parser system. All json files in input dir and its subdirs.",
    )

    asr_parse_parser.add_argument(
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Path to the input directory containing JSON files. Find all json files in this directory and this subdirs.",
    )
    asr_parse_parser.add_argument(
        "--system",
        "-s",
        type=str,
        choices=get_args(exist_parser_systems),
        required=True,
        help="Parser system to use for parsing JSON files.",
    )

    asr_metrics_parser = asr_subparsers.add_parser(
        "metrics",
        help="Calculate metrics for ASR results.",
        description="Calculate metrics for ASR results. Not implemented yet.",
    )

    asr_metrics_parser.add_argument(
        "--reference-dir",
        "-rd",
        type=str,
        required=True,
        help="Path to the directory containing reference text files.",
    )
    asr_metrics_parser.add_argument(
        "--hypotheses-dir",
        "-hd",
        type=str,
        required=True,
        help="Path to the directory containing hypotheses text files.",
    )
    asr_metrics_parser.add_argument(
        "--log-path",
        "-l",
        type=str,
        required=True,
        help="Path to the log file where metrics will be saved.",
    )

    llm_parser = subparsers.add_parser(
        "llm",
        help="run LLM",
    )

    llm_subparsers = llm_parser.add_subparsers(
        dest="sub_command",
        help="Available LLM commands",
        required=True,
    )

    llm_summarize_parser = llm_subparsers.add_parser(
        "summarize",
        help="Summarize transcripts using a specified LLM system.",
        description="Summarize transcripts using a specified LLM system. All txt files in dataset dir and its subdirs.",
    )
    llm_summarize_parser.add_argument(
        "--dataset-dir",
        "-d",
        type=str,
        required=True,
        help="Path to the dataset directory containing transcript files. Find all txt files in this directory and its subdirs.",
    )
    llm_summarize_parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        required=True,
        help="Path to the output directory where results will be saved. Repeats the structure of dataset dir and names of transcript files. All outputs files have json extension.",
    )
    llm_summarize_parser.add_argument(
        "--system",
        "-s",
        type=str,
        choices=get_args(exist_summarizer_systems),
        required=True,
        help="LLM system to use for summarization.",
    )
    llm_summarize_parser.add_argument(
        "--model",
        "-m",
        type=str,
        default=None,
        help="Name of the model to use",
    )
    llm_summarize_parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for responses from used model",
    )
    llm_summarize_parser.add_argument(
        "--max-threads",
        type=int,
        default=1,
        help="Max threads used for transcribe audios. Default value is 1",
    )

    llm_metrics_parser = llm_subparsers.add_parser(
        "metrics",
        help="Get core metrics for result of the summary",
    )
    llm_metrics_parser.add_argument(
        "--references-dir",
        "-rd",
        type=str,
        required=True,
        help="Path to directory of the references summary transcript",
    )
    llm_metrics_parser.add_argument(
        "--hypotheses-dir",
        "-hd",
        type=str,
        required=True,
        help="Path to directory of the hypotheses summary transcript",
    )
    llm_metrics_parser.add_argument(
        "--log-path",
        "-l",
        type=str,
        required=True,
        help="Path to the log file where metrics will be saved.",
    )

    args = parser.parse_args()

    processor = command_processors[args.command]
    handler = getattr(processor, args.sub_command)

    kwargs = vars(args).copy()
    kwargs.pop("command", None)
    kwargs.pop("sub_command", None)

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_name = kwargs.get("system", "")
    log_path = log_dir / (log_name + ".log")
    configure_logging(log_path)

    handler(**kwargs)


if __name__ == "__main__":
    main()
