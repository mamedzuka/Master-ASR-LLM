import re
import csv
import logging
import time
import json
import concurrent.futures
from collections import defaultdict
from typing import Literal, List, Dict, Callable, Optional, Union
from pathlib import Path

import nltk
from nltk.translate.bleu_score import SmoothingFunction
import pymorphy3
from rouge_score import rouge_scorer
from bert_score import score as bert_score

from utils.fs import find_files
from utils.logs import create_log_extra
from llm.sber import scripts as sber_scripts
from llm.yandex import scripts as yandex_scripts
from llm.openai import scripts as openai_scripts
from llm.google import scripts as google_scripts
from llm.azure import scripts as azure_scripts
from llm.deepseek import scripts as deepseek_scripts
from llm.anthropic import scripts as anthropic_scripts

logger = logging.getLogger(__name__)

exist_summarizer_systems = Literal["gigachat", "yandexgpt", "chatgpt", "gemini", "azure", "deepseek", "anthropic"]

summarizers: Dict[exist_summarizer_systems, Callable[[str], str]] = {
    "gigachat": sber_scripts.summarize_transcript,
    "yandexgpt": yandex_scripts.summarize_transcript,
    "chatgpt": openai_scripts.summarize_transcript,
    "gemini": google_scripts.summarize_transcript,
    "azure": azure_scripts.summarize_transcript,
    "deepseek": deepseek_scripts.summarize_transcript,
    "anthropic": anthropic_scripts.summarize_transcript,
}


class LLMProcessor:
    @staticmethod
    def summarize(
        dataset_dir: str,
        output_dir: str,
        system: exist_summarizer_systems,
        model: str,
        temperature: Optional[float] = None,
        max_threads: int = 1,
    ):
        summarizer = summarizers[system]
        summarizer_wrapper = _summarizer_wrapper(summarizer)

        output_dir_path = Path(output_dir) / system / model
        output_dir_path.mkdir(parents=True, exist_ok=True)

        exist_transcripts = {transcipt_path.stem for transcipt_path in find_files(str(output_dir_path), ".json")}

        count_processed_transcript = 0
        transcript_paths = list(
            filter(
                lambda x: x.stem not in exist_transcripts,
                find_files(dataset_dir, ".txt"),
            ),
        )

        logger.info(
            "found %d transcripts in dataset directory: '%s', start summarization", len(transcript_paths), dataset_dir
        )

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = [
                executor.submit(summarizer_wrapper, str(transcript_path), str(output_dir_path), model, temperature)
                for transcript_path in transcript_paths
            ]
            for future in concurrent.futures.as_completed(futures):
                count_processed_transcript += 1 if future.result() is not None else 0

        logger.info(
            "summarization completed for %d transcripts out of %d",
            count_processed_transcript,
            len(transcript_paths),
        )

    @staticmethod
    def metrics(references_dir: str, hypotheses_dir: str, log_path: str):
        references_paths = list(find_files(references_dir, ".json"))
        hypotheses_paths = list(find_files(hypotheses_dir, ".json"))

        # group references by filename
        references_paths_by_filename = {}
        for reference_path in references_paths:
            references_paths_by_filename.setdefault(reference_path.stem, []).append(reference_path)

        log_data = _parse_log_file(log_path)

        result = []
        for hypotheses_path in hypotheses_paths:
            if hypotheses_path.stem not in references_paths_by_filename:
                continue

            refs = [
                _extract_text_from_summary_file(ref_path)
                for ref_path in references_paths_by_filename[hypotheses_path.stem]
            ]
            refs = list(filter(lambda x: len(x) > 0, refs))
            hyp = _extract_text_from_summary_file(hypotheses_path)

            if len(hyp) == 0 or len(refs) == 0:
                continue

            rouge_metrics = _get_rouge_metrics(refs, hyp)
            bs_p, bs_r, bs_f = _get_bert_score_metrics(refs, hyp)
            bleu_1, bleu_2, bleu_4 = _get_bleu_metrics(refs, hyp)
            meteor = _get_meteor_metrics(refs, hyp)
            file_name, person_count, time_in_minute = _parse_file_path(str(hypotheses_path))
            elapsed_time_sec = log_data.get(hypotheses_path.stem, 0)

            result.append(
                (
                    file_name,
                    person_count,
                    time_in_minute,
                    f"{rouge_metrics['rouge1']['r']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rouge1']['p']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rouge1']['f']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rouge2']['r']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rouge2']['p']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rouge2']['f']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rougeL']['r']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rougeL']['p']:.3f}".replace(".", ","),
                    f"{rouge_metrics['rougeL']['f']:.3f}".replace(".", ","),
                    f"{bs_r:.3f}".replace(".", ","),
                    f"{bs_p:.3f}".replace(".", ","),
                    f"{bs_f:.3f}".replace(".", ","),
                    f"{bleu_1:.3f}".replace(".", ","),
                    f"{bleu_2:.3f}".replace(".", ","),
                    f"{bleu_4:.3f}".replace(".", ","),
                    f"{meteor:.3f}".replace(".", ","),
                    f"{elapsed_time_sec:.3f}".replace(".", ","),
                )
            )

        result.sort(key=lambda x: (x[2], x[1]))

        response_path = Path(hypotheses_dir) / "metrics.csv"
        with open(response_path, "w", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(
                [
                    "file_name",
                    "person_count",
                    "file_time_in_minute",
                    "rouge1_r",
                    "rouge1_p",
                    "rouge1_f",
                    "rouge2_r",
                    "rouge2_p",
                    "rouge2_f",
                    "rougeL_r",
                    "rougeL_p",
                    "rougeL_f",
                    "bertscore_r",
                    "bertscore_p",
                    "bertscore_f",
                    "bleu_1",
                    "bleu_2",
                    "bleu_4",
                    "meteor",
                    "elapsed_time_sec",
                ]
            )
            writer.writerows(result)


def _summarizer_wrapper(summarizer: Callable) -> Callable:
    def summarizer_wrapper(transcript_path: str, output_dir: str, model: str, temperature: Optional[float] = None):
        try:
            logger.info("starting summarization", extra=create_log_extra(current_file=transcript_path))
            with open(transcript_path, "r") as f:
                transcript = f.read()

            start_time = time.time()
            result = summarizer(
                transcript=transcript,
                model=model,
                temperature=temperature,
            )
            elapsed_time = time.time() - start_time

            output_path = Path(output_dir) / (Path(transcript_path).stem + ".json")
            with open(output_path, "w") as f:
                f.write(result)

            logger.info(
                "summarization completed, output saved to: '%s'",
                output_path,
                extra=create_log_extra(current_file=transcript_path, elapsed_time_sec=elapsed_time),
            )
            return str(output_path)

        except Exception:
            logger.exception("error while summarizing", extra=create_log_extra(current_file=transcript_path))
            return None

    return summarizer_wrapper


def _extract_text_from_summary_file(summary_path: Union[str, Path]) -> str:
    with open(summary_path) as f:
        try:
            sum_resp = json.loads(f.read())
            return " ".join([sum_resp["title"], sum_resp["tldr"], *sum_resp["resume"], sum_resp["conclusion"]])
        except json.JSONDecodeError:
            return ""


def _get_meteor_metrics(refs: List[str], hyp: str):
    normalized_refs = [_get_normalize_tokens(ref) for ref in refs]
    normalized_hyp = _get_normalize_tokens(hyp)

    return nltk.meteor(normalized_refs, normalized_hyp)


def _get_bert_score_metrics(refs: List[str], hyp: str):
    p, r, f = bert_score([hyp] * len(refs), refs, lang="ru")
    return p.mean().item(), r.mean().item(), f.mean().item()


def _get_bleu_metrics(refs: List[str], hyp: str):
    smoothie = SmoothingFunction().method4

    normalized_refs = [_get_normalize_tokens(ref) for ref in refs]
    normalized_hyp = _get_normalize_tokens(hyp)

    bleu_1 = nltk.bleu(normalized_refs, normalized_hyp, weights=(1, 0, 0, 0))
    bleu_2 = nltk.bleu(normalized_refs, normalized_hyp, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
    bleu_4 = nltk.bleu(normalized_refs, normalized_hyp, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

    return bleu_1, bleu_2, bleu_4


def _get_rouge_metrics(refs: List[str], hyp: str):
    normalized_refs = [" ".join(_get_normalize_tokens(ref)) for ref in refs]
    normalized_hyp = " ".join(_get_normalize_tokens(hyp))

    metrics = ["rouge1", "rouge2", "rougeL"]
    scorer = rouge_scorer.RougeScorer(metrics)

    result = {metric: defaultdict(float) for metric in metrics}

    for ref in normalized_refs:
        score = scorer.score(ref, normalized_hyp)
        for metric in score:
            result[metric]["r"] += score[metric].recall / len(normalized_refs)
            result[metric]["p"] += score[metric].precision / len(normalized_refs)
            result[metric]["f"] += score[metric].fmeasure / len(normalized_refs)

    return result


def _get_normalize_tokens(text: str) -> list[str]:
    text = text.lower()

    tokens = nltk.word_tokenize(text, language="russian")

    morph = pymorphy3.MorphAnalyzer()
    tokens = [morph.parse(token)[0].normal_form for token in tokens]

    return tokens


def _parse_log_file(log_path: str):
    re_complete = re.compile(
        r"summarization completed, output saved to: .+ \| extra_variables - current_file: (?P<file_path>.+), elapsed_time_sec: (?P<elapsed>[\d.]+)"
    )

    result = {}
    with open(log_path, "r") as f:
        for line in f:
            m_complete = re_complete.search(line)
            if m_complete:
                file_path = Path(m_complete.group("file_path"))
                elapsed_time_sec = m_complete.group("elapsed")
                result[file_path.stem] = float(elapsed_time_sec)

    return result


def _parse_file_path(file_path: str):
    file_name = Path(file_path).stem
    file_name_parts = file_name.split(".")
    people_count = file_name_parts[3].split("_")[0]
    time_in_minute = file_name_parts[4].split("_")[0]
    return file_name, people_count, time_in_minute
