"""
Модуль для валидации датасета
"""

import os
import re
from pathlib import Path
from mutagen.mp3 import MP3
from datetime import timedelta

def check_durations(dataset_dir: Path):
    """
    Проверяет соответствие длительностей аудиофайлов и их временных меток.
    
    Args:
        dataset_dir: Корневая папка датасета
    """
    RANGE_REGEX = re.compile(r'(?P<range>\d{2}.\d{2}.\d{2}-\d{2}.\d{2}.\d{2})')

    def normalize_time_str(raw: str) -> str:
        s = raw.replace('.', ':').replace('_', ':')
        parts = s.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid time format: '{raw}'")
        return f"{parts[0].zfill(2)}:{parts[1].zfill(2)}:{parts[2].zfill(2)}"

    def parse_hhmmss_to_seconds(hhmmss: str) -> int:
        h, m, s = map(int, hhmmss.split(':'))
        return h * 3600 + m * 60 + s

    def format_seconds_to_hhmmss(total_seconds: float) -> str:
        total_seconds = int(round(total_seconds))
        td = timedelta(seconds=total_seconds)
        hours = td.seconds // 3600 + td.days * 24
        minutes = (td.seconds % 3600) // 60
        seconds = td.seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    for root, _, files in os.walk(dataset_dir):
        for fname in files:
            if not fname.lower().endswith(".mp3"):
                continue
            
            filepath = Path(root) / fname
            match = RANGE_REGEX.search(fname)
            if not match:
                print(f"{filepath} -> no time range found in name")
                continue

            try:
                start_raw, end_raw = match.group("range").split('-')
                start_norm = normalize_time_str(start_raw)
                end_norm = normalize_time_str(end_raw)
                
                start_sec = parse_hhmmss_to_seconds(start_norm)
                end_sec = parse_hhmmss_to_seconds(end_norm)
                duration_range_sec = end_sec - start_sec
                
                if duration_range_sec < 0:
                    duration_range_sec += 24 * 3600
                
                audio = MP3(filepath)
                duration_file_sec = audio.info.length
                
                duration_range_str = format_seconds_to_hhmmss(duration_range_sec)
                duration_file_str = format_seconds_to_hhmmss(duration_file_sec)
                
                rel_path = filepath.relative_to(dataset_dir)
                print(f"{rel_path} -> range: {duration_range_str}, file: {duration_file_str}")
                
            except Exception as e:
                print(f"{filepath} -> error: {e}")

def validate_dataset(dataset_dir: Path):
    """
    Выполняет все проверки качества датасета.
    
    Args:
        dataset_dir: Корневая папка датасета
    """
    print("Checking audio durations...")
    check_durations(dataset_dir)
