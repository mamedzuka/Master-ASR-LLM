"""
Утилиты для работы с текстовыми данными датасета
"""

import re
import os
from pathlib import Path
from datetime import timedelta

def parse_timecode(tc: str) -> int:
    """Парсит строку времени в секунды"""
    parts = [int(p) for p in tc.split(':')]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    raise ValueError(f"Invalid timecode: {tc}")

def format_seconds(sec: int) -> str:
    """Форматирует секунды в HH:MM:SS"""
    td = timedelta(seconds=sec)
    hours = td.seconds // 3600 + td.days * 24
    minutes = (td.seconds % 3600) // 60
    seconds = td.seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def check_text_durations(directory: Path, threshold_factor: float = 0.9):
    """
    Проверяет соответствие заявленной и фактической длительности текстовых файлов.
    Выводит отчет для каждого файла.
    """
    print(f"\nChecking text durations in: {directory}")
    
    total_files = 0
    problematic_files = 0
    
    for file_path in directory.rglob('*.txt'):
        if 'normalize' in file_path.name or 'no_speakers' in file_path.name:
            continue
        
        total_files += 1
        fname = file_path.name
        
        # Ищем заявленную длительность и временной интервал
        mins_match = re.search(r"\.(\d+)_mins\.", fname)
        interval_match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)", fname)
        
        if not mins_match or not interval_match:
            print(f"{fname} -> SKIP (invalid name format)")
            continue
        
        try:
            declared_mins = int(mins_match.group(1))
            declared_sec = declared_mins * 60
            threshold = declared_sec * threshold_factor
            
            start_tc, end_tc = interval_match.groups()
            start_sec = parse_timecode(start_tc)
            end_sec = parse_timecode(end_tc)
            
            actual_sec = end_sec - start_sec
            if actual_sec < 0:
                actual_sec += 24 * 3600
            
            status = "OK" if actual_sec >= threshold else "SHORT"
            if status == "SHORT":
                problematic_files += 1
            
            print(
                f"{fname} -> {status} | "
                f"Declared: {format_seconds(declared_sec)} | "
                f"Actual: {format_seconds(actual_sec)} | "
                f"Threshold: {format_seconds(int(threshold))}"
            )
            
        except Exception as e:
            print(f"{fname} -> ERROR: {str(e)}")
            problematic_files += 1
    
    print(f"\nSummary: {total_files} files checked, {problematic_files} problematic files found")

def remove_short_text_files(directory: Path, threshold_factor: float = 0.9):
    """
    Удаляет текстовые файлы, где фактическая длительность меньше threshold_factor от заявленной.
    """
    print(f"\nRemoving short text files in: {directory}")
    
    removed_count = 0
    kept_count = 0
    
    for file_path in directory.rglob('*.txt'):
        if 'normalize' in file_path.name or 'no_speakers' in file_path.name:
            continue
        
        fname = file_path.name
        mins_match = re.search(r"\.(\d+)_mins\.", fname)
        interval_match = re.search(r"(\d{1,2}:\d{2}(?::\d{2})?)-(\d{1,2}:\d{2}(?::\d{2})?)", fname)
        
        if not mins_match or not interval_match:
            print(f"{fname} -> SKIP (invalid name format)")
            continue
        
        try:
            declared_mins = int(mins_match.group(1))
            declared_sec = declared_mins * 60
            threshold = declared_sec * threshold_factor
            
            start_tc, end_tc = interval_match.groups()
            start_sec = parse_timecode(start_tc)
            end_sec = parse_timecode(end_tc)
            
            actual_sec = end_sec - start_sec
            if actual_sec < 0:
                actual_sec += 24 * 3600
            
            if actual_sec < threshold:
                file_path.unlink()
                print(f"REMOVED: {fname} (actual: {format_seconds(actual_sec)}, threshold: {format_seconds(int(threshold))})")
                removed_count += 1
            else:
                kept_count += 1
                
        except Exception as e:
            print(f"{fname} -> ERROR: {str(e)}")
            kept_count += 1
    
    print(f"\nSummary: {kept_count} files kept, {removed_count} files removed")

def validate_text_dataset(directory: Path):
    """
    Комплексная проверка текстового датасета:
    1. Проверка длительностей
    2. Удаление коротких файлов
    """
    check_text_durations(directory)
    remove_short_text_files(directory)
