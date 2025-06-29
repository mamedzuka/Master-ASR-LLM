"""
Модуль для организации файлов датасета
"""

import re
import shutil
import sys
from pathlib import Path

def clean_name(filename: str) -> str:
    """Очищает имя файла от лишних маркеров"""
    s = filename
    s = s.replace('.record.', '.')
    s = s.replace('.benchmark.', '.')
    s = re.sub(r'\.{2,}', '.', s)
    return s

def organize_dataset(
    audio_folder: Path,
    transcript_folder: Path,
    dataset_folder: Path,
    move_files: bool = False
):
    """
    Организует аудио и текстовые файлы в структурированный датасет.
    
    Args:
        audio_folder: Папка с аудиофайлами
        transcript_folder: Папка с транскриптами
        dataset_folder: Корневая папка датасета
        move_files: Если True, перемещает файлы вместо копирования
    """
    # Проверка входных директорий
    if not audio_folder.is_dir():
        raise FileNotFoundError(f"Audio folder does not exist: {audio_folder}")
    if not transcript_folder.is_dir():
        raise FileNotFoundError(f"Transcript folder does not exist: {transcript_folder}")

    processed = 0
    pattern = re.compile(r'(?P<people>\d+)_people\.(?P<duration>\d+)_mins')

    for audio_path in audio_folder.glob('*.mp3'):
        name = audio_path.name
        m = pattern.search(name)
        if not m:
            print(f"Warning: could not parse people/duration from '{name}', skipping.")
            continue
        
        people = m.group('people')
        duration = m.group('duration')

        # Поиск соответствующего транскрипта
        suffix = name.split('record.', 1)[1].rsplit('.mp3', 1)[0]
        candidates = list(transcript_folder.glob(f'*{suffix}.txt'))
        if not candidates:
            print(f"Warning: transcript not found for '{name}', skipping.")
            continue
        if len(candidates) > 1:
            print(f"Warning: multiple transcripts found for '{name}', using '{candidates[0].name}'")
        transcript_path = candidates[0]

        # Очистка имен файлов
        new_audio_name = clean_name(name)
        new_transcript_name = clean_name(transcript_path.name)

        # Создание структуры папок
        dest_dir = dataset_folder / f"{people}_people" / f"{duration}_mins"
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Копирование/перемещение файлов
        op = shutil.move if move_files else shutil.copy2
        op(str(audio_path), str(dest_dir / new_audio_name))
        op(str(transcript_path), str(dest_dir / new_transcript_name))

        processed += 1

    print(f"Done. Total items organized: {processed}")
