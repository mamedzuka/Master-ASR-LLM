"""
Модуль для обработки текстовых данных датасета
"""

import re
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor

# Шаблоны для обработки текста
SPEAKER_PATTERN = re.compile(r'^\s*\[([^\]]+)\]\(\d{1,2}:\d{2}(?::\d{2})?\):\s*(.*)')
TIMECODE_PATTERN = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\b')

def normalize_text(text: str, remove_speakers: bool = True) -> str:
    """Нормализует текст с опциональным удалением информации о спикерах"""
    lines = text.splitlines()
    cleaned = []
    
    for line in lines:
        if remove_speakers:
            line = SPEAKER_PATTERN.sub('', line)
            line = TIMECODE_PATTERN.sub('', line)
        
        line = re.sub(r'[^\w\s]', ' ', line)
        line = line.lower().strip()
        line = re.sub(r'\s+', ' ', line)
        
        if line:
            cleaned.append(line)
    
    return '\n'.join(cleaned)

def process_text_file(input_path: Path, output_path: Path, remove_speakers: bool = True):
    """Обрабатывает один текстовый файл"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        normalized = normalize_text(text, remove_speakers)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(normalized)
            
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def create_text_only_dataset(
    source_dir: Path,
    output_dir: Path,
    remove_speakers: bool = True,
    max_workers: int = 4
):
    """Создает текстовый датасет из исходных файлов"""
    text_files = list(source_dir.rglob('*.txt'))
    text_files = [f for f in text_files if 'normalize' not in f.name and 'no_speakers' not in f.name]
    
    print(f"Found {len(text_files)} text files to process")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for input_path in text_files:
            suffix = 'no_speakers' if remove_speakers else 'normalize'
            output_name = input_path.stem + f'.{suffix}.txt'
            output_path = output_dir / input_path.relative_to(source_dir).parent / output_name
            futures.append(executor.submit(process_text_file, input_path, output_path, remove_speakers))
        
        for future in futures:
            future.result()

def select_samples(
    source_dir: Path,
    output_dir: Path,
    samples_per_group: int = 5,
    people_counts: List[int] = [2, 3, 4, 5, 6],
    durations: List[int] = [5, 10, 20, 40, 60],
    seed: int = None
) -> Dict[Tuple[int, int], List[str]]:
    """Выбирает образцы файлов по категориям"""
    if seed is not None:
        random.seed(seed)
    
    pattern = re.compile(r".*\.(?P<people>\d+)_people\.(?P<duration>\d+)_mins\..*")
    file_groups = {(p, d): [] for p in people_counts for d in durations}
    
    for file_path in source_dir.rglob('*.txt'):
        match = pattern.match(file_path.name)
        if not match:
            continue
        
        people = int(match.group('people'))
        duration = int(match.group('duration'))
        
        if people in people_counts and duration in durations:
            file_groups[(people, duration)].append(file_path)
    
    selected_files = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for group, files in file_groups.items():
        if not files:
            print(f"Warning: No files found for group {group}")
            continue
        
        files_sorted = sorted(files, key=lambda x: x.name)
        selected = (
            files_sorted[:samples_per_group] 
            if len(files_sorted) <= samples_per_group
            else random.sample(files_sorted, samples_per_group)
        )
        
        selected_files[group] = [f.name for f in selected]
        
        for src in selected:
            dst = output_dir / src.name
            shutil.copy2(src, dst)
    
    return selected_files
