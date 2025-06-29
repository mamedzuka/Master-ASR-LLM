#!/usr/bin/env python3
"""
Основной скрипт для обработки датасета подкастов.
Поддерживает как полный пайплайн (аудио+текст), так и текстовый-only режим.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def parse_args():
    """Парсинг аргументов командной строки"""
    parser = argparse.ArgumentParser(
        description="Полный пайплайн обработки датасета подкастов"
    )
    
    # Настройки загрузки
    parser.add_argument('--skip-download', action='store_true',
                       help="Пропустить этап загрузки подкастов")
    parser.add_argument('--start-episode', type=int, default=960,
                       help="Номер первого выпуска для загрузки")
    parser.add_argument('--end-episode', type=int, default=0,
                       help="Номер последнего выпуска для загрузки")
    
    # Пути к данным
    parser.add_argument('--audio-dir', type=Path, default='audio',
                       help="Папка для хранения аудиофайлов")
    parser.add_argument('--transcript-dir', type=Path, default='transcripts',
                       help="Папка для хранения транскриптов")
    parser.add_argument('--dataset-dir', type=Path, default='dataset',
                       help="Корневая папка датасета для создания")
    
    # Параметры обработки
    parser.add_argument('--chunk-minutes', type=int, default=5,
                       help="Длина сегментов в минутах")
    parser.add_argument('--move-files', action='store_true',
                       help="Перемещать файлы вместо копирования")
    
    # Текстовый режим
    parser.add_argument('--text-only', action='store_true',
                       help="Создать только текстовый датасет")
    parser.add_argument('--remove-speakers', action='store_true',
                       help="Удалить информацию о спикерах в текстовом режиме")
    parser.add_argument('--max-workers', type=int, default=4,
                       help="Максимальное число потоков для обработки")
    
    return parser.parse_args()

def download_podcasts(start: int, end: int, output_dir: Path):
    """Загружает подкасты с start по end номер в указанную папку"""
    from utils.download import download_podcasts as dp
    dp(start=start, end=end, output_dir=output_dir)

def organize_files(audio_dir: Path, transcript_dir: Path, dataset_dir: Path, move_files: bool):
    """Организует файлы в структурированный датасет"""
    from utils.organize import organize_dataset
    organize_dataset(
        audio_folder=audio_dir,
        transcript_folder=transcript_dir,
        dataset_folder=dataset_dir,
        move_files=move_files
    )

def process_audio_text_pairs(dataset_dir: Path, chunk_minutes: int):
    """Обрабатывает пары аудио-текст"""
    from utils.process import process_pairs
    process_pairs(dataset_dir=dataset_dir, chunk_minutes=chunk_minutes)

def validate_dataset(dataset_dir: Path):
    """Проверяет качество датасета"""
    from utils.validate import validate_dataset as vd
    vd(dataset_dir=dataset_dir)

def normalize_dataset(dataset_dir: Path):
    """Нормализует текстовые данные"""
    from utils.normalize import normalize_dataset as nd
    nd(dataset_dir=dataset_dir)

def create_text_only_dataset(dataset_dir: Path, remove_speakers: bool, max_workers: int):
    """Создает текстовый датасет"""
    from utils.text_processing import create_text_only_dataset as ct
    ct(
        source_dir=dataset_dir,
        output_dir=dataset_dir / "text_only",
        remove_speakers=remove_speakers,
        max_workers=max_workers
    )

def select_text_samples(dataset_dir: Path):
    """Выбирает образцы текстовых данных"""
    from utils.text_processing import select_samples
    selected = select_samples(
        source_dir=dataset_dir / "text_only",
        output_dir=dataset_dir / "text_samples",
        seed=42
    )
    
    # Вывод статистики
    print("\n=== Статистика выборки ===")
    for (people, duration), files in selected.items():
        print(f"Людей: {people}, Длительность: {duration} мин -> {len(files)} файлов")

def full_pipeline(args):
    """Выполняет полный процесс обработки датасета"""
    # 1. Загрузка подкастов
    if not args.skip_download:
        logging.info("\n=== Загрузка подкастов ===")
        download_podcasts(
            start=args.start_episode,
            end=args.end_episode,
            output_dir=args.audio_dir
        )
    
    # 2. Организация файлов
    logging.info("\n=== Организация файлов ===")
    organize_files(
        audio_dir=args.audio_dir,
        transcript_dir=args.transcript_dir,
        dataset_dir=args.dataset_dir,
        move_files=args.move_files
    )
    
    # 3. Обработка пар аудио-текст
    logging.info("\n=== Обработка пар аудио-текст ===")
    process_audio_text_pairs(
        dataset_dir=args.dataset_dir,
        chunk_minutes=args.chunk_minutes
    )
    
    # 4. Валидация
    logging.info("\n=== Валидация датасета ===")
    validate_dataset(dataset_dir=args.dataset_dir)
    
    # 5. Нормализация текста
    logging.info("\n=== Нормализация текста ===")
    normalize_dataset(dataset_dir=args.dataset_dir)

def text_only_pipeline(args):
    """Пайплайн для создания текстового датасета"""
    # 1. Создание текстового датасета
    logging.info("\n=== Создание текстового датасета ===")
    create_text_only_dataset(
        dataset_dir=args.dataset_dir,
        remove_speakers=args.remove_speakers,
        max_workers=args.max_workers
    )
    
    # 2. Проверка длительностей
    logging.info("\n=== Проверка длительностей текста ===")
    from utils.text_utils import check_text_durations
    check_text_durations(args.dataset_dir / "text_only")
    
    # 3. Удаление коротких файлов
    logging.info("\n=== Удаление коротких текстовых файлов ===")
    from utils.text_utils import remove_short_text_files
    remove_short_text_files(args.dataset_dir / "text_only")
    
    # 4. Выборка файлов
    logging.info("\n=== Выборка текстовых файлов ===")
    select_text_samples(args.dataset_dir)

def main():
    args = parse_args()
    
    # Создаем необходимые директории
    args.audio_dir.mkdir(exist_ok=True)
    args.transcript_dir.mkdir(exist_ok=True)
    args.dataset_dir.mkdir(exist_ok=True)
    
    # Запускаем соответствующий пайплайн
    if args.text_only:
        text_only_pipeline(args)
    else:
        full_pipeline(args)
    
    logging.info("\n=== Обработка завершена ===")

if __name__ == '__main__':
    main()
