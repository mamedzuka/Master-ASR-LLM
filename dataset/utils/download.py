"""
Модуль для загрузки подкастов
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from pathlib import Path

BASE_URL = "https://archive.rucast.net/radio-t/media/rt_podcast{num}.mp3"
WORKERS = 5
TIMEOUT = 15
RETRY = 2

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def download_episode(num: int, output_dir: Path):
    """Скачивает один mp3-файл и сохраняет его в указанную папку."""
    url = BASE_URL.format(num=num)
    filename = f"radio-t.record.{num}.mp3"
    local_path = output_dir / filename

    if local_path.exists():
        logging.info(f"[{num}] Пропускаю — уже есть {filename}")
        return

    for attempt in range(1, RETRY + 1):
        try:
            logging.info(f"[{num}] Загружаю {url} (попытка {attempt})")
            resp = requests.get(url, stream=True, timeout=TIMEOUT)
            resp.raise_for_status()

            with open(local_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            logging.info(f"[{num}] Сохранён → {local_path}")
            return
        except Exception as e:
            logging.warning(f"[{num}] Ошибка: {e} (попытка {attempt}/{RETRY})")

    logging.error(f"[{num}] Не удалось скачать после {RETRY} попыток")

def download_podcasts(start: int, end: int, output_dir: Path):
    """
    Загружает подкасты с start по end номер в указанную папку.
    
    Args:
        start: Номер первого выпуска
        end: Номер последнего выпуска (включительно)
        output_dir: Папка для сохранения
    """
    output_dir.mkdir(exist_ok=True)
    nums = list(range(start, end - 1, -1))
    
    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {
            executor.submit(download_episode, n, output_dir): n 
            for n in nums
        }
        for future in as_completed(futures):
            _ = future.result()
