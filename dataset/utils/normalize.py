"""
Модуль для нормализации текста
"""

import re
from pathlib import Path

def normalize_text(text: str) -> str:
    """
    Нормализует текст: удаляет метки, пунктуацию, приводит к нижнему регистру.
    """
    lines = text.splitlines()
    cleaned_lines = []
    prefix_pattern = re.compile(r'^\s*\[.*?\]\(.*?\):\s*')
    
    for line in lines:
        no_label = prefix_pattern.sub('', line)
        cleaned_lines.append(no_label)
    
    joined = ' '.join(cleaned_lines)
    without_punct = re.sub(r'[^\w\s]', ' ', joined)
    lowercased = without_punct.lower()
    normalized = re.sub(r'\s+', ' ', lowercased).strip()
    return normalized

def normalize_dataset(dataset_dir: Path):
    """
    Нормализует все текстовые файлы в датасете.
    
    Args:
        dataset_dir: Корневая папка датасета
    """
    for txt_path in dataset_dir.rglob('*.txt'):
        # Пропускаем уже нормализованные файлы
        if 'normalize' in txt_path.name:
            continue
        
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                original = f.read()
            
            normalized = normalize_text(original)
            
            # Создаем новое имя файла с 'normalize'
            parts = txt_path.stem.split('.')
            if len(parts) > 3:  # предполагаем формат name.id.people.mins...
                parts.insert(4, 'normalize')
                new_name = '.'.join(parts) + '.txt'
            else:
                new_name = txt_path.stem + '.normalize.txt'
            
            new_path = txt_path.with_name(new_name)
            
            with open(new_path, 'w', encoding='utf-8') as f:
                f.write(normalized)
            
            print(f"Normalized: {txt_path} -> {new_path}")
        
        except Exception as e:
            print(f"Error processing {txt_path}: {e}")
