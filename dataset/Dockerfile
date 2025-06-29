# Базовый образ с Python
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей (например, для сборки пакетов или работы с аудио)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями (если есть)
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект в контейнер
COPY . .

# Отключаем буферизацию вывода, чтобы логи шли сразу
ENV PYTHONUNBUFFERED=1

# Точка входа: запускаем ваш пайплайн
ENTRYPOINT ["python", "pipeline.py"]

# По умолчанию выводим помощь
CMD ["--help"]

