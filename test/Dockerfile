# Базовый образ с Python
FROM python:3.10-slim

# Рабочая директория внутри контейнера
WORKDIR /app

# Установка системных зависимостей (если нужны компиляционные инструменты)
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

# Копируем файл с зависимостями
COPY requirements.txt .

# Устанавливаем Python-зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Отключаем буферизацию вывода (чтобы логи сразу выводились)
ENV PYTHONUNBUFFERED=1

# Точка входа: запускаем ваш CLI-скрипт
ENTRYPOINT ["python", "main.py"]

# По умолчанию показываем справку
CMD ["--help"]

