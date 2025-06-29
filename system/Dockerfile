FROM python:3.11-slim

WORKDIR /app

# Ставим системные зависимости ffmpeg (нужен для mp3) и gcc для symspell
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg build-essential && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# Переменные окружения можно передавать через --env-file
CMD ["python", "-m", "src.main"]

