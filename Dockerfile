FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \    ffmpeg \    libgl1 \    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "cli.py", "--help"]
