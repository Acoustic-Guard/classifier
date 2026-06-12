FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

RUN mkdir -p /app/models && \
    wget -q -O /app/models/yamnet.tflite \
        "https://storage.googleapis.com/audioset/yamnet.tflite"

COPY src/ /app/src/

ENV PYTHONPATH=/app/src:/app/src/pb
ENV PYTHONUNBUFFERED=1

ENV MODEL_PATH=models/random_forest_v2.joblib
ENV YAMNET_MODEL_PATH=models/yamnet.tflite

EXPOSE 3232

CMD ["python3", "src/main.py"]