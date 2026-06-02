FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

COPY src/ /app/src/

ENV PYTHONPATH=/app/src:/app/src/pb
ENV PYTHONUNBUFFERED=1

EXPOSE 3232

CMD ["python3", "src/main.py"]