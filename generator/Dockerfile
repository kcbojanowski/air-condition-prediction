# Dockerfile for data generator
FROM python:3.10-slim

WORKDIR /app

COPY generator.py /app/generator.py

RUN pip install websockets

CMD ["python", "generator.py"]
