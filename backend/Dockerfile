FROM openjdk:11-jdk-slim as java

FROM python:3.9-slim

COPY --from=java /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME=/usr/local/openjdk-11

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyspark

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]