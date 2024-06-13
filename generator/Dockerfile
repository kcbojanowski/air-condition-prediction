FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install websockets httpx rich click

COPY . .

EXPOSE 5000

ENTRYPOINT ["python", "generator.py"]
CMD ["--help"]