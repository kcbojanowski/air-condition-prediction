# Stage 1: Install Java
FROM openjdk:11-jdk-slim as java

# Stage 2: Build the Python application
FROM python:3.9-slim

# Copy Java from the previous stage
COPY --from=java /usr/local/openjdk-11 /usr/local/openjdk-11
ENV JAVA_HOME=/usr/local/openjdk-11

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pyspark

# Copy the rest of the application
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]