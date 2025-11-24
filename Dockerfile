# Use official lightweight Python image
FROM python:3.11-slim

# Don't buffer logs
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Install system deps (optional but good for many Python libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install them
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your code
COPY . /app

# Default command: run your AI backtester
CMD ["python", "main.py"]
