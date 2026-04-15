FROM python:3.11-slim

# Prevent Python from buffering logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Default camera source (can override in compose)
ENV CAMERA_SOURCE=0

WORKDIR /app

# Install system dependencies for OpenCV + torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-6 \
    libxau6 \
    libxdmcp6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY camera/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy your code
COPY . .

# Expose API port
EXPOSE 8000

# Start FastAPI camera service
CMD ["uvicorn", "main7:app", "--host", "0.0.0.0", "--port", "8000"]