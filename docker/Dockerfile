FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . .

# Install Whale1.0
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r requirements/base.txt

# Create output dir
RUN mkdir -p /app/outputs /app/models

ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
