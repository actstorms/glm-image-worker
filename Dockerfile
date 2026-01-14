# GLM-Image Worker Dockerfile
# Based on: https://huggingface.co/zai-org/GLM-Image

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Note: Installing transformers and diffusers from source as required by GLM-Image
RUN pip install --no-cache-dir -r requirements.txt

# Copy server code
COPY server.py .

# Create cache directory for models
RUN mkdir -p /root/.cache/huggingface

# Expose port
EXPOSE 8000

# Health check (longer start period for ~16B+ model loading)
HEALTHCHECK --interval=60s --timeout=60s --start-period=600s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default environment variables
ENV MODEL_ID=zai-org/GLM-Image
ENV TORCH_DTYPE=bfloat16
ENV DEFAULT_HEIGHT=1024
ENV DEFAULT_WIDTH=1024
ENV DEFAULT_STEPS=50
ENV DEFAULT_GUIDANCE_SCALE=1.5
ENV HF_HOME=/root/.cache/huggingface

# Run the server
CMD ["python", "server.py"]
