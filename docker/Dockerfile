# Multi-stage Dockerfile for distributed LLM model serving
FROM nvidia/cuda:11.8-devel-ubuntu20.04 as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    git \
    curl \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
RUN update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip
RUN pip install --upgrade pip

# Training stage
FROM base as training

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install additional training dependencies
RUN pip install \
    deepspeed \
    accelerate \
    wandb \
    tensorboard

# Copy source code
COPY src/ ./src/
COPY azure/ ./azure/
COPY examples/ ./examples/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Default command for training
CMD ["python", "examples/train_code_model.py"]

# Serving stage (optimized for inference)
FROM base as serving

WORKDIR /app

# Install minimal dependencies for serving
RUN pip install \
    torch==2.1.0 \
    transformers==4.36.0 \
    tokenizers==0.15.0 \
    fastapi==0.104.0 \
    uvicorn==0.24.0 \
    pydantic==2.5.0 \
    peft==0.7.0 \
    safetensors==0.4.0 \
    prometheus-client==0.19.0 \
    psutil==5.9.0

# Copy only necessary files for serving
COPY src/serving/ ./src/serving/
COPY src/optimization/ ./src/optimization/
COPY src/monitoring/ ./src/monitoring/
COPY deployment/ ./deployment/

# Create model directory
RUN mkdir -p /app/models

# Set Python path
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Environment variables for serving
ENV MODEL_PATH="/app/models" \
    DEVICE="cpu" \
    MAX_BATCH_SIZE="8" \
    OPTIMIZATION_LEVEL="O2"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command for serving
CMD ["python", "-m", "uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000"]

# Development stage
FROM training as development

# Install development dependencies
RUN pip install \
    pytest \
    black \
    ruff \
    mypy \
    pre-commit \
    jupyter \
    ipython \
    notebook

# Copy test files
COPY tests/ ./tests/

# Install package in development mode
RUN pip install -e .

# Default command for development
CMD ["/bin/bash"]