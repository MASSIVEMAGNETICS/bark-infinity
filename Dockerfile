# Bark Infinity - Multi-stage Docker Build
# Supports both CPU and GPU deployment

ARG PYTHON_VERSION=3.11
ARG CUDA_VERSION=11.8.0

# Stage 1: Base image with common dependencies
FROM python:${PYTHON_VERSION}-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 bark && \
    chown -R bark:bark /app

# Stage 2: CPU-only image
FROM base AS cpu

USER bark

# Copy requirements
COPY --chown=bark:bark requirements-pip.txt .
COPY --chown=bark:bark pyproject.toml .
COPY --chown=bark:bark setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e .

# Copy application code
COPY --chown=bark:bark . .

# Set environment variables for CPU optimization
ENV SUNO_OFFLOAD_CPU=True \
    SUNO_USE_SMALL_MODELS=True \
    BARK_QUANTIZE_8BIT=False \
    OMP_NUM_THREADS=4 \
    MKL_NUM_THREADS=4

# Expose ports for web interfaces
EXPOSE 7860 8501

# Default command (can be overridden)
CMD ["python", "bark_webui.py"]

# Stage 3: GPU-enabled image (NVIDIA CUDA)
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu22.04 AS gpu

ARG PYTHON_VERSION

# Install Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create non-root user
RUN useradd -m -u 1000 bark && \
    chown -R bark:bark /app

USER bark

# Copy requirements
COPY --chown=bark:bark requirements-pip.txt .
COPY --chown=bark:bark pyproject.toml .
COPY --chown=bark:bark setup.py .

# Install Python dependencies with CUDA support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install --no-cache-dir -e .

# Install optional quantization dependencies
RUN pip install --no-cache-dir bitsandbytes optimum || true

# Copy application code
COPY --chown=bark:bark . .

# Set environment variables for GPU optimization
ENV CUDA_VISIBLE_DEVICES=0 \
    SUNO_OFFLOAD_CPU=False \
    SUNO_USE_SMALL_MODELS=False \
    BARK_QUANTIZE_8BIT=True

# Expose ports for web interfaces
EXPOSE 7860 8501

# Default command
CMD ["python", "bark_webui.py"]

# Stage 4: Production image (auto-detect GPU/CPU)
FROM base AS production

USER bark

# Copy requirements
COPY --chown=bark:bark requirements-pip.txt .
COPY --chown=bark:bark pyproject.toml .
COPY --chown=bark:bark setup.py .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e .

# Copy application code
COPY --chown=bark:bark . .

# Environment variables (will auto-detect GPU)
ENV SUNO_OFFLOAD_CPU=True \
    PYTHONUNBUFFERED=1

# Expose ports
EXPOSE 7860 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import bark_infinity; print('healthy')" || exit 1

# Copy entrypoint script
COPY --chown=bark:bark docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "bark_webui.py"]
