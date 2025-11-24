# Bark Infinity - Deployment Guide

This guide covers deploying Bark Infinity across different platforms and environments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Windows Executable](#windows-executable)
3. [Docker Deployment](#docker-deployment)
4. [Web Deployment](#web-deployment)
5. [Low Compute Mode](#low-compute-mode)
6. [Mobile Access](#mobile-access)
7. [Cloud Deployment](#cloud-deployment)

---

## Quick Start

### Installation from PyPI (Recommended)

```bash
pip install bark-infinity
```

### Installation from Source

```bash
git clone https://github.com/MASSIVEMAGNETICS/bark-infinity.git
cd bark-infinity
pip install -e .
```

### Basic Usage

```python
import bark_infinity

# Generate audio
audio = bark_infinity.generate_audio("Hello world!")
```

---

## Windows Executable

### Download Pre-built Executable

1. Go to [Releases](https://github.com/MASSIVEMAGNETICS/bark-infinity/releases)
2. Download `bark-infinity-webui.exe`
3. Run the executable
4. A browser window will open with the web interface

### Build from Source

Requirements:
- Python 3.8 or higher
- PyInstaller

```bash
# Install build dependencies
pip install -e .[build]

# Run build script
bash scripts/build_windows.sh

# Or use PyInstaller directly
pyinstaller bark-infinity.spec
```

The executable will be in the `dist/` directory.

**Note:** First run will download models (~12GB) from Hugging Face.

---

## Docker Deployment

### Quick Start with Docker Compose

```bash
# CPU-only deployment
docker-compose up bark-infinity-cpu

# GPU-accelerated deployment (requires NVIDIA Docker)
docker-compose up bark-infinity-gpu

# Auto-detect deployment
docker-compose up bark-infinity
```

### Manual Docker Build

```bash
# Build CPU image
docker build --target cpu -t bark-infinity:cpu .

# Build GPU image
docker build --target gpu -t bark-infinity:gpu .

# Build production image
docker build --target production -t bark-infinity:latest .
```

### Run Docker Container

```bash
# CPU mode
docker run -p 7860:7860 -p 8501:8501 bark-infinity:cpu

# GPU mode
docker run --gpus all -p 7860:7860 -p 8501:8501 bark-infinity:gpu

# With persistent storage
docker run -v $(pwd)/bark_samples:/app/bark_samples \
           -p 7860:7860 \
           bark-infinity:latest
```

Access the web interface at:
- Gradio UI: http://localhost:7860
- Streamlit UI: http://localhost:8501

---

## Web Deployment

### Gradio Web UI

```bash
python bark_webui.py
```

Features:
- User-friendly interface
- Real-time audio generation
- Voice preset selection
- Download generated audio

### Streamlit Web UI

```bash
streamlit run bark_streamlit.py
```

Features:
- Interactive parameter controls
- Audio playback
- History tracking

### Deploy to Hugging Face Spaces

1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Select "Gradio" as the SDK
3. Upload the repository
4. Set environment variables:
   ```
   SUNO_OFFLOAD_CPU=True
   SUNO_USE_SMALL_MODELS=True
   ```

### Deploy to Streamlit Cloud

1. Fork the repository
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Create new app
4. Point to `bark_streamlit.py`
5. Deploy

---

## Low Compute Mode

For systems with limited GPU memory or CPU-only systems.

### Environment Variables

```bash
# Enable CPU offloading
export SUNO_OFFLOAD_CPU=True

# Use smaller models
export SUNO_USE_SMALL_MODELS=True

# Enable 8-bit quantization (requires GPU + bitsandbytes)
export BARK_QUANTIZE_8BIT=True

# Enable 4-bit quantization (experimental)
export BARK_QUANTIZE_4BIT=True

# Use BetterTransformer optimization
export BARK_USE_BETTER_TRANSFORMER=True
```

### Python API

```python
from bark_infinity import setup_low_compute_mode, generate_audio

# Configure low compute mode
config = setup_low_compute_mode()

# Generate audio with optimizations
audio = generate_audio("Hello world!")
```

### Memory Requirements

| Configuration | VRAM/RAM | Speed | Quality |
|---------------|----------|-------|---------|
| Full (GPU) | ~12GB VRAM | Fast | Best |
| Small Models (GPU) | ~8GB VRAM | Medium | Good |
| 8-bit Quantized | ~6GB VRAM | Medium | Good |
| 4-bit Quantized | ~3GB VRAM | Slow | Fair |
| CPU Offloading | ~2GB VRAM + 8GB RAM | Slow | Best |
| CPU Only | ~16GB RAM | Very Slow | Best |

### Install Quantization Dependencies

```bash
# Install quantization support
pip install bark-infinity[quantization]

# Or manually
pip install bitsandbytes optimum
```

---

## Mobile Access

**Important:** Bark Infinity cannot run directly on mobile devices (iOS/Android) due to:
- Large model size (~12GB)
- High computational requirements
- Memory constraints

### Recommended Approach: Remote Access

#### Option 1: Self-Hosted Server

Deploy Bark Infinity on a server and access via mobile browser:

1. Deploy using Docker or cloud hosting
2. Configure SSL/HTTPS (required for mobile)
3. Access web UI from mobile browser

```bash
# Example with Docker
docker run -p 7860:7860 bark-infinity:latest
```

Access from mobile: `https://your-server.com:7860`

#### Option 2: Cloud API

Use cloud providers that support Bark:
- Hugging Face Inference API
- Replicate
- AWS SageMaker
- Google Cloud Run

Create a mobile app that calls the API.

#### Option 3: Progressive Web App (PWA)

The Gradio interface can be installed as a PWA:

1. Access the web UI from mobile browser
2. Use "Add to Home Screen" option
3. App will open in fullscreen mode

---

## Cloud Deployment

### AWS

#### EC2 Deployment

```bash
# Launch GPU instance (p3.2xlarge recommended)
# Install Docker
sudo yum update -y
sudo yum install -y docker
sudo service docker start

# Deploy Bark Infinity
git clone https://github.com/MASSIVEMAGNETICS/bark-infinity.git
cd bark-infinity
docker-compose up -d bark-infinity-gpu
```

#### ECS Deployment

Use the provided `Dockerfile` with AWS ECS:

1. Build and push to ECR
2. Create ECS task definition
3. Configure service with GPU support
4. Deploy to cluster

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to GCR
gcloud builds submit --tag gcr.io/PROJECT_ID/bark-infinity

# Deploy to Cloud Run
gcloud run deploy bark-infinity \
  --image gcr.io/PROJECT_ID/bark-infinity \
  --platform managed \
  --region us-central1 \
  --memory 16Gi \
  --cpu 4
```

### Azure

#### Container Instances

```bash
# Create container instance
az container create \
  --resource-group bark-rg \
  --name bark-infinity \
  --image bark-infinity:latest \
  --ports 7860 8501 \
  --memory 16 \
  --cpu 4
```

### Kubernetes

Example deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bark-infinity
spec:
  replicas: 1
  selector:
    matchLabels:
      app: bark-infinity
  template:
    metadata:
      labels:
        app: bark-infinity
    spec:
      containers:
      - name: bark-infinity
        image: bark-infinity:latest
        ports:
        - containerPort: 7860
        - containerPort: 8501
        resources:
          requests:
            memory: "8Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "4"
            nvidia.com/gpu: 1
```

---

## Performance Optimization

### GPU Optimization

```bash
# Use mixed precision
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6"

# Enable TF32
export NVIDIA_TF32_OVERRIDE=1
```

### CPU Optimization

```bash
# Set thread count
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable CPU optimizations
export BARK_CPU_OPTIMIZE=True
```

---

## Troubleshooting

### Out of Memory Errors

1. Enable CPU offloading: `SUNO_OFFLOAD_CPU=True`
2. Use smaller models: `SUNO_USE_SMALL_MODELS=True`
3. Enable quantization: `BARK_QUANTIZE_8BIT=True`
4. Reduce batch size if processing multiple prompts

### Slow Generation

1. Use GPU if available
2. Enable BetterTransformer: `BARK_USE_BETTER_TRANSFORMER=True`
3. Consider using smaller models
4. Ensure CUDA is properly installed (for NVIDIA GPUs)

### Model Download Issues

```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/cache

# Use mirror if needed
export HF_ENDPOINT=https://hf-mirror.com
```

---

## Security Considerations

### Production Deployment

1. **Use HTTPS**: Always use SSL/TLS in production
2. **Authentication**: Add authentication layer (e.g., nginx basic auth)
3. **Rate Limiting**: Prevent abuse with rate limiting
4. **Resource Limits**: Set container resource limits
5. **Monitoring**: Set up logging and monitoring

### Example Nginx Configuration

```nginx
server {
    listen 443 ssl;
    server_name bark.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:7860;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        
        # Basic authentication
        auth_basic "Restricted";
        auth_basic_user_file /etc/nginx/.htpasswd;
        
        # Rate limiting
        limit_req zone=api_limit burst=5;
    }
}
```

---

## Support

- **Issues**: [GitHub Issues](https://github.com/MASSIVEMAGNETICS/bark-infinity/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MASSIVEMAGNETICS/bark-infinity/discussions)
- **Documentation**: [README.md](README.md)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.
