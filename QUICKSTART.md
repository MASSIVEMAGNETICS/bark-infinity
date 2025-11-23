# Bark Infinity - Quick Reference Guide

## Installation

```bash
# Standard installation
pip install bark-infinity

# With quantization support
pip install bark-infinity[quantization]

# With all optional dependencies
pip install bark-infinity[all]
```

## Command Line Usage

```bash
# Show help
bark-infinity --help

# Generate audio
bark-infinity generate "Hello, world!" -o output.wav

# Generate with low-compute mode
bark-infinity generate "Hello!" --low-compute

# Start web UI
bark-infinity webui

# Start Streamlit UI
bark-infinity streamlit

# Show system info
bark-infinity info
```

## Python API

### Basic Usage

```python
import bark_infinity

# Generate audio
audio = bark_infinity.generate_audio("Hello, world!")

# Save audio
from scipy.io.wavfile import write as write_wav
write_wav("output.wav", bark_infinity.SAMPLE_RATE, audio)
```

### Low-Compute Mode

```python
from bark_infinity import setup_low_compute_mode, generate_audio

# Configure low-compute mode
config = setup_low_compute_mode()

# Generate audio
audio = generate_audio("Hello from low-compute mode!")
```

### Quantization Configuration

```python
from bark_infinity.quantization import QuantizationConfig

# 8-bit quantization
config = QuantizationConfig(enable_8bit=True)

# 4-bit quantization
config = QuantizationConfig(enable_4bit=True)

# Custom configuration
config = QuantizationConfig(
    enable_8bit=True,
    use_better_transformer=True,
    device='cuda'
)
```

## Environment Variables

```bash
# Enable CPU offloading
export SUNO_OFFLOAD_CPU=True

# Use smaller models
export SUNO_USE_SMALL_MODELS=True

# Enable 8-bit quantization
export BARK_QUANTIZE_8BIT=True

# Enable 4-bit quantization
export BARK_QUANTIZE_4BIT=True

# Enable BetterTransformer
export BARK_USE_BETTER_TRANSFORMER=True
```

## Docker Deployment

```bash
# CPU deployment
docker-compose up bark-infinity-cpu

# GPU deployment
docker-compose up bark-infinity-gpu

# Manual run
docker run -p 7860:7860 bark-infinity:latest
```

## Building Executables

### Windows EXE

```bash
# Install build dependencies
pip install bark-infinity[build]

# Build executable
bash scripts/build_windows.sh

# Or use PyInstaller directly
pyinstaller bark-infinity.spec
```

### Docker Image

```bash
# Build production image
bash scripts/build_docker.sh production latest

# Or use docker build directly
docker build -t bark-infinity:latest .
```

## Memory Requirements

| Configuration | VRAM/RAM | Speed | Quality |
|---------------|----------|-------|---------|
| Full (GPU) | ~12GB VRAM | Fast | Best |
| Small Models | ~8GB VRAM | Medium | Good |
| 8-bit Quantized | ~6GB VRAM | Medium | Good |
| 4-bit Quantized | ~3GB VRAM | Slow | Fair |
| CPU Offload | ~2GB VRAM + 8GB RAM | Slow | Best |
| CPU Only | ~16GB RAM | Very Slow | Best |

## Common Issues

### Out of Memory

```python
# Solution 1: Enable CPU offloading
import os
os.environ["SUNO_OFFLOAD_CPU"] = "True"

# Solution 2: Use smaller models
os.environ["SUNO_USE_SMALL_MODELS"] = "True"

# Solution 3: Enable quantization
os.environ["BARK_QUANTIZE_8BIT"] = "True"
```

### Slow Generation

```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Enable optimizations
export BARK_USE_BETTER_TRANSFORMER=True
```

### Model Download Issues

```bash
# Set cache directory
export HF_HOME=/path/to/cache

# Use mirror (if needed)
export HF_ENDPOINT=https://hf-mirror.com
```

## File Locations

- **Models**: `~/.cache/suno/bark_v0/`
- **Generated Audio**: `bark_samples/` (default)
- **Voice Prompts**: `bark_infinity/assets/prompts/`
- **Logs**: Current directory (*.log)

## Useful Links

- **Documentation**: [DEPLOYMENT.md](DEPLOYMENT.md)
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)
- **GitHub**: https://github.com/MASSIVEMAGNETICS/bark-infinity
- **Issues**: https://github.com/MASSIVEMAGNETICS/bark-infinity/issues

## Support

For help:
1. Check [DEPLOYMENT.md](DEPLOYMENT.md) for detailed guides
2. Run `bark-infinity info` to check your setup
3. Review [GitHub Issues](https://github.com/MASSIVEMAGNETICS/bark-infinity/issues)
4. Open a new issue if needed

## License

MIT License - See [LICENSE](LICENSE) for details.
