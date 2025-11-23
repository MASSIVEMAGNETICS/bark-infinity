# Changelog

All notable changes to Bark Infinity will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-11-23

### Added

#### Build & Packaging Infrastructure
- **Cross-Platform Build System**
  - Enhanced `pyproject.toml` with comprehensive metadata and dependencies
  - Added PyInstaller configuration for Windows executable builds
  - Created multi-stage Dockerfile for CPU, GPU, and production deployments
  - Added docker-compose.yml for easy container orchestration
  
#### Quantization & Low-Compute Support
- **New Quantization Module** (`bark_infinity/quantization.py`)
  - Support for 8-bit and 4-bit model quantization using bitsandbytes
  - BetterTransformer optimization support via Optimum
  - Automatic device detection (CUDA, MPS, CPU)
  - Environment variable configuration (`BARK_QUANTIZE_8BIT`, `BARK_QUANTIZE_4BIT`)
  - Memory usage estimation utilities
  - Low-compute preset configurations
  
- **Environment Configuration**
  - New environment variables for quantization control
  - Auto-detection of optimal settings based on hardware
  - Memory-efficient model loading strategies

#### Deployment Options
- **Windows Executable**
  - PyInstaller spec file for standalone EXE builds
  - All dependencies bundled in single executable
  - No Python installation required for end users
  
- **Docker Containers**
  - CPU-optimized image for low-resource deployment
  - GPU-accelerated image with CUDA support
  - Production image with auto-detection
  - Health checks and proper resource limits
  
- **Web Deployment**
  - Docker deployment ready for cloud platforms
  - Configuration for Hugging Face Spaces
  - Streamlit Cloud deployment support
  - Gradio web UI enhancements

#### CI/CD & Automation
- **GitHub Actions Workflows**
  - Multi-platform testing (Ubuntu, Windows, macOS)
  - Python version matrix (3.8, 3.9, 3.10, 3.11)
  - Automated Windows EXE builds
  - Docker image building and testing
  - PyPI publishing on release
  
- **Build Scripts**
  - `scripts/build_windows.sh` - Windows executable builder
  - `scripts/build_docker.sh` - Docker image builder
  - `scripts/release.sh` - Release artifact creator

#### Documentation
- **Comprehensive Deployment Guide** (`DEPLOYMENT.md`)
  - Windows executable deployment
  - Docker deployment (CPU/GPU)
  - Web deployment options
  - Low-compute mode configuration
  - Mobile access strategies
  - Cloud deployment guides (AWS, GCP, Azure)
  - Performance optimization tips
  - Security best practices
  
- **Enhanced README**
  - Installation methods
  - Quick start guides
  - Feature documentation
  
- **This Changelog**
  - Version tracking
  - Change documentation

#### Optional Dependencies
- New optional dependency groups in `pyproject.toml`:
  - `[quantization]` - For model quantization support
  - `[build]` - For building executables
  - `[web]` - For web deployment
  - `[dev]` - For development tools
  - `[all]` - Install everything

### Changed

- **Updated pyproject.toml**
  - Version bumped from 0.0.3 to 0.1.0
  - Added comprehensive project metadata
  - Explicit dependency versions with constraints
  - Better package structure definition
  - Added project classifiers
  - Added entry points for CLI tools
  
- **Enhanced bark_infinity/__init__.py**
  - Exposed quantization utilities
  - Added version string
  - Conditional imports for optional features

### Technical Details

#### Memory Optimization
- **Quantization Support**:
  - 8-bit quantization: ~50% memory reduction
  - 4-bit quantization: ~75% memory reduction
  - CPU offloading for models not in use
  - Small model variants for faster inference
  
- **Memory Requirements**:
  - Full model (GPU): ~12GB VRAM
  - Small models (GPU): ~8GB VRAM
  - 8-bit quantized: ~6GB VRAM
  - 4-bit quantized: ~3GB VRAM
  - CPU offloading: ~2GB VRAM + 8GB RAM
  - CPU only: ~16GB RAM

#### Platform Support
- **Windows**: Standalone executable (no Python needed)
- **Linux**: Native, Docker, or PyPI installation
- **macOS**: Native or Docker installation (MPS support for Apple Silicon)
- **Web**: Browser-based access via Gradio/Streamlit
- **Cloud**: AWS, GCP, Azure deployment ready

#### Mobile Strategy
Since mobile devices cannot run the full model directly:
- Web-based mobile interface
- Remote server access via browser
- Progressive Web App (PWA) support
- API server for mobile apps to call

### Known Limitations

- **Mobile Native Apps**: Not supported due to model size and compute requirements
  - iOS: Cannot package 12GB model + PyTorch
  - Android: APK size limits and memory constraints
  - Solution: Use web interface or remote API
  
- **4-bit Quantization**: Experimental, may affect quality
- **Windows EXE**: First run downloads ~12GB of models
- **Docker GPU**: Requires NVIDIA Docker runtime

### Dependencies

#### Core Requirements
- Python 3.8 - 3.12
- PyTorch 2.0+
- Transformers 4.30+
- Gradio 3.35+
- Streamlit 1.22+

#### Optional Requirements
- bitsandbytes 0.41+ (for quantization)
- optimum 1.12+ (for optimization)
- PyInstaller 5.13+ (for Windows build)

### Migration Guide

If upgrading from 0.0.3:

1. **Update installation**:
   ```bash
   pip install --upgrade bark-infinity
   ```

2. **Optional: Enable quantization**:
   ```bash
   pip install bark-infinity[quantization]
   export BARK_QUANTIZE_8BIT=True
   ```

3. **Optional: Build Windows EXE**:
   ```bash
   pip install bark-infinity[build]
   bash scripts/build_windows.sh
   ```

No breaking changes to existing API or functionality.

---

## [0.0.3] - Previous Release

Previous version with basic functionality:
- Command-line interface
- Gradio web UI
- Streamlit interface
- Basic model loading
- CPU offloading support

---

## Future Plans

### Planned for 0.2.0
- [ ] FastAPI REST API server
- [ ] OpenAPI documentation
- [ ] WebSocket support for streaming
- [ ] Model caching improvements
- [ ] Multi-GPU support
- [ ] Batch processing optimization

### Planned for 0.3.0
- [ ] Voice cloning improvements
- [ ] Fine-tuning support
- [ ] Custom model training
- [ ] Advanced audio effects
- [ ] Real-time generation

### Under Consideration
- [ ] ONNX export for wider deployment
- [ ] WebAssembly build (if models can be reduced)
- [ ] Electron desktop app
- [ ] Plugin system for extensions

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to contribute to this project.

## License

MIT License - See [LICENSE](LICENSE) for details.
