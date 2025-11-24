# Bark Infinity - Examples

This directory contains example scripts demonstrating various features of Bark Infinity.

## Available Examples

### 1. Basic Usage (`basic_usage.py`)

Demonstrates fundamental features:
- Simple audio generation
- Using different voice presets
- Saving audio files
- Error handling

**Run:**
```bash
python examples/basic_usage.py
```

**Prerequisites:** 
- Bark Infinity installed: `pip install bark-infinity`
- Models will download on first run (~12GB)

---

### 2. Low-Compute Mode (`low_compute_mode.py`)

Shows how to use Bark Infinity on resource-constrained devices:
- Enabling low-compute optimizations
- Quantization configuration
- Memory estimation
- Environment variable setup

**Run:**
```bash
python examples/low_compute_mode.py
```

**Prerequisites:**
- Bark Infinity with quantization: `pip install bark-infinity[quantization]`

---

## Quick Start

```bash
# Clone repository
git clone https://github.com/MASSIVEMAGNETICS/bark-infinity.git
cd bark-infinity

# Install with examples dependencies
pip install -e .[dev]

# Run basic example
python examples/basic_usage.py

# Run low-compute example
python examples/low_compute_mode.py
```

## Example Output

All examples save audio files to the current directory:
- `example_basic.wav` - Basic generation
- `example_voice.wav` - With voice preset
- `example_clip_*.wav` - Multiple clips
- `low_compute_example.wav` - Low-compute mode

## Common Issues

### Models Not Downloaded

**Symptom:** First run takes a long time
**Solution:** This is normal. Models (~12GB) are downloading. Subsequent runs will be faster.

### Out of Memory

**Symptom:** CUDA out of memory error
**Solution:** Use low-compute mode example or enable CPU offloading:
```python
import os
os.environ['SUNO_OFFLOAD_CPU'] = 'True'
```

### Missing Dependencies

**Symptom:** ImportError for torch, transformers, etc.
**Solution:** Install full dependencies:
```bash
pip install bark-infinity[all]
```

## More Examples

Looking for more examples?

- **Web UI**: Run `python bark_webui.py` for interactive interface
- **CLI**: Use `bark-infinity generate "text"` for command-line generation
- **Deployment**: See [DEPLOYMENT.md](../DEPLOYMENT.md) for production examples
- **API**: See [QUICKSTART.md](../QUICKSTART.md) for API examples

## Contributing Examples

Have a cool example? Contributions welcome!

1. Create a new example script
2. Add documentation to this README
3. Submit a pull request

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

## Support

- **Documentation**: See main [README.md](../README.md)
- **Issues**: [GitHub Issues](https://github.com/MASSIVEMAGNETICS/bark-infinity/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MASSIVEMAGNETICS/bark-infinity/discussions)
