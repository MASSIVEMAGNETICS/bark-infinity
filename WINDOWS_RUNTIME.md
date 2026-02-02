# Bark Infinity Studio - Windows 10 Runtime

## Enterprise-Grade Voice Cloning and Audio Generation Platform

### Overview

Bark Infinity Studio is a revolutionary Windows 10 application that brings enterprise-grade voice cloning and audio generation capabilities to your desktop. Built on top of the Bark Infinity framework, it features:

- **Multi-threaded Processing Engine**: Concurrent audio generation with configurable worker threads
- **CPU-Friendly Architecture**: Optimized for efficient CPU usage with chunked processing
- **Layered Generational System**: Progressive audio synthesis through semantic, coarse, and fine layers
- **Professional Voice Studio**: Comprehensive voice model management with import/export capabilities
- **Advanced Quality Analysis**: Cutting-edge algorithms for voice model quality assessment
- **Multi-Format Support**: NPZ, PKL, JSON voice model formats

---

## Features

### 🚀 Multi-Threaded Runtime

The Windows Runtime provides a sophisticated multi-threaded processing engine that:

- Automatically optimizes worker threads based on CPU core count
- Implements task queuing with priority support
- Provides chunked text processing for long-form content
- Includes smart caching for improved performance
- Handles concurrent generation requests efficiently

**Example Usage:**

```python
from bark_infinity import get_runtime, GenerationTask

# Initialize runtime (auto-detects optimal thread count)
runtime = get_runtime()

# Create and submit a task
task = GenerationTask(
    task_id="my_task",
    text="Hello, this is a test of the multi-threaded runtime!",
    temp=0.7
)

task_id = runtime.submit_task(task)

# Get result (blocking)
result = runtime.get_result(task_id, timeout=300)

if result.success:
    print(f"Audio generated: {result.audio_array.shape}")
    print(f"Processing time: {result.metadata['processing_time']:.2f}s")
```

### 🎙️ Voice Clone Studio

A comprehensive voice model management system that allows you to:

- **Import** voice models from multiple formats (NPZ, PKL, JSON, WAV)
- **Export** voice models to any supported format
- **Analyze** voice quality using advanced algorithms
- **Organize** voice models in a centralized library
- **Search** and filter voice models by name
- **Batch process** multiple voice models

**Example Usage:**

```python
from bark_infinity import get_voice_studio

# Initialize studio
studio = get_voice_studio()

# Import a voice model
model_id = studio.import_voice_model(
    file_path="path/to/voice_model.npz",
    name="Professional Voice",
    transcription="Sample transcription text"
)

# Analyze voice quality
analysis = studio.analyze_model(model_id)
print(f"Overall Quality: {analysis['overall_quality']:.2%}")

# Export to different format
studio.export_voice_model(
    model_id=model_id,
    output_path="exported_voice.json",
    output_format="json"
)

# List all models
models = studio.list_models()
for model in models:
    print(f"{model['name']}: {model['id']}")
```

### 🧩 Chunked Generation

For long-form content, the chunked generator automatically:

- Splits text at sentence boundaries
- Processes each chunk independently
- Maintains voice consistency across chunks
- Concatenates results seamlessly

**Example Usage:**

```python
from bark_infinity.windows_runtime import ChunkedGenerator

generator = ChunkedGenerator(chunk_size=1000)

# Split long text
chunks = generator.split_text(long_text)
print(f"Split into {len(chunks)} chunks")

# Generate audio for each chunk
for i, chunk in enumerate(chunks):
    audio = generator.generate_chunk(chunk, temp=0.7)
    print(f"Chunk {i+1}: {len(audio)} samples")
```

### 📊 Quality Analysis

Advanced algorithms analyze voice models across multiple dimensions:

- **Semantic Diversity**: Token uniqueness and entropy
- **Coarse Quality**: Spectral consistency and mean/std analysis
- **Fine Detail**: Variance and detail level assessment
- **Overall Score**: Comprehensive quality metric

---

## Windows 10 Application

### Quick Start

#### Option 1: Using Batch File (Easiest)

1. Double-click `launch_studio_windows.bat`
2. The application will start automatically
3. Your browser will open to the Studio interface

#### Option 2: Using PowerShell

1. Right-click `launch_studio_windows.ps1`
2. Select "Run with PowerShell"
3. The application will start with colored output

#### Option 3: Using Python Directly

```bash
python bark_infinity_studio.py
```

#### Option 4: Using Command Line

```bash
bark-studio
```

### Command Line Options

```bash
python bark_infinity_studio.py --help

Options:
  --port PORT              Port to run server on (default: 7860)
  --share                  Create public shareable link
  --server-name NAME       Server name/IP to bind to (default: 127.0.0.1)
```

**Examples:**

```bash
# Run on custom port
python bark_infinity_studio.py --port 8080

# Create public link
python bark_infinity_studio.py --share

# Run on all interfaces
python bark_infinity_studio.py --server-name 0.0.0.0
```

---

## GUI Features

### 🎵 Audio Generation Tab

- **Text Input**: Enter text to convert to speech
- **Voice Selection**: Choose from imported voice models
- **Temperature Control**: Adjust creativity (0.1-1.0)
- **Real-time Progress**: See generation progress
- **Audio Preview**: Play generated audio immediately
- **Generation Info**: View detailed statistics

### 🎙️ Voice Studio Tab

- **Import Models**: Drag and drop voice model files
- **Export Models**: Convert to different formats
- **Model Library**: Browse all imported models
- **Quick Search**: Find models by name
- **Model Metadata**: View creation date, format, etc.

### 📊 Voice Analysis Tab

- **Quality Metrics**: Comprehensive analysis report
- **Layer Statistics**: Semantic, coarse, and fine layer details
- **Overall Score**: Single quality metric
- **Detailed Breakdown**: Entropy, diversity, variance metrics

### ⚙️ Management Tab

- **Delete Models**: Remove unwanted voice models
- **System Info**: View platform and version details
- **Feature List**: See all enabled features

---

## Architecture

### Multi-Threaded Design

```
┌─────────────────────────────────────────┐
│     Application Layer (Gradio UI)      │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│    MultiThreadedRuntime (Coordinator)    │
│  • Task Queue Management                │
│  • Worker Thread Pool                   │
│  • Result Aggregation                   │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼───┐  ┌──▼───┐  ┌──▼───┐
│Worker │  │Worker│  │Worker│
│Thread │  │Thread│  │Thread│
│  #1   │  │  #2  │  │  #N  │
└───┬───┘  └──┬───┘  └──┬───┘
    │         │         │
    └────────┬┴─────────┘
             │
┌────────────▼──────────────────────────┐
│    LayeredGenerationEngine            │
│  • Semantic Layer (Tokens)           │
│  • Coarse Layer (Audio Structure)    │
│  • Fine Layer (Detail)               │
└───────────────────────────────────────┘
```

### Voice Studio Architecture

```
┌─────────────────────────────────────────┐
│         VoiceStudio (Manager)           │
│  • Model Index Database                │
│  • File System Management              │
│  • Search & Filter                     │
└──────────────┬──────────────────────────┘
               │
    ┌──────────┼──────────┐
    │          │          │
┌───▼────────┐ │ ┌────────▼─────┐
│  Converter │ │ │   Analyzer   │
│  • NPZ     │ │ │  • Semantic  │
│  • PKL     │ │ │  • Coarse    │
│  • JSON    │ │ │  • Fine      │
└────────────┘ │ └──────────────┘
               │
        ┌──────▼──────┐
        │Voice Library│
        │ (File Store)│
        └─────────────┘
```

---

## Performance Optimization

### CPU-Friendly Features

1. **Automatic Thread Management**: Uses `cpu_count() - 1` threads
2. **Chunked Processing**: Splits large texts into manageable pieces
3. **Smart Caching**: Caches semantic, coarse, and fine layers
4. **Priority Queue**: Processes high-priority tasks first
5. **Resource Cleanup**: Automatic garbage collection and cache clearing

### Memory Management

- Efficient numpy array handling
- Streaming audio generation
- Progressive result delivery
- Automatic model offloading

### Best Practices

```python
# Initialize runtime once and reuse
runtime = get_runtime(max_workers=4)

# Use batch processing for multiple texts
results = runtime.batch_generate(
    texts=["Text 1", "Text 2", "Text 3"],
    temp=0.7
)

# Clean up when done
shutdown_runtime()
```

---

## Advanced Usage

### Custom Task Priority

```python
task = GenerationTask(
    task_id="urgent_task",
    text="This needs to be processed first!",
    priority=10,  # Higher priority
    temp=0.7
)
```

### Batch Voice Model Import

```python
studio = get_voice_studio()

# Import all models from directory
model_ids = studio.batch_import("path/to/voice_models/")
print(f"Imported {len(model_ids)} models")
```

### Voice Cloning from Audio

```python
studio = get_voice_studio()

# Clone voice from audio file
model_id = studio.clone_from_audio(
    audio_path="path/to/audio.wav",
    transcription="The text spoken in the audio",
    name="My Cloned Voice"
)

print(f"Created voice model: {model_id}")
```

### Custom Analysis

```python
from bark_infinity.voice_studio import VoiceQualityAnalyzer

analyzer = VoiceQualityAnalyzer()

# Analyze specific layer
semantic_analysis = analyzer.analyze_semantic_diversity(
    voice_model.semantic_prompt
)

print(f"Diversity Score: {semantic_analysis['diversity_score']:.2%}")
print(f"Entropy: {semantic_analysis['entropy']:.2f}")
```

---

## System Requirements

### Minimum Requirements

- **OS**: Windows 10 (64-bit)
- **CPU**: Intel Core i5 or AMD equivalent (4+ cores recommended)
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 15 GB free space (for models and cache)
- **Python**: 3.8 or higher

### Recommended Requirements

- **OS**: Windows 10/11 (64-bit)
- **CPU**: Intel Core i7 or AMD Ryzen 7 (8+ cores)
- **RAM**: 16 GB or more
- **GPU**: NVIDIA GPU with CUDA support (optional, for acceleration)
- **Storage**: SSD with 20+ GB free space
- **Python**: 3.10 or 3.11

---

## Installation

### From PyPI

```bash
pip install bark-infinity
```

### From Source

```bash
git clone https://github.com/MASSIVEMAGNETICS/bark-infinity.git
cd bark-infinity
pip install -e .
```

### With All Features

```bash
pip install bark-infinity[all]
```

---

## Troubleshooting

### Issue: Application won't start

**Solution**: Ensure Python 3.8+ is installed and in PATH

```bash
python --version
# Should show Python 3.8 or higher
```

### Issue: Out of memory errors

**Solution**: Reduce worker threads or enable quantization

```python
runtime = get_runtime(max_workers=2)
```

### Issue: Slow generation

**Solution**: Use GPU acceleration if available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
```

### Issue: Voice model import fails

**Solution**: Check file format and integrity

```bash
# Verify file is valid NPZ
python -c "import numpy as np; data = np.load('model.npz'); print(data.files)"
```

---

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - See [LICENSE](LICENSE) for details

---

## Support

- **GitHub Issues**: https://github.com/MASSIVEMAGNETICS/bark-infinity/issues
- **Documentation**: https://github.com/MASSIVEMAGNETICS/bark-infinity#readme
- **Discussions**: https://github.com/MASSIVEMAGNETICS/bark-infinity/discussions

---

## Credits

Built on top of:
- Bark by Suno AI
- Gradio for UI
- PyTorch for deep learning
- NumPy for numerical computing

**Developed by MASSIVE MAGNETICS**

---

## Version History

### v1.0.0 - Windows 10 Runtime Release

- ✅ Multi-threaded processing engine
- ✅ CPU-optimized chunked generation
- ✅ Layered generational architecture
- ✅ Comprehensive Voice Clone Studio
- ✅ Multi-format import/export (NPZ, PKL, JSON)
- ✅ Advanced quality analysis algorithms
- ✅ Professional voice model library
- ✅ Windows 10 optimized application
- ✅ Gradio-based GUI
- ✅ Batch processing support
- ✅ Priority task queuing
- ✅ Smart caching system
