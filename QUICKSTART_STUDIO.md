# Quick Start Guide - Windows 10 Studio

## Installation

### Option 1: Install from PyPI (Recommended)

```bash
pip install bark-infinity
```

### Option 2: Install from Source

```bash
git clone https://github.com/MASSIVEMAGNETICS/bark-infinity.git
cd bark-infinity
pip install -e .
```

## First Time Setup

1. **Verify Installation**
```bash
python -c "from bark_infinity import get_runtime, get_voice_studio; print('✓ Installation successful!')"
```

2. **Launch the Studio**

Choose one of these methods:

**Method A: Python Script**
```bash
python bark_infinity_studio.py
```

**Method B: Windows Batch File**
```bash
launch_studio_windows.bat
```

**Method C: PowerShell**
```powershell
.\launch_studio_windows.ps1
```

**Method D: Command Line Tool**
```bash
bark-studio
```

3. **Open in Browser**

The studio will automatically open at http://localhost:7860

If it doesn't open automatically, manually navigate to http://localhost:7860 in your browser.

## Basic Usage

### 1. Generate Audio

1. Go to the **"🎵 Audio Generation"** tab
2. Enter text in the "Text to Generate" box
3. (Optional) Select a voice model from the dropdown
4. Adjust temperature (0.1-1.0) for creativity
5. Click **"🎵 Generate Audio"**
6. Listen to the result and download the audio file

### 2. Import Voice Model

1. Go to the **"🎙️ Voice Studio"** tab
2. Under "Import Voice Model":
   - Click "Voice Model File" and select your .npz, .pkl, or .json file
   - Enter a name for the model
   - (Optional) Add transcription text
3. Click **"📥 Import Model"**
4. Your model is now available in the voice library!

### 3. Export Voice Model

1. Go to the **"🎙️ Voice Studio"** tab
2. Under "Export Voice Model":
   - Select a model from the dropdown
   - Choose export format (npz, pkl, or json)
3. Click **"📤 Export Model"**
4. Download the exported file

### 4. Analyze Voice Quality

1. Go to the **"📊 Voice Analysis"** tab
2. Select a model to analyze
3. Click **"🔬 Analyze Model"**
4. View detailed quality metrics:
   - Overall Quality Score
   - Semantic Diversity
   - Coarse Quality
   - Fine Detail

## Advanced Features

### Multi-Threaded Generation

The runtime automatically uses multiple CPU cores:
- Default: CPU count - 1 cores
- Processes multiple chunks in parallel
- Automatic task queue management

### Batch Processing

Generate multiple audio files at once:

```python
from bark_infinity import get_runtime

runtime = get_runtime()
texts = ["Text 1", "Text 2", "Text 3"]
results = runtime.batch_generate(texts, temp=0.7)

for i, result in enumerate(results):
    print(f"Generated audio {i+1}: {len(result.audio_array)} samples")
```

### Voice Model Formats

Supported formats:
- **NPZ**: NumPy compressed (best for Python)
- **PKL**: Python pickle (compact)
- **JSON**: Human-readable (easy to inspect)

Convert between formats easily in the GUI!

## Command Line Usage

### Generate Audio

```bash
bark-infinity generate "Your text here"
```

### Launch Web UI

```bash
bark-webui
# or
bark-streamlit
```

### Launch Studio

```bash
bark-studio
```

### Custom Port

```bash
bark-studio --port 8080
```

### Create Public Link

```bash
bark-studio --share
```

## Examples

See the `examples/` directory for:

1. **windows_runtime_example.py** - Multi-threaded generation
2. **voice_studio_example.py** - Voice model management
3. **complete_workflow.py** - Full enterprise workflow

Run an example:
```bash
python examples/windows_runtime_example.py
```

## Troubleshooting

### Issue: "No module named 'bark_infinity'"

**Solution:**
```bash
pip install bark-infinity
# or if installed from source:
pip install -e .
```

### Issue: Port 7860 already in use

**Solution:** Use a different port
```bash
bark-studio --port 8080
```

### Issue: Out of memory

**Solution:** Reduce worker threads
```python
from bark_infinity import get_runtime
runtime = get_runtime(max_workers=2)
```

### Issue: Slow generation

**Solution:** 
- Increase worker threads (if you have more CPU cores)
- Enable GPU if available
- Use smaller chunks

## Configuration

### Customize Worker Count

```python
from bark_infinity import get_runtime

# Use 4 workers
runtime = get_runtime(max_workers=4)
```

### Customize Chunk Size

```python
from bark_infinity.windows_runtime import ChunkedGenerator

# Use smaller chunks (more memory efficient)
generator = ChunkedGenerator(chunk_size=500)
```

### Disable Caching

```python
from bark_infinity.windows_runtime import LayeredGenerationEngine

# Disable caching to save memory
engine = LayeredGenerationEngine(use_cache=False)
```

## Performance Tips

1. **Use appropriate worker count**
   - More workers = faster but more memory
   - Recommended: CPU cores - 1

2. **Enable caching for repeated generations**
   - Cache speeds up similar texts
   - Clear cache periodically to free memory

3. **Use batch processing for multiple texts**
   - More efficient than sequential processing
   - Automatically optimized

4. **Choose appropriate chunk size**
   - Smaller chunks: lower memory, more overhead
   - Larger chunks: higher memory, less overhead

## Next Steps

- Read the complete [WINDOWS_RUNTIME.md](WINDOWS_RUNTIME.md) guide
- Check out the [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md) for technical details
- Explore the `examples/` directory
- Join the community on GitHub

## Support

- **Issues**: https://github.com/MASSIVEMAGNETICS/bark-infinity/issues
- **Discussions**: https://github.com/MASSIVEMAGNETICS/bark-infinity/discussions
- **Documentation**: https://github.com/MASSIVEMAGNETICS/bark-infinity#readme

## Quick Reference

### Common Commands

```bash
# Install
pip install bark-infinity

# Launch studio
bark-studio

# Custom port
bark-studio --port 8080

# Public link
bark-studio --share

# Help
bark-studio --help
```

### Python API

```python
# Import modules
from bark_infinity import get_runtime, get_voice_studio

# Initialize
runtime = get_runtime()
studio = get_voice_studio()

# Generate audio
from bark_infinity.windows_runtime import GenerationTask
task = GenerationTask(task_id="test", text="Hello world", temp=0.7)
task_id = runtime.submit_task(task)
result = runtime.get_result(task_id)

# Import voice model
model_id = studio.import_voice_model("voice.npz", name="My Voice")

# Analyze quality
analysis = studio.analyze_model(model_id)
print(f"Quality: {analysis['overall_quality']:.2%}")
```

---

**Enjoy using Bark Infinity Studio!** 🎙️✨
