# Developer Guide - Windows 10 Runtime & Voice Studio

## Architecture Overview

This document provides technical details for developers working with the Windows 10 Runtime and Voice Clone Studio components.

## Component Structure

```
bark-infinity/
├── bark_infinity/
│   ├── windows_runtime.py      # Multi-threaded runtime engine
│   ├── voice_studio.py          # Voice model management
│   └── __init__.py              # Updated with new exports
├── bark_infinity_studio.py      # Windows 10 GUI application
├── test_windows_runtime.py      # Runtime unit tests
├── test_voice_studio.py         # Studio unit tests
├── examples/
│   ├── windows_runtime_example.py
│   ├── voice_studio_example.py
│   └── complete_workflow.py
├── launch_studio_windows.bat    # Windows launcher
├── launch_studio_windows.ps1    # PowerShell launcher
└── WINDOWS_RUNTIME.md           # User documentation
```

## Core Classes

### Windows Runtime (`windows_runtime.py`)

#### 1. ChunkedGenerator
Handles text chunking for long-form content.

```python
class ChunkedGenerator:
    def __init__(self, chunk_size: int = 1000)
    def split_text(self, text: str) -> List[str]
    def generate_chunk(self, chunk: str, ...) -> np.ndarray
```

**Key Features:**
- Splits text at sentence boundaries
- Respects maximum chunk size
- Maintains voice consistency across chunks

#### 2. LayeredGenerationEngine
Implements the layered generation architecture.

```python
class LayeredGenerationEngine:
    def __init__(self, use_cache: bool = True)
    def generate_semantic_layer(...) -> np.ndarray
    def generate_coarse_layer(...) -> np.ndarray
    def generate_fine_layer(...) -> np.ndarray
    def clear_caches()
```

**Layers:**
1. **Semantic Layer**: Text → semantic tokens
2. **Coarse Layer**: Semantic tokens → coarse audio structure
3. **Fine Layer**: Coarse tokens → fine audio details

**Caching:**
- Semantic, coarse, and fine layers are cached separately
- Cache keys are based on input hash
- Caches can be cleared to free memory

#### 3. MultiThreadedRuntime
Main runtime coordinator for concurrent processing.

```python
class MultiThreadedRuntime:
    def __init__(self, max_workers: Optional[int] = None, 
                 max_queue_size: int = 100)
    def start()
    def stop()
    def submit_task(self, task: GenerationTask) -> str
    def get_result(self, task_id: str, timeout: float) -> GenerationResult
    def batch_generate(self, texts: List[str], ...) -> List[GenerationResult]
```

**Threading Model:**
- Uses ThreadPoolExecutor for worker management
- Priority queue for task scheduling
- Separate worker thread for queue processing
- Thread-safe result storage

**Worker Count:**
- Default: `cpu_count() - 1`
- Configurable via `max_workers` parameter
- Automatically optimizes for CPU-bound workloads

#### 4. Data Structures

**GenerationTask:**
```python
@dataclass
class GenerationTask:
    task_id: str
    text: str
    history_prompt: Optional[Dict] = None
    temp: float = 0.7
    callback: Optional[Callable] = None
    priority: int = 0
    chunk_size: int = 1000
```

**GenerationResult:**
```python
@dataclass
class GenerationResult:
    task_id: str
    audio_array: np.ndarray
    sample_rate: int
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None
```

### Voice Studio (`voice_studio.py`)

#### 1. VoiceModel
Data structure for voice models.

```python
@dataclass
class VoiceModel:
    id: str
    name: str
    format: str
    created_at: str
    semantic_prompt: Optional[np.ndarray] = None
    coarse_prompt: Optional[np.ndarray] = None
    fine_prompt: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    
    def to_dict() -> Dict
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceModel'
```

#### 2. VoiceModelConverter
Handles format conversion between NPZ, PKL, and JSON.

```python
class VoiceModelConverter:
    @staticmethod
    def npz_to_dict(npz_path: str) -> Dict
    @staticmethod
    def dict_to_npz(data: Dict, npz_path: str)
    @staticmethod
    def pkl_to_dict(pkl_path: str) -> Dict
    @staticmethod
    def dict_to_pkl(data: Dict, pkl_path: str)
    @staticmethod
    def json_to_dict(json_path: str) -> Dict
    @staticmethod
    def dict_to_json(data: Dict, json_path: str)
    @staticmethod
    def convert(input_path: str, output_path: str, output_format: str)
```

**Supported Formats:**
- **NPZ**: NumPy compressed format (default)
- **PKL**: Python pickle format
- **JSON**: JavaScript Object Notation (human-readable)

#### 3. VoiceQualityAnalyzer
Analyzes voice model quality using advanced algorithms.

```python
class VoiceQualityAnalyzer:
    @staticmethod
    def analyze_semantic_diversity(semantic_prompt) -> Dict[str, float]
    @staticmethod
    def analyze_coarse_quality(coarse_prompt) -> Dict[str, float]
    @staticmethod
    def analyze_fine_quality(fine_prompt) -> Dict[str, float]
    @staticmethod
    def comprehensive_analysis(voice_model) -> Dict[str, Any]
```

**Quality Metrics:**
- **Semantic Diversity**: Token uniqueness, entropy
- **Coarse Quality**: Spectral consistency, mean/std
- **Fine Detail**: Variance, detail level
- **Overall Quality**: Weighted average of all metrics

#### 4. VoiceStudio
Main voice model management system.

```python
class VoiceStudio:
    def __init__(self, library_path: str = None)
    def import_voice_model(self, file_path, name, transcription) -> str
    def export_voice_model(self, model_id, output_path, output_format)
    def list_models() -> List[Dict]
    def get_model(self, model_id) -> Optional[VoiceModel]
    def delete_model(self, model_id)
    def analyze_model(self, model_id) -> Dict[str, Any]
    def clone_from_audio(self, audio_path, transcription, name) -> str
    def batch_import(self, directory) -> List[str]
    def search_models(self, query) -> List[Dict]
```

**Library Structure:**
```
~/.cache/bark_infinity/voice_library/
├── index.json              # Model index
└── models/                 # Model files
    ├── {model_id_1}.npz
    ├── {model_id_2}.npz
    └── ...
```

## Implementation Details

### Thread Safety

The MultiThreadedRuntime uses several mechanisms for thread safety:

1. **ThreadPoolExecutor**: Manages worker threads
2. **Queue.PriorityQueue**: Thread-safe task queue
3. **Dictionary locks**: Results dictionary accessed atomically
4. **Threading.Event**: For signaling worker thread

### Memory Management

Efficient memory usage is achieved through:

1. **Lazy loading**: Models loaded on demand
2. **Cache management**: Automatic cache cleanup
3. **Garbage collection**: Explicit gc calls after cleanup
4. **Streaming results**: Results delivered progressively
5. **numpy arrays**: Efficient numerical operations

### Error Handling

Comprehensive error handling at multiple levels:

1. **Task level**: Errors caught and stored in GenerationResult
2. **Runtime level**: Exceptions logged but don't crash workers
3. **Studio level**: File operations wrapped in try-catch
4. **GUI level**: User-friendly error messages

### Performance Optimization

Key optimizations for CPU-friendly operation:

1. **Chunked processing**: Breaks large texts into manageable pieces
2. **Smart caching**: Caches intermediate results
3. **Thread pooling**: Reuses threads efficiently
4. **Priority queue**: Processes urgent tasks first
5. **Batch operations**: Amortizes overhead across multiple tasks

## API Usage Examples

### Basic Runtime Usage

```python
from bark_infinity import get_runtime, GenerationTask

# Initialize runtime
runtime = get_runtime(max_workers=4)

# Create task
task = GenerationTask(
    task_id="my_task",
    text="Generate this text",
    temp=0.7,
    priority=10
)

# Submit and get result
task_id = runtime.submit_task(task)
result = runtime.get_result(task_id, timeout=300)

if result.success:
    print(f"Generated {len(result.audio_array)} samples")
```

### Voice Studio Usage

```python
from bark_infinity import get_voice_studio

# Initialize studio
studio = get_voice_studio()

# Import model
model_id = studio.import_voice_model(
    file_path="path/to/voice.npz",
    name="My Voice",
    transcription="Sample text"
)

# Analyze quality
analysis = studio.analyze_model(model_id)
print(f"Quality: {analysis['overall_quality']:.2%}")

# Export to different format
studio.export_voice_model(
    model_id=model_id,
    output_path="exported.json",
    output_format="json"
)
```

### Batch Processing

```python
from bark_infinity import get_runtime

runtime = get_runtime()

texts = ["Text 1", "Text 2", "Text 3"]
results = runtime.batch_generate(texts, temp=0.7)

for i, result in enumerate(results):
    if result.success:
        print(f"Task {i+1}: {result.metadata['duration_seconds']:.2f}s")
```

## Testing

### Running Tests

```bash
# Run all tests
python test_windows_runtime.py
python test_voice_studio.py

# Run specific test class
python -m unittest test_windows_runtime.TestChunkedGenerator

# Run with verbose output
python test_windows_runtime.py -v
```

### Test Coverage

**Runtime Tests:**
- ChunkedGenerator: text splitting, chunk generation
- LayeredGenerationEngine: layer generation, caching
- MultiThreadedRuntime: task submission, result retrieval
- Data structures: task/result creation
- Singleton pattern: instance management
- Integration: full workflow with mocks

**Studio Tests:**
- VoiceModel: serialization, deserialization
- VoiceModelConverter: format conversion (NPZ, PKL, JSON)
- VoiceQualityAnalyzer: quality metrics calculation
- VoiceStudio: import, export, search, delete
- Singleton pattern: instance management
- File operations: temporary directory handling

## Extending the System

### Adding New Voice Model Formats

1. Add converter methods to `VoiceModelConverter`:
```python
@staticmethod
def new_format_to_dict(path: str) -> Dict:
    # Load from new format
    pass

@staticmethod
def dict_to_new_format(data: Dict, path: str):
    # Save to new format
    pass
```

2. Update `convert()` method to handle new format

3. Add tests for new format

### Adding New Quality Metrics

1. Add analysis method to `VoiceQualityAnalyzer`:
```python
@staticmethod
def analyze_new_metric(data: np.ndarray) -> Dict[str, float]:
    # Calculate new metric
    return {"new_metric_score": score}
```

2. Update `comprehensive_analysis()` to include new metric

3. Add tests for new metric

### Customizing the GUI

The GUI in `bark_infinity_studio.py` uses Gradio. To customize:

1. **Add new tabs**: Use `with gr.Tab("Name"):`
2. **Add new components**: Use Gradio components (Textbox, Button, etc.)
3. **Add new functions**: Create handler functions and connect with `.click()`
4. **Customize styling**: Modify CSS in the `css` parameter

## Performance Tuning

### Optimal Worker Count

```python
import multiprocessing

# Conservative (leaves cores free)
workers = max(1, multiprocessing.cpu_count() - 2)

# Aggressive (uses all cores)
workers = multiprocessing.cpu_count()

# Custom
workers = 4  # Fixed number

runtime = get_runtime(max_workers=workers)
```

### Cache Management

```python
# Clear caches to free memory
runtime.layered_engine.clear_caches()

# Disable caching if memory is tight
engine = LayeredGenerationEngine(use_cache=False)
```

### Chunk Size Tuning

```python
# Smaller chunks = more overhead, but better for low memory
generator = ChunkedGenerator(chunk_size=500)

# Larger chunks = less overhead, but higher memory usage
generator = ChunkedGenerator(chunk_size=2000)
```

## Troubleshooting

### Common Issues

**Issue: Out of memory**
- Solution: Reduce worker count, disable caching, use smaller chunks

**Issue: Slow generation**
- Solution: Increase worker count, enable caching, use larger chunks

**Issue: Voice model import fails**
- Solution: Check file format, verify file integrity, check permissions

**Issue: Tests fail due to missing dependencies**
- Solution: Install dependencies: `pip install -e .[dev]`

## Contributing

When contributing to these components:

1. Follow existing code style
2. Add docstrings to all public methods
3. Write tests for new functionality
4. Update documentation
5. Run tests before submitting PR

## License

MIT License - See [LICENSE](LICENSE) for details

## Support

- GitHub Issues: https://github.com/MASSIVEMAGNETICS/bark-infinity/issues
- Documentation: [WINDOWS_RUNTIME.md](WINDOWS_RUNTIME.md)
