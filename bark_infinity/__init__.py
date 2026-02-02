from importlib import import_module

__version__ = "0.1.0"

_LAZY_ATTRS = {
    "generate_audio": ("bark_infinity.api", "generate_audio"),
    "text_to_semantic": ("bark_infinity.api", "text_to_semantic"),
    "semantic_to_waveform": ("bark_infinity.api", "semantic_to_waveform"),
    "save_as_prompt": ("bark_infinity.api", "save_as_prompt"),
    "generate_audio_long": ("bark_infinity.api", "generate_audio_long"),
    "render_npz_samples": ("bark_infinity.api", "render_npz_samples"),
    "list_speakers": ("bark_infinity.api", "list_speakers"),
    "SAMPLE_RATE": ("bark_infinity.generation", "SAMPLE_RATE"),
    "preload_models": ("bark_infinity.generation", "preload_models"),
    "logger": ("bark_infinity.config", "logger"),
    "console": ("bark_infinity.config", "console"),
    "get_default_values": ("bark_infinity.config", "get_default_values"),
    "load_all_defaults": ("bark_infinity.config", "load_all_defaults"),
    "VALID_HISTORY_PROMPT_DIRS": ("bark_infinity.config", "VALID_HISTORY_PROMPT_DIRS"),
    "FusionTransformer": ("bark_infinity.fusion_transformer", "FusionTransformer"),
    # Windows Runtime
    "get_runtime": ("bark_infinity.windows_runtime", "get_runtime"),
    "shutdown_runtime": ("bark_infinity.windows_runtime", "shutdown_runtime"),
    "MultiThreadedRuntime": ("bark_infinity.windows_runtime", "MultiThreadedRuntime"),
    "GenerationTask": ("bark_infinity.windows_runtime", "GenerationTask"),
    "GenerationResult": ("bark_infinity.windows_runtime", "GenerationResult"),
    "ChunkedGenerator": ("bark_infinity.windows_runtime", "ChunkedGenerator"),
    "LayeredGenerationEngine": ("bark_infinity.windows_runtime", "LayeredGenerationEngine"),
    # Voice Studio
    "get_voice_studio": ("bark_infinity.voice_studio", "get_voice_studio"),
    "VoiceStudio": ("bark_infinity.voice_studio", "VoiceStudio"),
    "VoiceModel": ("bark_infinity.voice_studio", "VoiceModel"),
    "VoiceModelConverter": ("bark_infinity.voice_studio", "VoiceModelConverter"),
    "VoiceQualityAnalyzer": ("bark_infinity.voice_studio", "VoiceQualityAnalyzer"),
}

# Quantization support for low-compute devices
try:
    from .quantization import (
        QuantizationConfig,
        quantize_model,
        setup_low_compute_mode,
        estimate_memory_savings,
    )
    __quantization_available__ = True
except ImportError:
    __quantization_available__ = False


def __getattr__(name):
    if name in _LAZY_ATTRS:
        module_name, attr_name = _LAZY_ATTRS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'bark_infinity' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_ATTRS.keys()))


__all__ = [
    "__version__",
    "__quantization_available__",
    *_LAZY_ATTRS.keys(),
]

if __quantization_available__:
    __all__.extend(
        [
            "QuantizationConfig",
            "quantize_model",
            "setup_low_compute_mode",
            "estimate_memory_savings",
        ]
    )
