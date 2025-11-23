#!/usr/bin/env python3
"""
Example: Low-compute mode for resource-constrained devices

This example demonstrates:
1. Enabling low-compute mode
2. Checking quantization availability
3. Memory estimation
4. Generating audio with optimizations
"""

import os
import sys
from scipy.io.wavfile import write as write_wav

# Import Bark Infinity
try:
    from bark_infinity import generate_audio, SAMPLE_RATE
    from bark_infinity.quantization import (
        setup_low_compute_mode,
        QuantizationConfig,
        estimate_memory_savings,
    )
except ImportError as e:
    print("Error: bark-infinity not properly installed")
    print("Install with: pip install bark-infinity[quantization]")
    sys.exit(1)

def check_system():
    """Check system capabilities."""
    print("System Check")
    print("-" * 50)
    
    import torch
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    print()

def show_memory_estimates():
    """Show memory reduction estimates."""
    print("Memory Reduction Estimates")
    print("-" * 50)
    
    base_memory = 12.0  # Bark's base requirement
    
    # No quantization
    print(f"Full model: {base_memory}GB VRAM")
    
    # 8-bit quantization
    savings_8bit = estimate_memory_savings(base_memory, enable_8bit=True)
    print(f"8-bit quantized: {savings_8bit['estimated_memory_gb']:.1f}GB "
          f"(saves {savings_8bit['reduction_percent']:.0f}%)")
    
    # 4-bit quantization
    savings_4bit = estimate_memory_savings(base_memory, enable_4bit=True)
    print(f"4-bit quantized: {savings_4bit['estimated_memory_gb']:.1f}GB "
          f"(saves {savings_4bit['reduction_percent']:.0f}%)")
    
    print()

def example_low_compute_mode():
    """Use low-compute mode for generation."""
    print("Low-Compute Mode Generation")
    print("-" * 50)
    
    # Enable low-compute mode
    print("Configuring low-compute mode...")
    config = setup_low_compute_mode()
    
    # Generate audio
    text = "This audio was generated in low-compute mode."
    print(f"Generating: {text}")
    
    audio_array = generate_audio(text)
    
    # Save
    output_file = "low_compute_example.wav"
    write_wav(output_file, SAMPLE_RATE, audio_array)
    print(f"✓ Audio saved to: {output_file}")
    print()

def example_custom_quantization():
    """Use custom quantization configuration."""
    print("Custom Quantization Configuration")
    print("-" * 50)
    
    # Create custom config
    config = QuantizationConfig(
        enable_8bit=True,
        use_better_transformer=True,
    )
    
    print(f"Device: {config.device}")
    print(f"8-bit quantization: {config.enable_8bit}")
    print(f"BetterTransformer: {config.use_better_transformer}")
    
    # Note: Actual model loading with quantization would happen here
    # For this example, we just show the configuration
    
    print()

def example_environment_config():
    """Configure via environment variables."""
    print("Environment Variable Configuration")
    print("-" * 50)
    
    # Set environment variables
    os.environ['SUNO_OFFLOAD_CPU'] = 'True'
    os.environ['SUNO_USE_SMALL_MODELS'] = 'True'
    os.environ['BARK_QUANTIZE_8BIT'] = 'True'
    
    print("Set environment variables:")
    print("  SUNO_OFFLOAD_CPU=True")
    print("  SUNO_USE_SMALL_MODELS=True")
    print("  BARK_QUANTIZE_8BIT=True")
    
    # Create config from environment
    config = QuantizationConfig.from_env()
    print(f"\nResulting configuration:")
    print(f"  8-bit: {config.enable_8bit}")
    print(f"  Device: {config.device}")
    
    print()

def main():
    """Run all low-compute examples."""
    print("=" * 60)
    print("Bark Infinity - Low-Compute Mode Examples")
    print("=" * 60)
    print()
    
    try:
        check_system()
        show_memory_estimates()
        example_environment_config()
        example_custom_quantization()
        
        # Only run actual generation if models are available
        print("Note: Actual audio generation requires model download")
        print("Run 'example_low_compute_mode()' separately if models are ready")
        print()
        
        print("=" * 60)
        print("Examples completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
