#!/usr/bin/env python3
"""
Test script for Bark Infinity quantization module.
Tests the quantization functionality without requiring full model installation.
"""

import sys
import os

# Add current directory to path for testing
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_quantization_config():
    """Test QuantizationConfig class."""
    print("Testing QuantizationConfig...")
    
    # Mock torch module for testing
    import types
    torch_mock = types.ModuleType('torch')
    torch_mock.cuda = types.ModuleType('cuda')
    torch_mock.cuda.is_available = lambda: False
    torch_mock.backends = types.ModuleType('backends')
    torch_mock.nn = types.ModuleType('nn')
    torch_mock.nn.Module = object
    sys.modules['torch'] = torch_mock
    
    from bark_infinity.quantization import QuantizationConfig, estimate_memory_savings
    
    # Test default config
    config = QuantizationConfig()
    assert config.device in ['cuda', 'cpu', 'mps'], f"Invalid device: {config.device}"
    print(f"✓ Default device: {config.device}")
    
    # Test explicit config
    config = QuantizationConfig(enable_8bit=True, device='cpu')
    # 8-bit should be disabled on CPU because bitsandbytes requires CUDA
    assert config.enable_8bit == False, "8-bit should be disabled on CPU (no CUDA)"
    print("✓ 8-bit correctly disabled on CPU")
    
    # Test from environment
    os.environ['BARK_QUANTIZE_8BIT'] = 'true'
    config = QuantizationConfig.from_env()
    print(f"✓ Config from environment: 8bit={config.enable_8bit}")
    
    # Test low compute preset
    config = QuantizationConfig.low_compute_preset()
    print(f"✓ Low compute preset: device={config.device}")
    
    # Test memory estimation
    savings = estimate_memory_savings(12.0, enable_8bit=True)
    assert savings['base_memory_gb'] == 12.0
    assert savings['estimated_memory_gb'] == 6.0
    assert savings['savings_gb'] == 6.0
    assert savings['reduction_percent'] == 50.0
    print("✓ Memory estimation (8-bit): 12GB -> 6GB (50% reduction)")
    
    savings = estimate_memory_savings(12.0, enable_4bit=True)
    assert savings['estimated_memory_gb'] == 3.0
    assert savings['reduction_percent'] == 75.0
    print("✓ Memory estimation (4-bit): 12GB -> 3GB (75% reduction)")
    
    print("\n✅ All quantization tests passed!\n")

def test_imports():
    """Test that quantization module can be imported."""
    print("Testing imports...")
    
    # This will fail if torch is not installed, which is expected
    try:
        from bark_infinity import quantization
        print("✓ Quantization module imported")
    except ImportError as e:
        if 'torch' in str(e):
            print("⚠ PyTorch not installed (expected in test environment)")
            return False
        raise
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Bark Infinity - Quantization Module Tests")
    print("=" * 60)
    print()
    
    if not test_imports():
        print("\nRunning tests without full dependencies...")
    
    test_quantization_config()
    
    print("=" * 60)
    print("Test suite completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
