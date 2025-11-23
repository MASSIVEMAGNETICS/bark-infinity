"""
Quantization support for Bark Infinity to enable low-compute deployment.

This module provides model quantization utilities to reduce memory footprint
and improve inference speed on resource-constrained devices.
"""

import os
import logging
from typing import Optional, Dict, Any
import torch
from .config import logger

# Check for optional quantization libraries
try:
    import bitsandbytes as bnb
    HAS_BITSANDBYTES = True
except ImportError:
    HAS_BITSANDBYTES = False
    logger.warning("bitsandbytes not available. Install with: pip install bitsandbytes")

try:
    from optimum.bettertransformer import BetterTransformer
    HAS_OPTIMUM = True
except ImportError:
    HAS_OPTIMUM = False
    logger.warning("optimum not available. Install with: pip install optimum")


class QuantizationConfig:
    """Configuration for model quantization."""
    
    def __init__(
        self,
        enable_8bit: bool = False,
        enable_4bit: bool = False,
        use_better_transformer: bool = False,
        device: Optional[str] = None,
    ):
        """
        Initialize quantization configuration.
        
        Args:
            enable_8bit: Enable 8-bit quantization (requires bitsandbytes)
            enable_4bit: Enable 4-bit quantization (requires bitsandbytes)
            use_better_transformer: Enable BetterTransformer optimization (requires optimum)
            device: Target device ('cuda', 'cpu', 'mps')
        """
        self.enable_8bit = enable_8bit
        self.enable_4bit = enable_4bit
        self.use_better_transformer = use_better_transformer
        self.device = device or self._get_default_device()
        
        # Validate configuration
        if self.enable_8bit and not HAS_BITSANDBYTES:
            logger.warning("8-bit quantization requested but bitsandbytes not available")
            self.enable_8bit = False
            
        if self.enable_4bit and not HAS_BITSANDBYTES:
            logger.warning("4-bit quantization requested but bitsandbytes not available")
            self.enable_4bit = False
            
        if self.use_better_transformer and not HAS_OPTIMUM:
            logger.warning("BetterTransformer requested but optimum not available")
            self.use_better_transformer = False
    
    @staticmethod
    def _get_default_device() -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    
    @classmethod
    def from_env(cls) -> "QuantizationConfig":
        """Create configuration from environment variables."""
        return cls(
            enable_8bit=os.getenv("BARK_QUANTIZE_8BIT", "").lower() == "true",
            enable_4bit=os.getenv("BARK_QUANTIZE_4BIT", "").lower() == "true",
            use_better_transformer=os.getenv("BARK_USE_BETTER_TRANSFORMER", "").lower() == "true",
        )
    
    @classmethod
    def low_compute_preset(cls) -> "QuantizationConfig":
        """Preset configuration for low-compute devices."""
        if HAS_BITSANDBYTES and torch.cuda.is_available():
            # Use 8-bit on CUDA with bitsandbytes
            return cls(enable_8bit=True, use_better_transformer=HAS_OPTIMUM)
        else:
            # Fallback to CPU optimizations
            return cls(use_better_transformer=HAS_OPTIMUM, device="cpu")


def quantize_model(model: torch.nn.Module, config: QuantizationConfig) -> torch.nn.Module:
    """
    Apply quantization to a model based on configuration.
    
    Args:
        model: PyTorch model to quantize
        config: Quantization configuration
        
    Returns:
        Quantized model
    """
    logger.info(f"Applying quantization: 8bit={config.enable_8bit}, 4bit={config.enable_4bit}")
    
    # Apply BetterTransformer optimization if available
    if config.use_better_transformer and HAS_OPTIMUM:
        try:
            logger.info("Applying BetterTransformer optimization")
            model = BetterTransformer.transform(model)
        except Exception as e:
            logger.warning(f"BetterTransformer failed: {e}")
    
    # Apply bitsandbytes quantization if enabled
    if (config.enable_8bit or config.enable_4bit) and HAS_BITSANDBYTES:
        if config.device == "cuda":
            try:
                # Note: Model-specific quantization would be applied during model loading
                # This is a placeholder for future model-specific quantization
                logger.info("Quantization will be applied during model loading")
            except Exception as e:
                logger.warning(f"Quantization failed: {e}")
        else:
            logger.warning("Quantization requires CUDA but device is: " + config.device)
    
    return model


def get_quantization_kwargs(config: QuantizationConfig) -> Dict[str, Any]:
    """
    Get kwargs for model loading with quantization.
    
    Args:
        config: Quantization configuration
        
    Returns:
        Dictionary of kwargs for model loading
    """
    kwargs = {}
    
    if config.enable_8bit and HAS_BITSANDBYTES and config.device == "cuda":
        kwargs.update({
            "load_in_8bit": True,
            "device_map": "auto",
        })
    elif config.enable_4bit and HAS_BITSANDBYTES and config.device == "cuda":
        kwargs.update({
            "load_in_4bit": True,
            "device_map": "auto",
        })
    
    return kwargs


def estimate_memory_savings(
    base_memory_gb: float,
    enable_8bit: bool = False,
    enable_4bit: bool = False,
) -> Dict[str, float]:
    """
    Estimate memory savings from quantization.
    
    Args:
        base_memory_gb: Base memory requirement in GB
        enable_8bit: Whether 8-bit quantization is enabled
        enable_4bit: Whether 4-bit quantization is enabled
        
    Returns:
        Dictionary with memory estimates
    """
    multiplier = 1.0
    if enable_4bit:
        multiplier = 0.25  # 4-bit = 25% of original
    elif enable_8bit:
        multiplier = 0.5   # 8-bit = 50% of original
    
    estimated_memory = base_memory_gb * multiplier
    savings = base_memory_gb - estimated_memory
    
    return {
        "base_memory_gb": base_memory_gb,
        "estimated_memory_gb": estimated_memory,
        "savings_gb": savings,
        "reduction_percent": (savings / base_memory_gb) * 100,
    }


# Convenience function for easy integration
def setup_low_compute_mode() -> QuantizationConfig:
    """
    Set up optimal configuration for low-compute devices.
    
    Returns:
        Configured QuantizationConfig
    """
    config = QuantizationConfig.low_compute_preset()
    
    # Log the configuration
    logger.info("=" * 60)
    logger.info("Low Compute Mode Configuration:")
    logger.info(f"  Device: {config.device}")
    logger.info(f"  8-bit quantization: {config.enable_8bit}")
    logger.info(f"  4-bit quantization: {config.enable_4bit}")
    logger.info(f"  BetterTransformer: {config.use_better_transformer}")
    
    if config.device == "cuda" and (config.enable_8bit or config.enable_4bit):
        savings = estimate_memory_savings(
            12.0,  # Base Bark model memory requirement
            enable_8bit=config.enable_8bit,
            enable_4bit=config.enable_4bit,
        )
        logger.info(f"  Estimated VRAM: {savings['estimated_memory_gb']:.1f}GB "
                   f"(~{savings['reduction_percent']:.0f}% reduction)")
    
    logger.info("=" * 60)
    
    return config
