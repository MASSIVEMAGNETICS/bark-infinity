"""
Comprehensive Voice Clone Studio
Enterprise-grade voice model management with multi-format import/export
"""

import os
import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import hashlib
from datetime import datetime
import shutil

from .config import logger
from .generation import SAMPLE_RATE
from .clonevoice import clone_voice

try:
    import torchaudio
except ImportError:
    torchaudio = None


@dataclass
class VoiceModel:
    """Voice model metadata and data"""
    id: str
    name: str
    format: str  # 'npz', 'pkl', 'json', 'wav'
    created_at: str
    semantic_prompt: Optional[np.ndarray] = None
    coarse_prompt: Optional[np.ndarray] = None
    fine_prompt: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None
    audio_path: Optional[str] = None
    transcription: Optional[str] = None
    sample_rate: int = SAMPLE_RATE
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        # Convert numpy arrays to lists for JSON serialization
        if self.semantic_prompt is not None:
            data['semantic_prompt'] = self.semantic_prompt.tolist()
        if self.coarse_prompt is not None:
            data['coarse_prompt'] = self.coarse_prompt.tolist()
        if self.fine_prompt is not None:
            data['fine_prompt'] = self.fine_prompt.tolist()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VoiceModel':
        """Create VoiceModel from dictionary"""
        # Convert lists back to numpy arrays
        if 'semantic_prompt' in data and data['semantic_prompt'] is not None:
            data['semantic_prompt'] = np.array(data['semantic_prompt'])
        if 'coarse_prompt' in data and data['coarse_prompt'] is not None:
            data['coarse_prompt'] = np.array(data['coarse_prompt'])
        if 'fine_prompt' in data and data['fine_prompt'] is not None:
            data['fine_prompt'] = np.array(data['fine_prompt'])
        return cls(**data)


class VoiceModelConverter:
    """
    Convert between different voice model formats
    Supports: NPZ, PKL, JSON, WAV
    """
    
    @staticmethod
    def npz_to_dict(npz_path: str) -> Dict:
        """Load NPZ format voice model"""
        data = np.load(npz_path, allow_pickle=True)
        return {
            'semantic_prompt': data.get('semantic_prompt'),
            'coarse_prompt': data.get('coarse_prompt'),
            'fine_prompt': data.get('fine_prompt'),
        }
    
    @staticmethod
    def dict_to_npz(data: Dict, npz_path: str):
        """Save voice model as NPZ format"""
        save_data = {}
        if 'semantic_prompt' in data and data['semantic_prompt'] is not None:
            save_data['semantic_prompt'] = data['semantic_prompt']
        if 'coarse_prompt' in data and data['coarse_prompt'] is not None:
            save_data['coarse_prompt'] = data['coarse_prompt']
        if 'fine_prompt' in data and data['fine_prompt'] is not None:
            save_data['fine_prompt'] = data['fine_prompt']
        
        np.savez(npz_path, **save_data)
    
    @staticmethod
    def pkl_to_dict(pkl_path: str) -> Dict:
        """Load PKL format voice model"""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    @staticmethod
    def dict_to_pkl(data: Dict, pkl_path: str):
        """Save voice model as PKL format"""
        with open(pkl_path, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def json_to_dict(json_path: str) -> Dict:
        """Load JSON format voice model"""
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        if 'semantic_prompt' in data and data['semantic_prompt']:
            data['semantic_prompt'] = np.array(data['semantic_prompt'])
        if 'coarse_prompt' in data and data['coarse_prompt']:
            data['coarse_prompt'] = np.array(data['coarse_prompt'])
        if 'fine_prompt' in data and data['fine_prompt']:
            data['fine_prompt'] = np.array(data['fine_prompt'])
        
        return data
    
    @staticmethod
    def dict_to_json(data: Dict, json_path: str):
        """Save voice model as JSON format"""
        save_data = data.copy()
        
        # Convert numpy arrays to lists
        if 'semantic_prompt' in save_data and isinstance(save_data['semantic_prompt'], np.ndarray):
            save_data['semantic_prompt'] = save_data['semantic_prompt'].tolist()
        if 'coarse_prompt' in save_data and isinstance(save_data['coarse_prompt'], np.ndarray):
            save_data['coarse_prompt'] = save_data['coarse_prompt'].tolist()
        if 'fine_prompt' in save_data and isinstance(save_data['fine_prompt'], np.ndarray):
            save_data['fine_prompt'] = save_data['fine_prompt'].tolist()
        
        with open(json_path, 'w') as f:
            json.dump(save_data, f, indent=2)
    
    @staticmethod
    def convert(input_path: str, output_path: str, output_format: str):
        """
        Convert voice model from one format to another
        
        Args:
            input_path: Path to input voice model file
            output_path: Path to output voice model file
            output_format: Target format ('npz', 'pkl', 'json')
        """
        # Detect input format
        input_ext = Path(input_path).suffix.lower()
        
        # Load data based on input format
        if input_ext == '.npz':
            data = VoiceModelConverter.npz_to_dict(input_path)
        elif input_ext == '.pkl':
            data = VoiceModelConverter.pkl_to_dict(input_path)
        elif input_ext == '.json':
            data = VoiceModelConverter.json_to_dict(input_path)
        else:
            raise ValueError(f"Unsupported input format: {input_ext}")
        
        # Save data based on output format
        if output_format == 'npz':
            VoiceModelConverter.dict_to_npz(data, output_path)
        elif output_format == 'pkl':
            VoiceModelConverter.dict_to_pkl(data, output_path)
        elif output_format == 'json':
            VoiceModelConverter.dict_to_json(data, output_path)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
        
        logger.info(f"Converted {input_path} to {output_path} ({output_format})")


class VoiceQualityAnalyzer:
    """
    Analyze voice model quality using advanced algorithms
    """
    
    @staticmethod
    def analyze_semantic_diversity(semantic_prompt: np.ndarray) -> Dict[str, float]:
        """Analyze semantic token diversity"""
        if semantic_prompt is None or len(semantic_prompt) == 0:
            return {"diversity_score": 0.0}
        
        # Calculate unique token ratio
        unique_tokens = len(np.unique(semantic_prompt))
        total_tokens = len(semantic_prompt)
        diversity_ratio = unique_tokens / total_tokens
        
        # Calculate entropy
        token_counts = np.bincount(semantic_prompt)
        probabilities = token_counts / total_tokens
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return {
            "diversity_score": float(diversity_ratio),
            "entropy": float(entropy),
            "unique_tokens": int(unique_tokens),
            "total_tokens": int(total_tokens)
        }
    
    @staticmethod
    def analyze_coarse_quality(coarse_prompt: np.ndarray) -> Dict[str, float]:
        """Analyze coarse token quality"""
        if coarse_prompt is None or coarse_prompt.size == 0:
            return {"quality_score": 0.0}
        
        # Calculate spectral flatness
        mean_val = np.mean(coarse_prompt)
        std_val = np.std(coarse_prompt)
        
        # Check for consistency
        consistency_score = 1.0 - min(std_val / (abs(mean_val) + 1e-10), 1.0)
        
        return {
            "quality_score": float(consistency_score),
            "mean": float(mean_val),
            "std": float(std_val),
            "shape": coarse_prompt.shape
        }
    
    @staticmethod
    def analyze_fine_quality(fine_prompt: np.ndarray) -> Dict[str, float]:
        """Analyze fine token quality"""
        if fine_prompt is None or fine_prompt.size == 0:
            return {"detail_score": 0.0}
        
        # Calculate detail level
        detail_variance = np.var(fine_prompt)
        
        return {
            "detail_score": float(min(detail_variance / 1000.0, 1.0)),
            "variance": float(detail_variance),
            "shape": fine_prompt.shape
        }
    
    @staticmethod
    def comprehensive_analysis(voice_model: VoiceModel) -> Dict[str, Any]:
        """Perform comprehensive quality analysis"""
        analysis = {
            "model_id": voice_model.id,
            "model_name": voice_model.name,
            "timestamp": datetime.now().isoformat()
        }
        
        if voice_model.semantic_prompt is not None:
            analysis["semantic"] = VoiceQualityAnalyzer.analyze_semantic_diversity(
                voice_model.semantic_prompt
            )
        
        if voice_model.coarse_prompt is not None:
            analysis["coarse"] = VoiceQualityAnalyzer.analyze_coarse_quality(
                voice_model.coarse_prompt
            )
        
        if voice_model.fine_prompt is not None:
            analysis["fine"] = VoiceQualityAnalyzer.analyze_fine_quality(
                voice_model.fine_prompt
            )
        
        # Calculate overall quality score
        scores = []
        if "semantic" in analysis:
            scores.append(analysis["semantic"].get("diversity_score", 0))
        if "coarse" in analysis:
            scores.append(analysis["coarse"].get("quality_score", 0))
        if "fine" in analysis:
            scores.append(analysis["fine"].get("detail_score", 0))
        
        analysis["overall_quality"] = sum(scores) / len(scores) if scores else 0.0
        
        return analysis


class VoiceStudio:
    """
    Enterprise-grade voice clone studio
    Comprehensive voice model management system
    """
    
    def __init__(self, library_path: str = None):
        """
        Initialize Voice Studio
        
        Args:
            library_path: Path to voice model library directory
        """
        if library_path is None:
            library_path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "bark_infinity",
                "voice_library"
            )
        
        self.library_path = Path(library_path)
        self.library_path.mkdir(parents=True, exist_ok=True)
        
        self.index_path = self.library_path / "index.json"
        self.models_dir = self.library_path / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.converter = VoiceModelConverter()
        self.analyzer = VoiceQualityAnalyzer()
        
        self.index = self._load_index()
        
        logger.info(f"Voice Studio initialized at {self.library_path}")
    
    def _load_index(self) -> Dict[str, Dict]:
        """Load voice model index"""
        if self.index_path.exists():
            with open(self.index_path, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_index(self):
        """Save voice model index"""
        with open(self.index_path, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _generate_id(self, name: str) -> str:
        """Generate unique ID for voice model"""
        timestamp = datetime.now().isoformat()
        hash_input = f"{name}_{timestamp}".encode()
        return hashlib.sha256(hash_input).hexdigest()[:16]
    
    def import_voice_model(self, file_path: str, name: str = None,
                          transcription: str = None) -> str:
        """
        Import voice model from file
        
        Args:
            file_path: Path to voice model file
            name: Name for the voice model
            transcription: Optional transcription text
            
        Returns:
            Voice model ID
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Voice model file not found: {file_path}")
        
        # Determine format
        ext = file_path.suffix.lower()
        format_map = {'.npz': 'npz', '.pkl': 'pkl', '.json': 'json', '.wav': 'wav'}
        model_format = format_map.get(ext, 'unknown')
        
        if model_format == 'unknown':
            raise ValueError(f"Unsupported voice model format: {ext}")
        
        # Generate ID and name
        if name is None:
            name = file_path.stem
        
        model_id = self._generate_id(name)
        
        # Load voice data
        voice_data = {}
        if model_format in ['npz', 'pkl', 'json']:
            if model_format == 'npz':
                voice_data = self.converter.npz_to_dict(str(file_path))
            elif model_format == 'pkl':
                voice_data = self.converter.pkl_to_dict(str(file_path))
            elif model_format == 'json':
                voice_data = self.converter.json_to_dict(str(file_path))
        
        # Create voice model
        voice_model = VoiceModel(
            id=model_id,
            name=name,
            format=model_format,
            created_at=datetime.now().isoformat(),
            semantic_prompt=voice_data.get('semantic_prompt'),
            coarse_prompt=voice_data.get('coarse_prompt'),
            fine_prompt=voice_data.get('fine_prompt'),
            transcription=transcription
        )
        
        # Save to library
        model_path = self.models_dir / f"{model_id}.npz"
        self.converter.dict_to_npz(voice_data, str(model_path))
        
        # Update index
        self.index[model_id] = {
            "id": model_id,
            "name": name,
            "format": model_format,
            "created_at": voice_model.created_at,
            "model_path": str(model_path),
            "transcription": transcription
        }
        self._save_index()
        
        logger.info(f"Imported voice model: {name} ({model_id})")
        return model_id
    
    def export_voice_model(self, model_id: str, output_path: str,
                          output_format: str = 'npz'):
        """
        Export voice model to file
        
        Args:
            model_id: Voice model ID
            output_path: Output file path
            output_format: Output format ('npz', 'pkl', 'json')
        """
        if model_id not in self.index:
            raise ValueError(f"Voice model not found: {model_id}")
        
        model_info = self.index[model_id]
        input_path = model_info['model_path']
        
        self.converter.convert(input_path, output_path, output_format)
        logger.info(f"Exported voice model {model_id} to {output_path}")
    
    def list_models(self) -> List[Dict]:
        """List all voice models in library"""
        return list(self.index.values())
    
    def get_model(self, model_id: str) -> Optional[VoiceModel]:
        """Get voice model by ID"""
        if model_id not in self.index:
            return None
        
        model_info = self.index[model_id]
        model_path = model_info['model_path']
        
        voice_data = self.converter.npz_to_dict(model_path)
        
        return VoiceModel(
            id=model_id,
            name=model_info['name'],
            format=model_info['format'],
            created_at=model_info['created_at'],
            semantic_prompt=voice_data.get('semantic_prompt'),
            coarse_prompt=voice_data.get('coarse_prompt'),
            fine_prompt=voice_data.get('fine_prompt'),
            transcription=model_info.get('transcription')
        )
    
    def delete_model(self, model_id: str):
        """Delete voice model from library"""
        if model_id not in self.index:
            raise ValueError(f"Voice model not found: {model_id}")
        
        model_info = self.index[model_id]
        model_path = Path(model_info['model_path'])
        
        if model_path.exists():
            model_path.unlink()
        
        del self.index[model_id]
        self._save_index()
        
        logger.info(f"Deleted voice model: {model_id}")
    
    def analyze_model(self, model_id: str) -> Dict[str, Any]:
        """Analyze voice model quality"""
        voice_model = self.get_model(model_id)
        if voice_model is None:
            raise ValueError(f"Voice model not found: {model_id}")
        
        return self.analyzer.comprehensive_analysis(voice_model)
    
    def clone_from_audio(self, audio_path: str, transcription: str,
                        name: str = None) -> str:
        """
        Create voice model from audio file
        
        Args:
            audio_path: Path to audio file
            transcription: Text transcription of audio
            name: Name for voice model
            
        Returns:
            Voice model ID
        """
        if name is None:
            name = Path(audio_path).stem
        
        # Use existing clone_voice functionality
        dest_filename = f"cloned_{name}"
        result = clone_voice(audio_path, transcription, dest_filename)
        
        # Import the cloned voice
        # Note: clone_voice creates files in voice_clone_samples directory
        clone_dir = Path(f"voice_clone_samples/{dest_filename}_clones")
        
        if clone_dir.exists():
            # Find the first .npz file
            npz_files = list(clone_dir.glob("*.npz"))
            if npz_files:
                model_id = self.import_voice_model(
                    str(npz_files[0]),
                    name=name,
                    transcription=transcription
                )
                logger.info(f"Cloned voice model: {name} ({model_id})")
                return model_id
        
        raise RuntimeError("Failed to clone voice from audio")
    
    def batch_import(self, directory: str) -> List[str]:
        """
        Import all voice models from directory
        
        Args:
            directory: Directory containing voice model files
            
        Returns:
            List of imported model IDs
        """
        directory = Path(directory)
        imported_ids = []
        
        for file_path in directory.glob("*"):
            if file_path.suffix.lower() in ['.npz', '.pkl', '.json']:
                try:
                    model_id = self.import_voice_model(str(file_path))
                    imported_ids.append(model_id)
                except Exception as e:
                    logger.error(f"Failed to import {file_path}: {e}")
        
        logger.info(f"Batch imported {len(imported_ids)} voice models")
        return imported_ids
    
    def search_models(self, query: str) -> List[Dict]:
        """Search voice models by name"""
        query = query.lower()
        results = []
        
        for model_info in self.index.values():
            if query in model_info['name'].lower():
                results.append(model_info)
        
        return results


# Singleton instance
_studio_instance = None


def get_voice_studio(library_path: str = None) -> VoiceStudio:
    """Get or create the global voice studio instance"""
    global _studio_instance
    if _studio_instance is None:
        _studio_instance = VoiceStudio(library_path=library_path)
    return _studio_instance
