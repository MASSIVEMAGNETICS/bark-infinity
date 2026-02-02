"""
Unit tests for Voice Studio
Tests voice model management, conversion, and quality analysis
"""

import unittest
import numpy as np
import sys
import os
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity.voice_studio import (
    VoiceModel,
    VoiceModelConverter,
    VoiceQualityAnalyzer,
    VoiceStudio,
    get_voice_studio,
)


class TestVoiceModel(unittest.TestCase):
    """Test VoiceModel data structure"""
    
    def test_model_creation(self):
        """Test creating a voice model"""
        model = VoiceModel(
            id="test_id",
            name="Test Voice",
            format="npz",
            created_at="2024-01-01T00:00:00"
        )
        
        self.assertEqual(model.id, "test_id")
        self.assertEqual(model.name, "Test Voice")
        self.assertEqual(model.format, "npz")
    
    def test_model_to_dict(self):
        """Test converting model to dictionary"""
        model = VoiceModel(
            id="test_id",
            name="Test Voice",
            format="npz",
            created_at="2024-01-01T00:00:00",
            semantic_prompt=np.array([1, 2, 3])
        )
        
        model_dict = model.to_dict()
        
        self.assertIsInstance(model_dict, dict)
        self.assertEqual(model_dict['id'], "test_id")
        self.assertEqual(model_dict['semantic_prompt'], [1, 2, 3])
    
    def test_model_from_dict(self):
        """Test creating model from dictionary"""
        data = {
            'id': 'test_id',
            'name': 'Test Voice',
            'format': 'npz',
            'created_at': '2024-01-01T00:00:00',
            'semantic_prompt': [1, 2, 3]
        }
        
        model = VoiceModel.from_dict(data)
        
        self.assertEqual(model.id, "test_id")
        self.assertIsInstance(model.semantic_prompt, np.ndarray)


class TestVoiceModelConverter(unittest.TestCase):
    """Test voice model format conversion"""
    
    def setUp(self):
        """Set up temporary directory for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.converter = VoiceModelConverter()
    
    def tearDown(self):
        """Clean up temporary directory"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_npz_save_load(self):
        """Test saving and loading NPZ format"""
        data = {
            'semantic_prompt': np.array([1, 2, 3]),
            'coarse_prompt': np.array([4, 5, 6]),
            'fine_prompt': np.array([7, 8, 9])
        }
        
        npz_path = os.path.join(self.temp_dir, "test.npz")
        
        # Save
        self.converter.dict_to_npz(data, npz_path)
        self.assertTrue(os.path.exists(npz_path))
        
        # Load
        loaded_data = self.converter.npz_to_dict(npz_path)
        
        self.assertTrue(np.array_equal(loaded_data['semantic_prompt'], data['semantic_prompt']))
    
    def test_json_save_load(self):
        """Test saving and loading JSON format"""
        data = {
            'semantic_prompt': np.array([1, 2, 3]),
            'coarse_prompt': np.array([4, 5, 6])
        }
        
        json_path = os.path.join(self.temp_dir, "test.json")
        
        # Save
        self.converter.dict_to_json(data, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Load
        loaded_data = self.converter.json_to_dict(json_path)
        
        self.assertTrue(np.array_equal(loaded_data['semantic_prompt'], data['semantic_prompt']))
    
    def test_pkl_save_load(self):
        """Test saving and loading PKL format"""
        data = {
            'semantic_prompt': np.array([1, 2, 3]),
            'coarse_prompt': np.array([4, 5, 6])
        }
        
        pkl_path = os.path.join(self.temp_dir, "test.pkl")
        
        # Save
        self.converter.dict_to_pkl(data, pkl_path)
        self.assertTrue(os.path.exists(pkl_path))
        
        # Load
        loaded_data = self.converter.pkl_to_dict(pkl_path)
        
        self.assertTrue(np.array_equal(loaded_data['semantic_prompt'], data['semantic_prompt']))
    
    def test_format_conversion(self):
        """Test converting between formats"""
        data = {
            'semantic_prompt': np.array([1, 2, 3])
        }
        
        # Create NPZ file
        npz_path = os.path.join(self.temp_dir, "test.npz")
        self.converter.dict_to_npz(data, npz_path)
        
        # Convert to JSON
        json_path = os.path.join(self.temp_dir, "test.json")
        self.converter.convert(npz_path, json_path, 'json')
        
        self.assertTrue(os.path.exists(json_path))
        
        # Verify conversion
        loaded_data = self.converter.json_to_dict(json_path)
        self.assertTrue(np.array_equal(loaded_data['semantic_prompt'], data['semantic_prompt']))


class TestVoiceQualityAnalyzer(unittest.TestCase):
    """Test voice quality analysis"""
    
    def setUp(self):
        self.analyzer = VoiceQualityAnalyzer()
    
    def test_semantic_diversity_analysis(self):
        """Test semantic diversity analysis"""
        semantic_prompt = np.array([1, 2, 3, 1, 2, 3, 4, 5])
        
        analysis = self.analyzer.analyze_semantic_diversity(semantic_prompt)
        
        self.assertIn('diversity_score', analysis)
        self.assertIn('entropy', analysis)
        self.assertIn('unique_tokens', analysis)
        self.assertGreater(analysis['diversity_score'], 0)
    
    def test_coarse_quality_analysis(self):
        """Test coarse quality analysis"""
        coarse_prompt = np.random.rand(2, 100)
        
        analysis = self.analyzer.analyze_coarse_quality(coarse_prompt)
        
        self.assertIn('quality_score', analysis)
        self.assertIn('mean', analysis)
        self.assertIn('std', analysis)
    
    def test_fine_quality_analysis(self):
        """Test fine quality analysis"""
        fine_prompt = np.random.rand(8, 100)
        
        analysis = self.analyzer.analyze_fine_quality(fine_prompt)
        
        self.assertIn('detail_score', analysis)
        self.assertIn('variance', analysis)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis"""
        model = VoiceModel(
            id="test_id",
            name="Test Voice",
            format="npz",
            created_at="2024-01-01T00:00:00",
            semantic_prompt=np.array([1, 2, 3, 4, 5]),
            coarse_prompt=np.random.rand(2, 100),
            fine_prompt=np.random.rand(8, 100)
        )
        
        analysis = self.analyzer.comprehensive_analysis(model)
        
        self.assertIn('model_id', analysis)
        self.assertIn('overall_quality', analysis)
        self.assertIn('semantic', analysis)
        self.assertIn('coarse', analysis)
        self.assertIn('fine', analysis)
        
        # Check overall quality is a valid percentage
        self.assertGreaterEqual(analysis['overall_quality'], 0)
        self.assertLessEqual(analysis['overall_quality'], 1)


class TestVoiceStudio(unittest.TestCase):
    """Test Voice Studio management system"""
    
    def setUp(self):
        """Set up temporary library for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.studio = VoiceStudio(library_path=self.temp_dir)
    
    def tearDown(self):
        """Clean up temporary library"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_studio_initialization(self):
        """Test studio initialization"""
        self.assertTrue(self.studio.library_path.exists())
        self.assertTrue(self.studio.models_dir.exists())
        self.assertIsInstance(self.studio.index, dict)
    
    def test_import_voice_model(self):
        """Test importing a voice model"""
        # Create a test NPZ file
        test_data = {
            'semantic_prompt': np.array([1, 2, 3]),
            'coarse_prompt': np.array([4, 5, 6])
        }
        
        test_file = os.path.join(self.temp_dir, "test_voice.npz")
        np.savez(test_file, **test_data)
        
        # Import the model
        model_id = self.studio.import_voice_model(
            file_path=test_file,
            name="Test Voice",
            transcription="Test transcription"
        )
        
        self.assertIsNotNone(model_id)
        self.assertIn(model_id, self.studio.index)
    
    def test_list_models(self):
        """Test listing models"""
        models = self.studio.list_models()
        self.assertIsInstance(models, list)
    
    def test_get_model(self):
        """Test getting a model by ID"""
        # Import a test model first
        test_data = {
            'semantic_prompt': np.array([1, 2, 3])
        }
        
        test_file = os.path.join(self.temp_dir, "test_voice.npz")
        np.savez(test_file, **test_data)
        
        model_id = self.studio.import_voice_model(test_file, name="Test Voice")
        
        # Get the model
        model = self.studio.get_model(model_id)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.id, model_id)
        self.assertEqual(model.name, "Test Voice")
    
    def test_delete_model(self):
        """Test deleting a model"""
        # Import a test model first
        test_data = {
            'semantic_prompt': np.array([1, 2, 3])
        }
        
        test_file = os.path.join(self.temp_dir, "test_voice.npz")
        np.savez(test_file, **test_data)
        
        model_id = self.studio.import_voice_model(test_file, name="Test Voice")
        
        # Delete the model
        self.studio.delete_model(model_id)
        
        # Verify it's gone
        self.assertNotIn(model_id, self.studio.index)
        model = self.studio.get_model(model_id)
        self.assertIsNone(model)
    
    def test_search_models(self):
        """Test searching models"""
        # Import some test models
        for i in range(3):
            test_data = {'semantic_prompt': np.array([1, 2, 3])}
            test_file = os.path.join(self.temp_dir, f"test_voice_{i}.npz")
            np.savez(test_file, **test_data)
            self.studio.import_voice_model(test_file, name=f"Voice {i}")
        
        # Search for "Voice"
        results = self.studio.search_models("Voice")
        
        self.assertEqual(len(results), 3)
    
    def test_export_voice_model(self):
        """Test exporting a voice model"""
        # Import a test model first
        test_data = {
            'semantic_prompt': np.array([1, 2, 3])
        }
        
        test_file = os.path.join(self.temp_dir, "test_voice.npz")
        np.savez(test_file, **test_data)
        
        model_id = self.studio.import_voice_model(test_file, name="Test Voice")
        
        # Export to JSON
        export_path = os.path.join(self.temp_dir, "exported.json")
        self.studio.export_voice_model(model_id, export_path, 'json')
        
        self.assertTrue(os.path.exists(export_path))


class TestStudioSingleton(unittest.TestCase):
    """Test studio singleton pattern"""
    
    def setUp(self):
        """Reset singleton"""
        import bark_infinity.voice_studio
        bark_infinity.voice_studio._studio_instance = None
    
    def tearDown(self):
        """Clean up singleton"""
        import bark_infinity.voice_studio
        if bark_infinity.voice_studio._studio_instance:
            library_path = bark_infinity.voice_studio._studio_instance.library_path
            bark_infinity.voice_studio._studio_instance = None
            if library_path.exists():
                shutil.rmtree(library_path, ignore_errors=True)
    
    def test_get_voice_studio_creates_instance(self):
        """Test that get_voice_studio creates an instance"""
        studio = get_voice_studio()
        self.assertIsNotNone(studio)
        self.assertIsInstance(studio, VoiceStudio)
    
    def test_get_voice_studio_returns_same_instance(self):
        """Test that get_voice_studio returns same instance"""
        studio1 = get_voice_studio()
        studio2 = get_voice_studio()
        self.assertIs(studio1, studio2)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
