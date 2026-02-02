"""
Unit tests for Windows Runtime
Tests multi-threaded processing, chunked generation, and layered engine
"""

import unittest
import numpy as np
import sys
import os
from unittest.mock import Mock, patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity.windows_runtime import (
    ChunkedGenerator,
    LayeredGenerationEngine,
    MultiThreadedRuntime,
    GenerationTask,
    GenerationResult,
    get_runtime,
    shutdown_runtime,
)


class TestChunkedGenerator(unittest.TestCase):
    """Test chunked text processing"""
    
    def setUp(self):
        self.generator = ChunkedGenerator(chunk_size=100)
    
    def test_split_text_short(self):
        """Test splitting short text"""
        text = "This is a short text."
        chunks = self.generator.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].endswith('.'))
        # Should not have double periods
        self.assertFalse(chunks[0].endswith('..'))
    
    def test_split_text_without_period(self):
        """Test splitting text without trailing period"""
        text = "This is text without period"
        chunks = self.generator.split_text(text)
        self.assertEqual(len(chunks), 1)
        self.assertTrue(chunks[0].endswith('.'))
        self.assertFalse(chunks[0].endswith('..'))
    
    def test_split_text_long(self):
        """Test splitting long text into chunks"""
        # Create text longer than chunk size
        sentences = ["Sentence number {}.".format(i) for i in range(10)]
        text = " ".join(sentences)
        
        chunks = self.generator.split_text(text)
        self.assertGreater(len(chunks), 1)
        
        # Check all chunks are under size limit (with some buffer)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), self.generator.chunk_size * 2)
    
    def test_split_text_empty(self):
        """Test splitting empty text"""
        text = ""
        chunks = self.generator.split_text(text)
        self.assertEqual(len(chunks), 1)


class TestLayeredGenerationEngine(unittest.TestCase):
    """Test layered generation architecture"""
    
    def setUp(self):
        self.engine = LayeredGenerationEngine(use_cache=True)
    
    def test_cache_initialization(self):
        """Test cache initialization"""
        self.assertIsInstance(self.engine.semantic_cache, dict)
        self.assertIsInstance(self.engine.coarse_cache, dict)
        self.assertIsInstance(self.engine.fine_cache, dict)
    
    def test_clear_caches(self):
        """Test clearing caches"""
        # Add dummy data to caches
        self.engine.semantic_cache['test'] = np.array([1, 2, 3])
        self.engine.coarse_cache['test'] = np.array([4, 5, 6])
        self.engine.fine_cache['test'] = np.array([7, 8, 9])
        
        # Clear caches
        self.engine.clear_caches()
        
        # Verify caches are empty
        self.assertEqual(len(self.engine.semantic_cache), 0)
        self.assertEqual(len(self.engine.coarse_cache), 0)
        self.assertEqual(len(self.engine.fine_cache), 0)


class TestGenerationTask(unittest.TestCase):
    """Test generation task data structure"""
    
    def test_task_creation(self):
        """Test creating a generation task"""
        task = GenerationTask(
            task_id="test_task",
            text="Test text",
            temp=0.7,
            priority=5
        )
        
        self.assertEqual(task.task_id, "test_task")
        self.assertEqual(task.text, "Test text")
        self.assertEqual(task.temp, 0.7)
        self.assertEqual(task.priority, 5)
        self.assertIsNone(task.history_prompt)
    
    def test_task_with_history(self):
        """Test task with history prompt"""
        history = {
            'semantic_prompt': np.array([1, 2, 3]),
            'coarse_prompt': np.array([4, 5, 6])
        }
        
        task = GenerationTask(
            task_id="test_task",
            text="Test text",
            history_prompt=history
        )
        
        self.assertIsNotNone(task.history_prompt)
        self.assertIn('semantic_prompt', task.history_prompt)


class TestGenerationResult(unittest.TestCase):
    """Test generation result data structure"""
    
    def test_result_success(self):
        """Test successful result"""
        audio = np.random.rand(1000)
        result = GenerationResult(
            task_id="test_task",
            audio_array=audio,
            sample_rate=24000,
            metadata={'test': 'data'},
            success=True
        )
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        self.assertEqual(len(result.audio_array), 1000)
    
    def test_result_failure(self):
        """Test failed result"""
        result = GenerationResult(
            task_id="test_task",
            audio_array=np.array([]),
            sample_rate=24000,
            metadata={},
            success=False,
            error="Test error"
        )
        
        self.assertFalse(result.success)
        self.assertEqual(result.error, "Test error")


class TestMultiThreadedRuntime(unittest.TestCase):
    """Test multi-threaded runtime"""
    
    def test_runtime_initialization(self):
        """Test runtime initialization"""
        runtime = MultiThreadedRuntime(max_workers=2)
        
        self.assertEqual(runtime.max_workers, 2)
        self.assertIsNotNone(runtime.executor)
        self.assertIsNotNone(runtime.task_queue)
        self.assertFalse(runtime.is_running)
        
        runtime.executor.shutdown(wait=False)
    
    def test_runtime_start_stop(self):
        """Test starting and stopping runtime"""
        runtime = MultiThreadedRuntime(max_workers=2)
        
        # Start runtime
        runtime.start()
        self.assertTrue(runtime.is_running)
        
        # Stop runtime
        runtime.stop()
        self.assertFalse(runtime.is_running)
    
    def test_task_submission(self):
        """Test submitting a task"""
        runtime = MultiThreadedRuntime(max_workers=2)
        runtime.start()
        
        task = GenerationTask(
            task_id="test_task",
            text="Test text",
            temp=0.7
        )
        
        task_id = runtime.submit_task(task)
        self.assertEqual(task_id, "test_task")
        
        runtime.stop()
    
    def test_get_task_status(self):
        """Test getting task status"""
        runtime = MultiThreadedRuntime(max_workers=2)
        
        status = runtime.get_task_status("nonexistent_task")
        self.assertEqual(status, "unknown")
        
        runtime.executor.shutdown(wait=False)


class TestRuntimeSingleton(unittest.TestCase):
    """Test runtime singleton pattern"""
    
    def tearDown(self):
        """Clean up singleton after each test"""
        shutdown_runtime()
    
    def test_get_runtime_creates_instance(self):
        """Test that get_runtime creates an instance"""
        runtime = get_runtime()
        self.assertIsNotNone(runtime)
        self.assertIsInstance(runtime, MultiThreadedRuntime)
        shutdown_runtime()
    
    def test_get_runtime_returns_same_instance(self):
        """Test that get_runtime returns same instance"""
        runtime1 = get_runtime()
        runtime2 = get_runtime()
        self.assertIs(runtime1, runtime2)
        shutdown_runtime()
    
    def test_shutdown_runtime(self):
        """Test shutting down runtime"""
        runtime = get_runtime()
        self.assertIsNotNone(runtime)
        
        shutdown_runtime()
        
        # Get new runtime should create new instance
        new_runtime = get_runtime()
        self.assertIsNotNone(new_runtime)
        shutdown_runtime()


class TestIntegration(unittest.TestCase):
    """Integration tests for the full system"""
    
    def setUp(self):
        """Set up for integration tests"""
        shutdown_runtime()
    
    def tearDown(self):
        """Clean up after integration tests"""
        shutdown_runtime()
    
    @patch('bark_infinity.windows_runtime.generate_text_semantic')
    @patch('bark_infinity.windows_runtime.generate_coarse')
    @patch('bark_infinity.windows_runtime.generate_fine')
    @patch('bark_infinity.windows_runtime.codec_decode')
    def test_chunked_generator_with_mocks(self, mock_decode, mock_fine, 
                                         mock_coarse, mock_semantic):
        """Test chunked generator with mocked generation functions"""
        # Set up mocks
        mock_semantic.return_value = np.array([1, 2, 3])
        mock_coarse.return_value = np.array([4, 5, 6])
        mock_fine.return_value = np.array([7, 8, 9])
        mock_decode.return_value = np.random.rand(1000)
        
        # Create generator and generate chunk
        generator = ChunkedGenerator()
        audio = generator.generate_chunk("Test text", temp=0.7)
        
        # Verify mocks were called
        mock_semantic.assert_called_once()
        mock_coarse.assert_called_once()
        mock_fine.assert_called_once()
        mock_decode.assert_called_once()
        
        # Verify audio was generated
        self.assertEqual(len(audio), 1000)


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
