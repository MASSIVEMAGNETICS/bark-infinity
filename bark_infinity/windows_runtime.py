"""
Windows 10 Runtime for Bark Infinity
Multi-threaded, CPU-friendly, chunked layered generational audio processing
"""

import os
import threading
import queue
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import List, Dict, Optional, Callable, Any
import numpy as np
from dataclasses import dataclass
import time
from pathlib import Path

from .generation import (
    generate_text_semantic,
    generate_coarse,
    generate_fine,
    codec_decode,
    SAMPLE_RATE
)
from .config import logger


@dataclass
class GenerationTask:
    """Represents a single audio generation task"""
    task_id: str
    text: str
    history_prompt: Optional[Dict] = None
    temp: float = 0.7
    callback: Optional[Callable] = None
    priority: int = 0
    chunk_size: int = 1000  # characters per chunk


@dataclass
class GenerationResult:
    """Result of audio generation"""
    task_id: str
    audio_array: np.ndarray
    sample_rate: int
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class ChunkedGenerator:
    """
    Handles chunked text processing for long-form generation
    Splits text into manageable chunks for CPU-friendly processing
    """
    
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks at sentence boundaries"""
        # Split by sentences while respecting chunk size
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        
        return chunks
    
    def generate_chunk(self, chunk: str, history_prompt: Optional[Dict] = None,
                      temp: float = 0.7) -> np.ndarray:
        """Generate audio for a single chunk"""
        try:
            # Semantic generation
            semantic_tokens = generate_text_semantic(
                chunk,
                history_prompt=history_prompt,
                temp=temp,
                silent=True,
                use_kv_caching=True
            )
            
            # Coarse generation
            coarse_tokens = generate_coarse(
                semantic_tokens,
                history_prompt=history_prompt,
                temp=temp,
                silent=True,
                use_kv_caching=True
            )
            
            # Fine generation
            fine_tokens = generate_fine(
                coarse_tokens,
                history_prompt=history_prompt,
                temp=0.5,
            )
            
            # Decode to audio
            audio_arr = codec_decode(fine_tokens)
            return audio_arr
            
        except Exception as e:
            logger.error(f"Error generating chunk: {e}")
            raise


class LayeredGenerationEngine:
    """
    Layered generational architecture for progressive audio synthesis
    Handles semantic -> coarse -> fine layers with caching and optimization
    """
    
    def __init__(self, use_cache: bool = True):
        self.use_cache = use_cache
        self.semantic_cache = {}
        self.coarse_cache = {}
        self.fine_cache = {}
    
    def generate_semantic_layer(self, text: str, history_prompt: Optional[Dict] = None,
                               temp: float = 0.7) -> np.ndarray:
        """Generate semantic tokens (first layer)"""
        cache_key = hash((text, str(history_prompt), temp))
        
        if self.use_cache and cache_key in self.semantic_cache:
            logger.info("Using cached semantic tokens")
            return self.semantic_cache[cache_key]
        
        semantic_tokens = generate_text_semantic(
            text,
            history_prompt=history_prompt,
            temp=temp,
            silent=True,
            use_kv_caching=True
        )
        
        if self.use_cache:
            self.semantic_cache[cache_key] = semantic_tokens
        
        return semantic_tokens
    
    def generate_coarse_layer(self, semantic_tokens: np.ndarray,
                             history_prompt: Optional[Dict] = None,
                             temp: float = 0.7) -> np.ndarray:
        """Generate coarse tokens (second layer)"""
        coarse_tokens = generate_coarse(
            semantic_tokens,
            history_prompt=history_prompt,
            temp=temp,
            silent=True,
            use_kv_caching=True
        )
        return coarse_tokens
    
    def generate_fine_layer(self, coarse_tokens: np.ndarray,
                           history_prompt: Optional[Dict] = None) -> np.ndarray:
        """Generate fine tokens (third layer)"""
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
        return fine_tokens
    
    def clear_caches(self):
        """Clear all layer caches"""
        self.semantic_cache.clear()
        self.coarse_cache.clear()
        self.fine_cache.clear()


class MultiThreadedRuntime:
    """
    Multi-threaded runtime for concurrent audio generation
    CPU-friendly with configurable thread pool and task queue
    """
    
    def __init__(self, max_workers: Optional[int] = None, max_queue_size: int = 100):
        """
        Initialize multi-threaded runtime
        
        Args:
            max_workers: Maximum number of worker threads (defaults to CPU count)
            max_queue_size: Maximum size of task queue
        """
        if max_workers is None:
            # Use CPU count but leave some cores free for system
            max_workers = max(1, multiprocessing.cpu_count() - 1)
        
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.results = {}
        self.active_tasks = {}
        self.is_running = False
        self.worker_thread = None
        
        self.chunked_generator = ChunkedGenerator()
        self.layered_engine = LayeredGenerationEngine()
        
        logger.info(f"Initialized MultiThreadedRuntime with {max_workers} workers")
    
    def start(self):
        """Start the runtime worker thread"""
        if self.is_running:
            logger.warning("Runtime already running")
            return
        
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
        logger.info("Runtime started")
    
    def stop(self):
        """Stop the runtime and wait for tasks to complete"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        self.executor.shutdown(wait=True)
        logger.info("Runtime stopped")
    
    def submit_task(self, task: GenerationTask) -> str:
        """
        Submit a task for processing
        
        Args:
            task: GenerationTask to process
            
        Returns:
            task_id for tracking
        """
        priority_tuple = (task.priority, time.time())
        self.task_queue.put((priority_tuple, task))
        logger.info(f"Task {task.task_id} submitted with priority {task.priority}")
        return task.task_id
    
    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Optional[GenerationResult]:
        """Get result for a task (blocking)"""
        start_time = time.time()
        while True:
            if task_id in self.results:
                return self.results.pop(task_id)
            
            if timeout and (time.time() - start_time) > timeout:
                return None
            
            time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> str:
        """Get status of a task"""
        if task_id in self.results:
            return "completed"
        elif task_id in self.active_tasks:
            return "processing"
        else:
            return "unknown"
    
    def _process_queue(self):
        """Worker thread that processes tasks from queue"""
        while self.is_running:
            try:
                # Get task with timeout to allow checking is_running
                priority_tuple, task = self.task_queue.get(timeout=0.5)
                
                self.active_tasks[task.task_id] = task
                
                # Submit to thread pool
                future = self.executor.submit(self._execute_task, task)
                future.add_done_callback(lambda f, tid=task.task_id: self._task_completed(tid, f))
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in queue processing: {e}")
    
    def _execute_task(self, task: GenerationTask) -> GenerationResult:
        """Execute a single generation task"""
        try:
            start_time = time.time()
            
            # Split text into chunks
            chunks = self.chunked_generator.split_text(task.text)
            logger.info(f"Task {task.task_id}: Processing {len(chunks)} chunks")
            
            # Generate audio for each chunk
            audio_chunks = []
            for i, chunk in enumerate(chunks):
                logger.info(f"Task {task.task_id}: Processing chunk {i+1}/{len(chunks)}")
                audio = self.chunked_generator.generate_chunk(
                    chunk,
                    history_prompt=task.history_prompt,
                    temp=task.temp
                )
                audio_chunks.append(audio)
            
            # Concatenate chunks
            full_audio = np.concatenate(audio_chunks) if len(audio_chunks) > 1 else audio_chunks[0]
            
            elapsed_time = time.time() - start_time
            
            result = GenerationResult(
                task_id=task.task_id,
                audio_array=full_audio,
                sample_rate=SAMPLE_RATE,
                metadata={
                    "chunks": len(chunks),
                    "duration_seconds": len(full_audio) / SAMPLE_RATE,
                    "processing_time": elapsed_time,
                    "text_length": len(task.text)
                },
                success=True
            )
            
            logger.info(f"Task {task.task_id} completed in {elapsed_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return GenerationResult(
                task_id=task.task_id,
                audio_array=np.array([]),
                sample_rate=SAMPLE_RATE,
                metadata={},
                success=False,
                error=str(e)
            )
    
    def _task_completed(self, task_id: str, future):
        """Callback when task completes"""
        try:
            result = future.result()
            self.results[task_id] = result
            
            # Remove from active tasks
            task = self.active_tasks.pop(task_id, None)
            
            # Call user callback if provided
            if task and task.callback:
                task.callback(result)
                
        except Exception as e:
            logger.error(f"Error in task completion callback: {e}")
    
    def batch_generate(self, texts: List[str], history_prompt: Optional[Dict] = None,
                      temp: float = 0.7) -> List[GenerationResult]:
        """
        Generate audio for multiple texts in parallel
        
        Args:
            texts: List of texts to generate
            history_prompt: Voice prompt to use
            temp: Generation temperature
            
        Returns:
            List of GenerationResults
        """
        tasks = []
        for i, text in enumerate(texts):
            task = GenerationTask(
                task_id=f"batch_{i}_{time.time()}",
                text=text,
                history_prompt=history_prompt,
                temp=temp
            )
            self.submit_task(task)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = []
        for task in tasks:
            result = self.get_result(task.task_id, timeout=300)  # 5 minute timeout
            if result:
                results.append(result)
        
        return results


# Singleton instance
_runtime_instance = None


def get_runtime(max_workers: Optional[int] = None) -> MultiThreadedRuntime:
    """Get or create the global runtime instance"""
    global _runtime_instance
    if _runtime_instance is None:
        _runtime_instance = MultiThreadedRuntime(max_workers=max_workers)
        _runtime_instance.start()
    return _runtime_instance


def shutdown_runtime():
    """Shutdown the global runtime instance"""
    global _runtime_instance
    if _runtime_instance:
        _runtime_instance.stop()
        _runtime_instance = None
