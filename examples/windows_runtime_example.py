"""
Example: Using the Multi-Threaded Runtime
Demonstrates concurrent audio generation with the Windows 10 runtime
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity import get_runtime, shutdown_runtime, GenerationTask
from scipy.io.wavfile import write as write_wav
import time


def main():
    print("=" * 60)
    print("Bark Infinity - Multi-Threaded Runtime Example")
    print("=" * 60)
    print()
    
    # Initialize runtime
    print("Initializing multi-threaded runtime...")
    runtime = get_runtime(max_workers=4)
    print(f"Runtime initialized with {runtime.max_workers} workers")
    print()
    
    # Example texts to generate
    texts = [
        "Hello! This is the first audio sample generated using the multi-threaded runtime.",
        "The Windows 10 runtime provides efficient CPU-friendly audio generation.",
        "Voice cloning and audio synthesis have never been easier!",
    ]
    
    # Submit tasks
    print("Submitting tasks...")
    task_ids = []
    for i, text in enumerate(texts):
        task = GenerationTask(
            task_id=f"example_task_{i}",
            text=text,
            temp=0.7,
            priority=i  # Higher number = higher priority
        )
        task_id = runtime.submit_task(task)
        task_ids.append(task_id)
        print(f"  Task {i+1} submitted: {task_id}")
    
    print()
    print("Processing tasks...")
    print()
    
    # Collect results
    results = []
    for i, task_id in enumerate(task_ids):
        print(f"Waiting for task {i+1}...")
        result = runtime.get_result(task_id, timeout=300)
        
        if result and result.success:
            results.append(result)
            print(f"  ✓ Task {i+1} completed!")
            print(f"    Duration: {result.metadata['duration_seconds']:.2f}s")
            print(f"    Processing time: {result.metadata['processing_time']:.2f}s")
            print(f"    Chunks: {result.metadata['chunks']}")
            
            # Save audio file
            output_file = f"output_task_{i+1}.wav"
            write_wav(output_file, result.sample_rate, result.audio_array)
            print(f"    Saved to: {output_file}")
        else:
            error = result.error if result else "Unknown error"
            print(f"  ✗ Task {i+1} failed: {error}")
        
        print()
    
    # Summary
    print("=" * 60)
    print("Summary:")
    print(f"  Total tasks: {len(texts)}")
    print(f"  Successful: {len(results)}")
    print(f"  Failed: {len(texts) - len(results)}")
    print("=" * 60)
    
    # Cleanup
    print()
    print("Cleaning up...")
    shutdown_runtime()
    print("Done!")


if __name__ == "__main__":
    main()
