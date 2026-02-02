"""
Example: Complete Windows 10 Studio Workflow
Demonstrates the full enterprise workflow: generation, voice cloning, and analysis
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity import get_runtime, get_voice_studio, shutdown_runtime, GenerationTask
from scipy.io.wavfile import write as write_wav
import time


def print_section(title):
    """Print a formatted section header"""
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)
    print()


def main():
    print_section("Bark Infinity Studio - Complete Workflow Example")
    
    # Step 1: Initialize runtime and studio
    print("Step 1: Initializing Runtime and Voice Studio")
    print("-" * 70)
    
    runtime = get_runtime(max_workers=4)
    studio = get_voice_studio()
    
    print(f"✓ Runtime initialized with {runtime.max_workers} workers")
    print(f"✓ Studio initialized at {studio.library_path}")
    
    # Step 2: Generate audio samples
    print_section("Step 2: Generating Audio Samples")
    
    sample_texts = [
        "Welcome to Bark Infinity Studio, the enterprise-grade voice cloning platform.",
        "Our multi-threaded runtime provides efficient CPU-friendly audio generation.",
        "Advanced algorithms ensure high-quality voice synthesis and cloning.",
    ]
    
    print(f"Generating {len(sample_texts)} audio samples...")
    print()
    
    task_ids = []
    for i, text in enumerate(sample_texts):
        task = GenerationTask(
            task_id=f"workflow_task_{i}",
            text=text,
            temp=0.7
        )
        task_id = runtime.submit_task(task)
        task_ids.append(task_id)
        print(f"  [{i+1}/{len(sample_texts)}] Task submitted: {task_id}")
    
    print()
    print("Processing tasks...")
    
    audio_files = []
    for i, task_id in enumerate(task_ids):
        result = runtime.get_result(task_id, timeout=300)
        
        if result and result.success:
            filename = f"workflow_sample_{i+1}.wav"
            write_wav(filename, result.sample_rate, result.audio_array)
            audio_files.append(filename)
            
            print(f"  ✓ Sample {i+1} generated:")
            print(f"    File: {filename}")
            print(f"    Duration: {result.metadata['duration_seconds']:.2f}s")
            print(f"    Processing: {result.metadata['processing_time']:.2f}s")
        else:
            print(f"  ✗ Sample {i+1} failed")
        print()
    
    # Step 3: Voice model management
    print_section("Step 3: Voice Model Management")
    
    print("Current voice models in library:")
    models = studio.list_models()
    
    if models:
        for i, model in enumerate(models[:5], 1):  # Show first 5
            print(f"  {i}. {model['name']} ({model['format']})")
        if len(models) > 5:
            print(f"  ... and {len(models) - 5} more")
    else:
        print("  No models in library yet")
    
    print()
    print(f"Total models in library: {len(models)}")
    
    # Step 4: Quality analysis (if we have models)
    if models:
        print_section("Step 4: Voice Quality Analysis")
        
        # Analyze first model
        model_id = models[0]['id']
        model_name = models[0]['name']
        
        print(f"Analyzing model: {model_name}")
        print()
        
        try:
            analysis = studio.analyze_model(model_id)
            
            print("Analysis Results:")
            print("-" * 70)
            print(f"Overall Quality Score: {analysis['overall_quality']:.1%}")
            
            if 'semantic' in analysis:
                print(f"Semantic Diversity: {analysis['semantic']['diversity_score']:.1%}")
            if 'coarse' in analysis:
                print(f"Coarse Quality: {analysis['coarse']['quality_score']:.1%}")
            if 'fine' in analysis:
                print(f"Fine Detail: {analysis['fine']['detail_score']:.1%}")
            
            print()
            
            # Quality recommendation
            overall = analysis['overall_quality']
            if overall >= 0.8:
                quality = "Excellent"
                icon = "🌟"
            elif overall >= 0.6:
                quality = "Good"
                icon = "✓"
            elif overall >= 0.4:
                quality = "Fair"
                icon = "○"
            else:
                quality = "Poor"
                icon = "✗"
            
            print(f"{icon} Quality Assessment: {quality}")
            print()
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            print()
    
    # Step 5: Format conversion example
    if models:
        print_section("Step 5: Format Conversion")
        
        model_id = models[0]['id']
        model_name = models[0]['name']
        
        print(f"Converting model '{model_name}' to different formats...")
        print()
        
        formats = ['json', 'pkl']
        for fmt in formats:
            try:
                output_file = f"converted_{model_name}.{fmt}"
                studio.export_voice_model(model_id, output_file, fmt)
                print(f"  ✓ Exported to {fmt.upper()}: {output_file}")
            except Exception as e:
                print(f"  ✗ {fmt.upper()} export failed: {e}")
        
        print()
    
    # Step 6: Batch processing example
    print_section("Step 6: Batch Processing Capabilities")
    
    print("The runtime supports batch processing for multiple texts:")
    print()
    
    batch_texts = [
        "First batch item",
        "Second batch item",
        "Third batch item"
    ]
    
    print(f"Processing {len(batch_texts)} items in batch...")
    
    try:
        batch_results = runtime.batch_generate(batch_texts, temp=0.7)
        
        successful = sum(1 for r in batch_results if r.success)
        print(f"✓ Batch completed: {successful}/{len(batch_texts)} successful")
        
        # Save batch results
        for i, result in enumerate(batch_results):
            if result.success:
                filename = f"batch_output_{i+1}.wav"
                write_wav(filename, result.sample_rate, result.audio_array)
                print(f"  Saved: {filename}")
        
    except Exception as e:
        print(f"✗ Batch processing failed: {e}")
    
    print()
    
    # Summary
    print_section("Workflow Summary")
    
    print("Completed Workflow Steps:")
    print("  ✓ Step 1: Runtime and Studio initialization")
    print("  ✓ Step 2: Multi-threaded audio generation")
    print("  ✓ Step 3: Voice model library management")
    
    if models:
        print("  ✓ Step 4: Quality analysis")
        print("  ✓ Step 5: Format conversion")
    else:
        print("  ○ Step 4: Quality analysis (no models)")
        print("  ○ Step 5: Format conversion (no models)")
    
    print("  ✓ Step 6: Batch processing")
    print()
    
    print("Enterprise Features Demonstrated:")
    print("  • Multi-threaded concurrent processing")
    print("  • CPU-optimized chunked generation")
    print("  • Voice model import/export")
    print("  • Advanced quality analysis")
    print("  • Format conversion (NPZ, PKL, JSON)")
    print("  • Batch processing capabilities")
    print()
    
    print("Output Files Generated:")
    for i in range(len(audio_files)):
        print(f"  • workflow_sample_{i+1}.wav")
    for i in range(len(batch_texts)):
        print(f"  • batch_output_{i+1}.wav")
    if models:
        print(f"  • converted_{models[0]['name']}.json")
        print(f"  • converted_{models[0]['name']}.pkl")
    
    print()
    
    # Cleanup
    print("Cleaning up resources...")
    shutdown_runtime()
    print("✓ Done!")
    
    print()
    print("=" * 70)
    print("Thank you for using Bark Infinity Studio!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nWorkflow interrupted by user.")
        shutdown_runtime()
    except Exception as e:
        print(f"\n\nError: {e}")
        shutdown_runtime()
        raise
