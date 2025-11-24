#!/usr/bin/env python3
"""
Example: Basic audio generation with Bark Infinity

This example demonstrates:
1. Basic audio generation
2. Using different voices
3. Saving audio files
4. Error handling
"""

import os
from scipy.io.wavfile import write as write_wav

# Import Bark Infinity
from bark_infinity import generate_audio, SAMPLE_RATE

def example_basic():
    """Generate simple audio."""
    print("Example 1: Basic audio generation")
    print("-" * 50)
    
    text = "Hello, world! This is Bark Infinity."
    
    print(f"Generating audio for: {text}")
    audio_array = generate_audio(text)
    
    # Save to file
    output_file = "example_basic.wav"
    write_wav(output_file, SAMPLE_RATE, audio_array)
    print(f"✓ Audio saved to: {output_file}")
    print()

def example_with_voice():
    """Generate audio with a specific voice."""
    print("Example 2: Using a specific voice")
    print("-" * 50)
    
    text = "This is an example with a specific voice."
    voice = "v2/en_speaker_1"
    
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    
    audio_array = generate_audio(text, history_prompt=voice)
    
    output_file = "example_voice.wav"
    write_wav(output_file, SAMPLE_RATE, audio_array)
    print(f"✓ Audio saved to: {output_file}")
    print()

def example_multiple_clips():
    """Generate multiple audio clips."""
    print("Example 3: Multiple audio clips")
    print("-" * 50)
    
    texts = [
        "First audio clip.",
        "Second audio clip.",
        "Third audio clip.",
    ]
    
    for i, text in enumerate(texts, 1):
        print(f"Generating clip {i}/{len(texts)}: {text}")
        audio_array = generate_audio(text)
        
        output_file = f"example_clip_{i}.wav"
        write_wav(output_file, SAMPLE_RATE, audio_array)
        print(f"✓ Saved to: {output_file}")
    
    print()

def example_error_handling():
    """Demonstrate error handling."""
    print("Example 4: Error handling")
    print("-" * 50)
    
    try:
        # Empty text should be handled gracefully
        text = ""
        print(f"Attempting to generate audio for empty text...")
        audio_array = generate_audio(text)
        print("✓ Generated successfully")
    except Exception as e:
        print(f"✗ Error (expected): {type(e).__name__}: {e}")
    
    print()

def main():
    """Run all examples."""
    print("=" * 60)
    print("Bark Infinity - Basic Examples")
    print("=" * 60)
    print()
    
    # Check if models are downloaded
    print("Note: First run will download models (~12GB)")
    print("This may take several minutes...")
    print()
    
    # Run examples
    try:
        example_basic()
        example_with_voice()
        example_multiple_clips()
        example_error_handling()
        
        print("=" * 60)
        print("All examples completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"Error: Missing dependencies")
        print(f"Install with: pip install bark-infinity")
        print(f"Details: {e}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
