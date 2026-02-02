"""
Example: Voice Studio - Import, Export, and Analyze Voice Models
Demonstrates the comprehensive voice model management system
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity import get_voice_studio
import json


def main():
    print("=" * 60)
    print("Bark Infinity - Voice Studio Example")
    print("=" * 60)
    print()
    
    # Initialize studio
    print("Initializing Voice Studio...")
    studio = get_voice_studio()
    print(f"Studio initialized at: {studio.library_path}")
    print()
    
    # List existing models
    print("Current Voice Models:")
    print("-" * 60)
    models = studio.list_models()
    
    if models:
        for i, model in enumerate(models, 1):
            print(f"{i}. {model['name']}")
            print(f"   ID: {model['id']}")
            print(f"   Format: {model['format']}")
            print(f"   Created: {model['created_at'][:19]}")
            print()
    else:
        print("No voice models found in library.")
        print()
    
    # Example: Import a voice model (if you have one)
    print("=" * 60)
    print("Voice Model Import Example")
    print("-" * 60)
    print()
    print("To import a voice model, use:")
    print()
    print("  model_id = studio.import_voice_model(")
    print("      file_path='path/to/your/voice_model.npz',")
    print("      name='My Voice',")
    print("      transcription='Optional transcription text'")
    print("  )")
    print()
    
    # Example: Export a model (if we have any)
    if models:
        print("=" * 60)
        print("Voice Model Export Example")
        print("-" * 60)
        print()
        
        model_id = models[0]['id']
        model_name = models[0]['name']
        
        print(f"Exporting model: {model_name}")
        print()
        
        # Export to JSON format
        try:
            output_path = f"exported_{model_name}.json"
            studio.export_voice_model(
                model_id=model_id,
                output_path=output_path,
                output_format='json'
            )
            print(f"✓ Exported to: {output_path}")
            print()
        except Exception as e:
            print(f"✗ Export failed: {e}")
            print()
        
        # Analyze the model
        print("=" * 60)
        print("Voice Model Analysis")
        print("-" * 60)
        print()
        
        try:
            analysis = studio.analyze_model(model_id)
            
            print(f"Model: {analysis['model_name']}")
            print(f"Overall Quality: {analysis['overall_quality']:.2%}")
            print()
            
            if 'semantic' in analysis:
                sem = analysis['semantic']
                print("Semantic Layer:")
                print(f"  Diversity Score: {sem['diversity_score']:.2%}")
                print(f"  Entropy: {sem['entropy']:.2f}")
                print(f"  Unique Tokens: {sem['unique_tokens']}")
                print(f"  Total Tokens: {sem['total_tokens']}")
                print()
            
            if 'coarse' in analysis:
                coarse = analysis['coarse']
                print("Coarse Layer:")
                print(f"  Quality Score: {coarse['quality_score']:.2%}")
                print(f"  Mean: {coarse['mean']:.2f}")
                print(f"  Std Dev: {coarse['std']:.2f}")
                print()
            
            if 'fine' in analysis:
                fine = analysis['fine']
                print("Fine Layer:")
                print(f"  Detail Score: {fine['detail_score']:.2%}")
                print(f"  Variance: {fine['variance']:.2f}")
                print()
            
        except Exception as e:
            print(f"✗ Analysis failed: {e}")
            print()
    
    # Example: Search models
    print("=" * 60)
    print("Voice Model Search Example")
    print("-" * 60)
    print()
    
    if models:
        search_query = models[0]['name'].split()[0]  # First word of first model
        print(f"Searching for: '{search_query}'")
        results = studio.search_models(search_query)
        print(f"Found {len(results)} matching models")
        print()
    
    # Example: Format conversion
    print("=" * 60)
    print("Format Conversion Examples")
    print("-" * 60)
    print()
    print("The Voice Studio supports multiple formats:")
    print("  • NPZ (NumPy compressed)")
    print("  • PKL (Python pickle)")
    print("  • JSON (JavaScript Object Notation)")
    print()
    print("Convert between formats using:")
    print()
    print("  from bark_infinity.voice_studio import VoiceModelConverter")
    print("  converter = VoiceModelConverter()")
    print("  converter.convert('model.npz', 'model.json', 'json')")
    print()
    
    print("=" * 60)
    print("Voice Studio Features:")
    print("-" * 60)
    print("✓ Multi-format import/export (NPZ, PKL, JSON)")
    print("✓ Comprehensive quality analysis")
    print("✓ Centralized model library")
    print("✓ Search and filter capabilities")
    print("✓ Batch import/export")
    print("✓ Voice cloning from audio")
    print("=" * 60)


if __name__ == "__main__":
    main()
