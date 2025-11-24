#!/usr/bin/env python3
"""
Command-line interface for Bark Infinity.
Provides easy access to Bark Infinity features from the command line.
"""

import sys
import argparse
from bark_infinity import __version__


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bark Infinity - Advanced text-to-audio generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate audio from text
  bark-infinity generate "Hello, world!"
  
  # Start web UI
  bark-infinity webui
  
  # Enable low-compute mode
  bark-infinity generate "Hello" --low-compute
  
  # Show version
  bark-infinity --version
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'Bark Infinity {__version__}'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser(
        'generate',
        help='Generate audio from text'
    )
    generate_parser.add_argument(
        'text',
        help='Text to convert to audio'
    )
    generate_parser.add_argument(
        '--output', '-o',
        default='output.wav',
        help='Output file path (default: output.wav)'
    )
    generate_parser.add_argument(
        '--voice',
        help='Voice preset to use (e.g., v2/en_speaker_1)'
    )
    generate_parser.add_argument(
        '--low-compute',
        action='store_true',
        help='Enable low-compute mode with optimizations'
    )
    
    # Web UI command
    webui_parser = subparsers.add_parser(
        'webui',
        help='Start Gradio web interface'
    )
    webui_parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the web UI on (default: 7860)'
    )
    webui_parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public share link'
    )
    
    # Streamlit command
    streamlit_parser = subparsers.add_parser(
        'streamlit',
        help='Start Streamlit web interface'
    )
    streamlit_parser.add_argument(
        '--port',
        type=int,
        default=8501,
        help='Port to run Streamlit on (default: 8501)'
    )
    
    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show system and configuration information'
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Handle commands
    if args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'webui':
        return cmd_webui(args)
    elif args.command == 'streamlit':
        return cmd_streamlit(args)
    elif args.command == 'info':
        return cmd_info(args)
    
    return 0


def cmd_generate(args):
    """Handle generate command."""
    try:
        from bark_infinity import generate_audio, SAMPLE_RATE
        from scipy.io.wavfile import write as write_wav
        
        print(f"Generating audio for: {args.text}")
        
        if args.low_compute:
            print("Enabling low-compute mode...")
            try:
                from bark_infinity import setup_low_compute_mode
                config = setup_low_compute_mode()
            except ImportError:
                print("Warning: Quantization dependencies not installed.")
                print("Install with: pip install bark-infinity[quantization]")
                print("Continuing without quantization optimizations...")
        
        # Generate audio
        audio_array = generate_audio(
            args.text,
            history_prompt=args.voice
        )
        
        # Save to file
        write_wav(args.output, SAMPLE_RATE, audio_array)
        
        print(f"✓ Audio saved to: {args.output}")
        return 0
        
    except ImportError as e:
        print(f"Error: Missing dependencies. Install with: pip install bark-infinity")
        print(f"Details: {e}")
        return 1
    except Exception as e:
        print(f"Error generating audio: {e}")
        return 1


def cmd_webui(args):
    """Handle webui command."""
    try:
        import subprocess
        import os
        
        # Find bark_webui.py
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        webui_path = os.path.join(script_dir, 'bark_webui.py')
        
        if not os.path.exists(webui_path):
            print(f"Error: bark_webui.py not found at {webui_path}")
            return 1
        
        print(f"Starting Gradio web UI on port {args.port}...")
        print(f"Open your browser to: http://localhost:{args.port}")
        
        env = os.environ.copy()
        env['GRADIO_SERVER_PORT'] = str(args.port)
        if args.share:
            env['GRADIO_SHARE'] = '1'
        
        subprocess.run([sys.executable, webui_path], env=env)
        return 0
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"Error starting web UI: {e}")
        return 1


def cmd_streamlit(args):
    """Handle streamlit command."""
    try:
        import subprocess
        import os
        
        # Find bark_streamlit.py
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        streamlit_path = os.path.join(script_dir, 'bark_streamlit.py')
        
        if not os.path.exists(streamlit_path):
            print(f"Error: bark_streamlit.py not found at {streamlit_path}")
            return 1
        
        print(f"Starting Streamlit UI on port {args.port}...")
        print(f"Open your browser to: http://localhost:{args.port}")
        
        subprocess.run([
            'streamlit', 'run',
            streamlit_path,
            '--server.port', str(args.port)
        ])
        return 0
        
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except FileNotFoundError:
        print("Error: Streamlit not found. Install with: pip install streamlit")
        return 1
    except Exception as e:
        print(f"Error starting Streamlit: {e}")
        return 1


def cmd_info(args):
    """Handle info command."""
    import platform
    import os
    
    print("=" * 60)
    print(f"Bark Infinity v{__version__}")
    print("=" * 60)
    print()
    
    # System info
    print("System Information:")
    print(f"  OS: {platform.system()} {platform.release()}")
    print(f"  Python: {platform.python_version()}")
    print(f"  Platform: {platform.platform()}")
    print()
    
    # Check dependencies
    print("Dependencies:")
    
    try:
        import torch
        print(f"  ✓ PyTorch: {torch.__version__}")
        print(f"    CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"    CUDA version: {torch.version.cuda}")
            print(f"    GPU count: {torch.cuda.device_count()}")
    except ImportError:
        print("  ✗ PyTorch: Not installed")
    
    try:
        import transformers
        print(f"  ✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("  ✗ Transformers: Not installed")
    
    try:
        import gradio
        print(f"  ✓ Gradio: {gradio.__version__}")
    except ImportError:
        print("  ✗ Gradio: Not installed")
    
    try:
        import streamlit
        print(f"  ✓ Streamlit: {streamlit.__version__}")
    except ImportError:
        print("  ✗ Streamlit: Not installed")
    
    print()
    
    # Quantization support
    print("Quantization Support:")
    
    try:
        import bitsandbytes
        print(f"  ✓ bitsandbytes: {bitsandbytes.__version__}")
    except ImportError:
        print("  ✗ bitsandbytes: Not installed (for 8-bit/4-bit quantization)")
    
    try:
        import optimum
        print(f"  ✓ optimum: {optimum.__version__}")
    except ImportError:
        print("  ✗ optimum: Not installed (for BetterTransformer)")
    
    print()
    
    # Environment variables
    print("Environment Configuration:")
    env_vars = [
        'SUNO_OFFLOAD_CPU',
        'SUNO_USE_SMALL_MODELS',
        'SUNO_ENABLE_MPS',
        'BARK_QUANTIZE_8BIT',
        'BARK_QUANTIZE_4BIT',
        'BARK_USE_BETTER_TRANSFORMER',
        'HF_HOME',
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '(not set)')
        print(f"  {var}: {value}")
    
    print()
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
