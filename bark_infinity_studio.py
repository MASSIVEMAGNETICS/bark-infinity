"""
Windows 10 Application - Bark Infinity Studio
Enterprise-grade GUI for voice cloning and audio generation
"""

import gradio as gr
import os
import sys
from pathlib import Path
from typing import Optional, List, Tuple
import numpy as np
from scipy.io.wavfile import write as write_wav

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bark_infinity import config
from bark_infinity.windows_runtime import (
    get_runtime,
    shutdown_runtime,
    GenerationTask,
    MultiThreadedRuntime
)
from bark_infinity.voice_studio import (
    get_voice_studio,
    VoiceStudio,
    VoiceModel
)
from bark_infinity import generation
from bark_infinity import api

logger = config.logger


class BarkInfinityStudioApp:
    """
    Windows 10 Application for Bark Infinity
    """
    
    def __init__(self):
        self.runtime: Optional[MultiThreadedRuntime] = None
        self.studio: Optional[VoiceStudio] = None
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
    
    def initialize(self):
        """Initialize runtime and studio"""
        if self.runtime is None:
            self.runtime = get_runtime()
        if self.studio is None:
            self.studio = get_voice_studio()
        logger.info("Bark Infinity Studio initialized")
    
    def cleanup(self):
        """Cleanup resources"""
        if self.runtime:
            shutdown_runtime()
        logger.info("Bark Infinity Studio cleaned up")
    
    def generate_audio_multi_threaded(self, text: str, voice_model_id: str = None,
                                     temperature: float = 0.7, 
                                     progress=gr.Progress()) -> Tuple[str, str]:
        """
        Generate audio using multi-threaded runtime
        """
        try:
            progress(0, desc="Initializing...")
            self.initialize()
            
            # Get voice model if specified
            history_prompt = None
            if voice_model_id and voice_model_id != "None":
                progress(0.1, desc="Loading voice model...")
                voice_model = self.studio.get_model(voice_model_id)
                if voice_model:
                    history_prompt = {
                        'semantic_prompt': voice_model.semantic_prompt,
                        'coarse_prompt': voice_model.coarse_prompt,
                        'fine_prompt': voice_model.fine_prompt
                    }
            
            # Create generation task
            progress(0.2, desc="Submitting task...")
            task = GenerationTask(
                task_id=f"task_{api.generate_unique_dirpath('').split('/')[-1]}",
                text=text,
                history_prompt=history_prompt,
                temp=temperature
            )
            
            task_id = self.runtime.submit_task(task)
            
            # Wait for result with progress updates
            progress(0.3, desc="Generating audio...")
            result = self.runtime.get_result(task_id, timeout=300)
            
            if result and result.success:
                progress(0.9, desc="Saving audio...")
                
                # Save audio file
                output_path = self.temp_dir / f"{task_id}.wav"
                write_wav(str(output_path), result.sample_rate, result.audio_array)
                
                # Create info text
                info = f"""
                Generation completed successfully!
                
                Chunks processed: {result.metadata.get('chunks', 1)}
                Duration: {result.metadata.get('duration_seconds', 0):.2f} seconds
                Processing time: {result.metadata.get('processing_time', 0):.2f} seconds
                Text length: {result.metadata.get('text_length', 0)} characters
                Sample rate: {result.sample_rate} Hz
                """
                
                progress(1.0, desc="Done!")
                return str(output_path), info
            else:
                error_msg = result.error if result else "Unknown error"
                return None, f"Generation failed: {error_msg}"
                
        except Exception as e:
            logger.error(f"Error in generation: {e}")
            return None, f"Error: {str(e)}"
    
    def import_voice_model(self, file_path: str, name: str, 
                          transcription: str = "", progress=gr.Progress()) -> str:
        """Import voice model"""
        try:
            progress(0, desc="Importing voice model...")
            self.initialize()
            
            if not file_path:
                return "Error: No file selected"
            
            if not name:
                name = Path(file_path).stem
            
            progress(0.5, desc="Processing model...")
            model_id = self.studio.import_voice_model(
                file_path,
                name=name,
                transcription=transcription if transcription else None
            )
            
            progress(1.0, desc="Done!")
            return f"Successfully imported voice model: {name} (ID: {model_id})"
            
        except Exception as e:
            logger.error(f"Error importing voice model: {e}")
            return f"Error: {str(e)}"
    
    def export_voice_model(self, model_id: str, output_format: str,
                          progress=gr.Progress()) -> Tuple[str, str]:
        """Export voice model"""
        try:
            progress(0, desc="Exporting voice model...")
            self.initialize()
            
            if not model_id or model_id == "None":
                return None, "Error: No model selected"
            
            # Create output path
            model = self.studio.get_model(model_id)
            if not model:
                return None, f"Error: Model {model_id} not found"
            
            output_path = self.temp_dir / f"{model.name}_export.{output_format}"
            
            progress(0.5, desc="Converting model...")
            self.studio.export_voice_model(model_id, str(output_path), output_format)
            
            progress(1.0, desc="Done!")
            return str(output_path), f"Successfully exported to {output_path}"
            
        except Exception as e:
            logger.error(f"Error exporting voice model: {e}")
            return None, f"Error: {str(e)}"
    
    def list_voice_models(self) -> List[List[str]]:
        """List all voice models"""
        try:
            self.initialize()
            models = self.studio.list_models()
            
            # Format for display
            model_list = []
            for model in models:
                model_list.append([
                    model['id'],
                    model['name'],
                    model['format'],
                    model['created_at'][:19]  # Truncate timestamp
                ])
            
            return model_list
            
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def get_voice_model_choices(self) -> List[str]:
        """Get voice model choices for dropdown"""
        try:
            self.initialize()
            models = self.studio.list_models()
            choices = ["None"] + [f"{m['name']} ({m['id']})" for m in models]
            return choices
        except Exception as e:
            logger.error(f"Error getting model choices: {e}")
            return ["None"]
    
    def analyze_voice_model(self, model_id: str, progress=gr.Progress()) -> str:
        """Analyze voice model quality"""
        try:
            progress(0, desc="Analyzing voice model...")
            self.initialize()
            
            if not model_id or model_id == "None":
                return "Error: No model selected"
            
            progress(0.5, desc="Running analysis...")
            analysis = self.studio.analyze_model(model_id)
            
            # Format analysis results
            result = f"""
            Voice Model Analysis Report
            {'=' * 50}
            
            Model: {analysis['model_name']}
            ID: {analysis['model_id']}
            Timestamp: {analysis['timestamp']}
            
            Overall Quality Score: {analysis['overall_quality']:.2%}
            
            """
            
            if 'semantic' in analysis:
                sem = analysis['semantic']
                result += f"""
            Semantic Layer:
            - Diversity Score: {sem['diversity_score']:.2%}
            - Entropy: {sem['entropy']:.2f}
            - Unique Tokens: {sem['unique_tokens']}
            - Total Tokens: {sem['total_tokens']}
            """
            
            if 'coarse' in analysis:
                coarse = analysis['coarse']
                result += f"""
            Coarse Layer:
            - Quality Score: {coarse['quality_score']:.2%}
            - Mean: {coarse['mean']:.2f}
            - Std Dev: {coarse['std']:.2f}
            """
            
            if 'fine' in analysis:
                fine = analysis['fine']
                result += f"""
            Fine Layer:
            - Detail Score: {fine['detail_score']:.2%}
            - Variance: {fine['variance']:.2f}
            """
            
            progress(1.0, desc="Done!")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing model: {e}")
            return f"Error: {str(e)}"
    
    def delete_voice_model(self, model_id: str) -> str:
        """Delete voice model"""
        try:
            self.initialize()
            
            if not model_id or model_id == "None":
                return "Error: No model selected"
            
            self.studio.delete_model(model_id)
            return f"Successfully deleted voice model: {model_id}"
            
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return f"Error: {str(e)}"
    
    def create_ui(self) -> gr.Blocks:
        """Create Gradio UI"""
        
        with gr.Blocks(
            title="Bark Infinity Studio - Windows 10",
            theme=gr.themes.Soft(),
            css="""
            .main-header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 20px;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 20px;
            }
            .feature-box {
                border: 2px solid #667eea;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
            }
            """
        ) as demo:
            
            gr.Markdown(
                """
                <div class="main-header">
                    <h1>🎙️ Bark Infinity Studio</h1>
                    <h3>Enterprise-Grade Voice Cloning & Audio Generation</h3>
                    <p>Multi-threaded | CPU-Optimized | Professional Voice Studio</p>
                </div>
                """
            )
            
            with gr.Tabs():
                
                # Audio Generation Tab
                with gr.Tab("🎵 Audio Generation"):
                    gr.Markdown("### Multi-threaded Audio Generation")
                    gr.Markdown("Generate audio using the advanced multi-threaded runtime engine")
                    
                    with gr.Row():
                        with gr.Column(scale=2):
                            gen_text = gr.Textbox(
                                label="Text to Generate",
                                placeholder="Enter text to convert to speech...",
                                lines=5
                            )
                            
                            gen_voice = gr.Dropdown(
                                label="Voice Model",
                                choices=self.get_voice_model_choices(),
                                value="None"
                            )
                            
                            gen_temp = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.7,
                                step=0.1,
                                label="Temperature (Creativity)"
                            )
                            
                            gen_button = gr.Button("🎵 Generate Audio", variant="primary")
                        
                        with gr.Column(scale=1):
                            gen_audio = gr.Audio(label="Generated Audio")
                            gen_info = gr.Textbox(label="Generation Info", lines=10)
                    
                    gen_button.click(
                        fn=self.generate_audio_multi_threaded,
                        inputs=[gen_text, gen_voice, gen_temp],
                        outputs=[gen_audio, gen_info]
                    )
                
                # Voice Studio Tab
                with gr.Tab("🎙️ Voice Studio"):
                    gr.Markdown("### Comprehensive Voice Model Management")
                    gr.Markdown("Import, export, analyze, and manage voice models")
                    
                    with gr.Row():
                        with gr.Column():
                            gr.Markdown("#### Import Voice Model")
                            import_file = gr.File(label="Voice Model File (.npz, .pkl, .json)")
                            import_name = gr.Textbox(label="Model Name")
                            import_trans = gr.Textbox(label="Transcription (optional)", lines=3)
                            import_button = gr.Button("📥 Import Model")
                            import_result = gr.Textbox(label="Import Result")
                            
                            import_button.click(
                                fn=self.import_voice_model,
                                inputs=[import_file, import_name, import_trans],
                                outputs=import_result
                            )
                        
                        with gr.Column():
                            gr.Markdown("#### Export Voice Model")
                            export_model = gr.Dropdown(
                                label="Select Model",
                                choices=self.get_voice_model_choices()
                            )
                            export_format = gr.Radio(
                                label="Export Format",
                                choices=["npz", "pkl", "json"],
                                value="npz"
                            )
                            export_button = gr.Button("📤 Export Model")
                            export_file = gr.File(label="Exported File")
                            export_result = gr.Textbox(label="Export Result")
                            
                            export_button.click(
                                fn=self.export_voice_model,
                                inputs=[export_model, export_format],
                                outputs=[export_file, export_result]
                            )
                    
                    gr.Markdown("#### Voice Model Library")
                    refresh_button = gr.Button("🔄 Refresh List")
                    model_list = gr.Dataframe(
                        headers=["ID", "Name", "Format", "Created"],
                        label="Voice Models",
                        interactive=False
                    )
                    
                    refresh_button.click(
                        fn=self.list_voice_models,
                        outputs=model_list
                    )
                
                # Voice Analysis Tab
                with gr.Tab("📊 Voice Analysis"):
                    gr.Markdown("### Advanced Voice Quality Analysis")
                    gr.Markdown("Analyze voice models using cutting-edge algorithms")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            analyze_model = gr.Dropdown(
                                label="Select Model to Analyze",
                                choices=self.get_voice_model_choices()
                            )
                            analyze_button = gr.Button("🔬 Analyze Model", variant="primary")
                        
                        with gr.Column(scale=2):
                            analyze_result = gr.Textbox(
                                label="Analysis Report",
                                lines=20,
                                max_lines=30
                            )
                    
                    analyze_button.click(
                        fn=self.analyze_voice_model,
                        inputs=analyze_model,
                        outputs=analyze_result
                    )
                
                # Management Tab
                with gr.Tab("⚙️ Management"):
                    gr.Markdown("### Voice Model Management")
                    
                    with gr.Row():
                        delete_model = gr.Dropdown(
                            label="Select Model to Delete",
                            choices=self.get_voice_model_choices()
                        )
                        delete_button = gr.Button("🗑️ Delete Model", variant="stop")
                    
                    delete_result = gr.Textbox(label="Result")
                    
                    delete_button.click(
                        fn=self.delete_voice_model,
                        inputs=delete_model,
                        outputs=delete_result
                    )
                    
                    gr.Markdown("---")
                    gr.Markdown(
                        """
                        ### About Bark Infinity Studio
                        
                        **Enterprise Features:**
                        - ✅ Multi-threaded processing engine
                        - ✅ CPU-optimized chunked generation
                        - ✅ Layered generational architecture
                        - ✅ Multi-format voice model support (NPZ, PKL, JSON)
                        - ✅ Advanced quality analysis algorithms
                        - ✅ Professional voice model library
                        - ✅ Batch processing capabilities
                        - ✅ Windows 10 optimized runtime
                        
                        **Version:** 1.0.0  
                        **Platform:** Windows 10  
                        **Engine:** Bark Infinity Multi-threaded Runtime
                        """
                    )
            
            # Load initial data
            demo.load(
                fn=self.list_voice_models,
                outputs=model_list
            )
            
            return demo


def main():
    """Main entry point for Windows 10 application"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Bark Infinity Studio - Windows 10 Application"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public shareable link"
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to"
    )
    
    args = parser.parse_args()
    
    # Create and launch app
    app = BarkInfinityStudioApp()
    
    try:
        demo = app.create_ui()
        
        logger.info("=" * 60)
        logger.info("Bark Infinity Studio - Windows 10")
        logger.info("Enterprise-Grade Voice Cloning & Audio Generation")
        logger.info("=" * 60)
        logger.info(f"Starting server on {args.server_name}:{args.port}")
        
        demo.launch(
            server_name=args.server_name,
            server_port=args.port,
            share=args.share,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        app.cleanup()
    except Exception as e:
        logger.error(f"Error: {e}")
        app.cleanup()
        raise


if __name__ == "__main__":
    main()
