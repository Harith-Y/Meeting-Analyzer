"""
Enhanced transcription module with improved formatting and error handling
"""
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Any
import sys
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import TRANSCRIPTION_MODELS, NVIDIA_API_KEY, TEMP_DIR
from src.audio_processor import AudioProcessor
from src.logger import setup_logger

logger = setup_logger(__name__)


class TranscriptionEngine:
    """Handle audio transcription using NVIDIA Riva models"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or NVIDIA_API_KEY
        self.audio_processor = AudioProcessor()
        self.models = TRANSCRIPTION_MODELS
    
    def transcribe(
        self,
        audio_file,
        model: str = "nvidia/parakeet-ctc-1.1b-asr",
        progress_callback=None,
        enable_diarization: bool = False,
        speaker_labels: Optional[Dict[int, str]] = None
    ) -> Dict[str, Any]:
        """
        Transcribe audio file using specified model.
        
        Args:
            audio_file: Streamlit UploadedFile or file path
            model: Model identifier
            progress_callback: Optional callback function for progress updates
        
        Returns:
            dict: Transcription results with transcript, metadata, and status
        """
        logger.info(f"Starting transcription with model: {model}")
        
        # Validate API key
        if not self.api_key:
            error_msg = "NVIDIA API key not found. Please set NVIDIA_API_KEY in .env file."
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        # Validate model
        if model not in self.models:
            error_msg = f"Invalid model: {model}. Available models: {list(self.models.keys())}"
            logger.error(error_msg)
            return self._create_error_response(error_msg)
        
        model_config = self.models[model]
        temp_audio_path = None
        wav_audio_path = None
        
        try:
            # Save uploaded file temporarily
            if hasattr(audio_file, 'read'):  # Streamlit UploadedFile
                temp_audio_path = str(TEMP_DIR / f"temp_audio{os.path.splitext(audio_file.name)[1]}")
                with open(temp_audio_path, "wb") as f:
                    f.write(audio_file.getvalue())
                logger.info(f"Saved uploaded file to: {temp_audio_path}")
            else:  # File path
                temp_audio_path = str(audio_file)
            
            # Validate audio file
            if progress_callback:
                progress_callback("Validating audio file...")
            
            audio_metadata = self.audio_processor.validate_audio_file(temp_audio_path)
            
            # Convert to WAV if necessary
            if progress_callback:
                progress_callback("Converting audio format...")
            
            if not temp_audio_path.endswith('.wav'):
                wav_audio_path = self.audio_processor.convert_to_wav(temp_audio_path)
                logger.info(f"Audio converted to WAV: {wav_audio_path}")
            else:
                wav_audio_path = temp_audio_path
            
            # Transcribe audio
            if progress_callback:
                progress_callback(f"Transcribing audio using {model_config['name']}...")
            
            transcript_result = self._run_transcription(
                wav_audio_path,
                model_config,
                enable_diarization=enable_diarization
            )
            
            if not transcript_result['success']:
                return transcript_result
            
            # Format transcript
            if progress_callback:
                progress_callback("Formatting transcript...")
            
            if enable_diarization and transcript_result.get('has_speakers'):
                formatted_transcript = self._format_with_speakers(
                    transcript_result['raw_transcript'],
                    speaker_labels or {0: "Professor", 1: "Students"}
                )
            elif model == "openai/whisper-large-v3":
                formatted_transcript = self._format_whisper_output(transcript_result['raw_transcript'])
            else:
                formatted_transcript = self._format_parakeet_output(transcript_result['raw_transcript'])
            
            # Create successful response
            result = {
                'success': True,
                'transcript': transcript_result['raw_transcript'],
                'formatted_transcript': formatted_transcript,
                'model': model,
                'model_name': model_config['name'],
                'audio_metadata': audio_metadata,
                'timestamp': datetime.now().isoformat(),
                'errors': transcript_result.get('warnings'),
                'word_count': len(formatted_transcript.split()),
                'char_count': len(formatted_transcript)
            }
            
            logger.info(f"Transcription completed successfully. Word count: {result['word_count']}")
            return result
            
        except Exception as e:
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return self._create_error_response(error_msg)
        
        finally:
            # Cleanup temporary files
            self._cleanup_files([temp_audio_path, wav_audio_path])
    
    def _run_transcription(self, audio_path: str, model_config: Dict, enable_diarization: bool = False) -> Dict[str, Any]:
        """
        Run the actual transcription using NVIDIA Riva.
        
        Args:
            audio_path: Path to WAV audio file
            model_config: Model configuration dictionary
        
        Returns:
            dict: Raw transcription results
        """
        try:
            # Get Python executable
            python_executable = sys.executable
            
            # Prepare command
            client_script = f"python-clients/scripts/asr/{model_config['client_file']}"
            
            if not os.path.exists(client_script):
                raise FileNotFoundError(
                    f"Transcription client script not found: {client_script}\n"
                    "Please ensure python-clients directory is properly set up."
                )
            
            # Run transcription
            logger.info(f"Executing transcription command with {model_config['name']}")
            
            # Build command arguments
            cmd_args = [
                python_executable,
                client_script,
                "--server", "grpc.nvcf.nvidia.com:443",
                "--use-ssl",
                "--metadata", "function-id", model_config['function_id'],
                "--metadata", "authorization", f"Bearer {self.api_key}",
                "--language-code", model_config['language_code'],
                "--input-file", audio_path
            ]
            
            # Add diarization flags if enabled
            if enable_diarization:
                logger.info("Speaker diarization enabled with max 2 speakers")
                cmd_args.extend([
                    "--speaker-diarization",
                    "--diarization-max-speakers", "2"
                ])
            
            # Set timeout based on whether diarization is enabled
            # Diarization takes significantly longer, especially for long files
            timeout_seconds = 3600 if enable_diarization else 600  # 60 min with diarization, 10 min without
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_seconds
            )
            
            # Check if output contains speaker labels
            raw_output = result.stdout.strip()
            has_speakers = enable_diarization and ('[speaker' in raw_output.lower() or 'speaker_' in raw_output.lower())
            
            return {
                'success': True,
                'raw_transcript': raw_output,
                'has_speakers': has_speakers,
                'warnings': result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            timeout_min = 60 if enable_diarization else 10
            error_msg = f"Transcription timed out after {timeout_min} minutes. Try processing a shorter audio file or disable speaker diarization."
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        except subprocess.CalledProcessError as e:
            error_msg = f"Transcription process failed: {e.stderr}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        
        except Exception as e:
            error_msg = f"Unexpected error during transcription: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {'success': False, 'error': error_msg}
    
    def _format_parakeet_output(self, text: str) -> str:
        """
        Format Parakeet model output for readability.
        
        Args:
            text: Raw transcription text
        
        Returns:
            str: Formatted transcript
        """
        # Remove ## markers and strip whitespace
        lines = [line.strip().replace('##', '').strip() for line in text.split('\n')]
        
        # Filter out empty lines
        lines = [line for line in lines if line]
        
        # Add period to end of each line if not present
        sentences = []
        for line in lines:
            if line and not line.endswith(('.', '!', '?')):
                sentences.append(line + '.')
            elif line:
                sentences.append(line)
        
        # Join sentences
        formatted_text = ' '.join(sentences)
        
        # Clean up extra spaces
        formatted_text = ' '.join(formatted_text.split())
        
        # Capitalize first letter
        if formatted_text:
            formatted_text = formatted_text[0].upper() + formatted_text[1:]
        
        return formatted_text
    
    def _format_whisper_output(self, text: str) -> str:
        """
        Format Whisper model output for readability.
        
        Args:
            text: Raw transcription text
        
        Returns:
            str: Formatted transcript
        """
        # Extract final transcript if present
        for line in text.split('\n'):
            if line.startswith("Final transcript:"):
                text = line.replace("Final transcript:", "").strip()
                break
        
        # Clean up the text
        text = text.strip()
        
        # Capitalize first letter
        if text:
            text = text[0].upper() + text[1:]
        
        # Ensure it ends with punctuation
        if text and not text.endswith(('.', '!', '?')):
            text = text + '.'
        
        return text
    
    def _format_with_speakers(self, text: str, speaker_labels: Dict[int, str]) -> str:
        """
        Format transcript with speaker labels for diarization output.
        
        Args:
            text: Raw transcription text with speaker markers
            speaker_labels: Dictionary mapping speaker IDs to labels
        
        Returns:
            str: Formatted transcript with speaker labels
        """
        import re
        
        lines = text.split('\n')
        formatted_lines = []
        current_speaker = None
        current_text = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to detect speaker markers
            # Format 1: [speaker_0] or [speaker_1]
            speaker_match = re.match(r'\[speaker[_\s]*(\d+)\]\s*(.*)', line, re.IGNORECASE)
            if speaker_match:
                # Save previous speaker's text
                if current_speaker is not None and current_text:
                    speaker_label = speaker_labels.get(current_speaker, f"Speaker {current_speaker}")
                    text_content = ' '.join(current_text).strip()
                    if text_content:
                        formatted_lines.append(f"\n**{speaker_label}**: {text_content}")
                
                # Start new speaker
                current_speaker = int(speaker_match.group(1))
                text_part = speaker_match.group(2).strip()
                current_text = [text_part] if text_part else []
            
            # Format 2: speaker_0: or speaker_1:
            elif ':' in line:
                parts = line.split(':', 1)
                if 'speaker' in parts[0].lower():
                    speaker_num_match = re.search(r'(\d+)', parts[0])
                    if speaker_num_match:
                        # Save previous speaker's text
                        if current_speaker is not None and current_text:
                            speaker_label = speaker_labels.get(current_speaker, f"Speaker {current_speaker}")
                            text_content = ' '.join(current_text).strip()
                            if text_content:
                                formatted_lines.append(f"\n**{speaker_label}**: {text_content}")
                        
                        # Start new speaker
                        current_speaker = int(speaker_num_match.group(1))
                        current_text = [parts[1].strip()] if len(parts) > 1 and parts[1].strip() else []
                    else:
                        current_text.append(line)
                else:
                    current_text.append(line)
            else:
                # Continue current speaker's text
                current_text.append(line)
        
        # Add final speaker's text
        if current_speaker is not None and current_text:
            speaker_label = speaker_labels.get(current_speaker, f"Speaker {current_speaker}")
            text_content = ' '.join(current_text).strip()
            if text_content:
                formatted_lines.append(f"\n**{speaker_label}**: {text_content}")
        
        # If no speakers detected, fall back to regular formatting
        if not formatted_lines:
            logger.warning("No speaker labels detected in output, using standard formatting")
            return self._format_parakeet_output(text)
        
        result = '\n'.join(formatted_lines).strip()
        logger.info(f"Formatted transcript with {len(set([current_speaker]))} speakers")
        return result
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'success': False,
            'transcript': None,
            'formatted_transcript': None,
            'errors': error_message,
            'timestamp': datetime.now().isoformat()
        }
    
    def _cleanup_files(self, file_paths: list):
        """Clean up temporary files."""
        for file_path in file_paths:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
                    logger.debug(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file {file_path}: {str(e)}")
    
    def get_available_models(self) -> Dict[str, Dict]:
        """Get list of available transcription models."""
        return self.models


if __name__ == "__main__":
    # Test the transcription engine
    engine = TranscriptionEngine()
    print("Transcription Engine initialized successfully")
    print("\nAvailable models:")
    for model_id, config in engine.get_available_models().items():
        print(f"  - {config['name']} ({model_id})")
        print(f"    {config['description']}")
