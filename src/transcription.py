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
                enable_diarization=enable_diarization,
                audio_duration_seconds=audio_metadata.get('duration_seconds', 0)
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
                # Create a clean version without speaker labels for API calls
                clean_transcript = self._strip_timestamps_and_labels(formatted_transcript)
            elif model == "openai/whisper-large-v3":
                formatted_transcript = self._format_whisper_output(transcript_result['raw_transcript'])
                clean_transcript = formatted_transcript
            else:
                formatted_transcript = self._format_parakeet_output(transcript_result['raw_transcript'])
                clean_transcript = formatted_transcript
            
            # Create successful response
            result = {
                'success': True,
                'transcript': transcript_result['raw_transcript'],
                'formatted_transcript': formatted_transcript,
                'clean_transcript': clean_transcript,  # Version without timestamps/labels for AI processing
                'model': model,
                'model_name': model_config['name'],
                'audio_metadata': audio_metadata,
                'timestamp': datetime.now().isoformat(),
                'errors': transcript_result.get('warnings'),
                'word_count': len(clean_transcript.split()),
                'char_count': len(clean_transcript)
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
    
    def _run_transcription(self, audio_path: str, model_config: Dict, enable_diarization: bool = False, audio_duration_seconds: float = 0) -> Dict[str, Any]:
        """
        Run the actual transcription using NVIDIA Riva.
        
        Args:
            audio_path: Path to WAV audio file
            model_config: Model configuration dictionary
            enable_diarization: Whether speaker diarization is enabled
            audio_duration_seconds: Duration of audio in seconds for timeout calculation
        
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
            
            # Set timeout based on audio duration and whether diarization is enabled
            # Rule: audio_duration * 2 (minimum 10 min, maximum 2 hours)
            # Diarization adds significant overhead, so use longer timeout
            if audio_duration_seconds > 0:
                if enable_diarization:
                    # Diarization is 3-4x slower
                    timeout_seconds = min(max(audio_duration_seconds * 4, 600), 7200)
                else:
                    # Regular transcription is about 2x slower than real-time
                    timeout_seconds = min(max(audio_duration_seconds * 2, 600), 7200)
            else:
                # Fallback if duration unknown
                timeout_seconds = 3600 if enable_diarization else 1800
            
            logger.info(f"Transcription timeout set to {timeout_seconds/60:.1f} minutes")
            
            result = subprocess.run(
                cmd_args,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout_seconds
            )
            
            # Check if output contains speaker labels
            raw_output = result.stdout.strip()
            # Riva diarization output contains "Transcript N:" and "Speaker." in metadata
            has_speakers = enable_diarization and ('Transcript' in raw_output and 'Speaker' in raw_output)
            
            if enable_diarization:
                logger.info(f"Diarization output detection: has_speakers={has_speakers}")
            
            return {
                'success': True,
                'raw_transcript': raw_output,
                'has_speakers': has_speakers,
                'warnings': result.stderr if result.stderr else None
            }
            
        except subprocess.TimeoutExpired:
            timeout_min = timeout_seconds / 60 if 'timeout_seconds' in locals() else (60 if enable_diarization else 30)
            error_msg = f"Transcription timed out after {timeout_min:.1f} minutes. The audio file may be too long or try again later."
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
        Extracts ONLY the clean text from "Transcript N: text." sections.
        
        Args:
            text: Raw transcription text with speaker markers and timestamps
            speaker_labels: Dictionary mapping speaker IDs to labels
        
        Returns:
            str: Formatted transcript with speaker labels and clean text
        """
        import re
        
        logger.info("Parsing diarized transcript with speaker labels")
        
        # Strategy: Find all "Transcript N: actual_clean_text." patterns
        # These contain the final transcribed text without timestamps
        # Format: "Transcript 0: i also have methods in a. Timestamps:. Word Start..."
        
        # Split by "Transcript" to get segments
        segments_raw = re.split(r'Transcript \d+:', text)
        
        parsed_segments = []
        
        for segment in segments_raw:
            if not segment.strip() or len(segment) < 5:
                continue
            
            # The clean text is everything between start and "Timestamps:"
            # Extract text before "Timestamps:"
            before_timestamps = segment.split('Timestamps:')[0].strip()
            
            # Remove any remaining "Time X.XXs:" markers
            before_timestamps = re.sub(r'Time \d+\.\d+s:', '', before_timestamps)
            before_timestamps = re.sub(r'>>>+', '', before_timestamps)
            
            # Clean up and get just the sentence
            # The actual transcript is usually at the start, ending with a period
            lines = before_timestamps.split('\n')
            for line in lines:
                line = line.strip()
                # Look for actual sentences (not metadata)
                if line and not re.match(r'^(Word|Start|End|Speaker)', line) and len(line) > 3:
                    # Remove trailing period for now
                    clean_line = line.rstrip('.')
                    if clean_line:
                        # Now find speaker ID from the metadata section after "Timestamps:"
                        speaker_id = 0  # default
                        
                        # Look for speaker in the full segment
                        if 'Timestamps:' in segment:
                            metadata = segment.split('Timestamps:')[1] if len(segment.split('Timestamps:')) > 1 else ""
                            # Find first occurrence of pattern "word start_ms end_ms speaker_id"
                            speaker_match = re.search(r'\w+\s+\d+\s+\d+\s+(\d+)', metadata)
                            if speaker_match:
                                speaker_id = int(speaker_match.group(1))
                        
                        parsed_segments.append((speaker_id, clean_line))
                        break  # Only take first valid line per segment
        
        if not parsed_segments:
            logger.error("Failed to parse any transcript segments!")
            logger.debug(f"Raw text sample: {text[:500]}...")
            return self._format_parakeet_output(text)
        
        logger.info(f"Successfully parsed {len(parsed_segments)} transcript segments")
        
        # Group by speaker and format
        formatted_lines = []
        current_speaker = None
        current_texts = []
        
        for speaker_id, text_content in parsed_segments:
            if speaker_id != current_speaker:
                # Save previous speaker's text
                if current_speaker is not None and current_texts:
                    speaker_label = speaker_labels.get(current_speaker, f"Speaker {current_speaker}")
                    combined = ' '.join(current_texts)
                    formatted_lines.append(f"\n**{speaker_label}**: {combined}.")
                
                # Start new speaker
                current_speaker = speaker_id
                current_texts = [text_content]
            else:
                # Same speaker, accumulate
                current_texts.append(text_content)
        
        # Add final speaker
        if current_speaker is not None and current_texts:
            speaker_label = speaker_labels.get(current_speaker, f"Speaker {current_speaker}")
            combined = ' '.join(current_texts)
            formatted_lines.append(f"\n**{speaker_label}**: {combined}.")
        
        result = '\n'.join(formatted_lines).strip()
        
        # Log reduction
        original_len = len(text)
        formatted_len = len(result)
        reduction = ((original_len - formatted_len) / original_len * 100) if original_len > 0 else 0
        logger.info(f"Formatting reduced size: {original_len} → {formatted_len} chars ({reduction:.1f}% reduction)")
        
        return result
    
    def _strip_timestamps_and_labels(self, text: str) -> str:
        """
        Remove speaker labels, timestamps, and metadata from formatted transcript for AI processing.
        This creates a clean continuous text version with ONLY the spoken words.
        
        Args:
            text: Formatted transcript with speaker labels
        
        Returns:
            str: Clean text without labels, timestamps, or metadata
        """
        import re
        
        # Remove speaker labels like "**Professor**:" or "**Students**:"
        cleaned = re.sub(r'\*\*[^*]+\*\*:\s*', '', text)
        
        # Remove any remaining markdown formatting
        cleaned = re.sub(r'\*\*([^*]+)\*\*', r'\1', cleaned)
        
        # Remove ALL timestamp formats
        cleaned = re.sub(r'>>>Time \d+\.\d+s:\s*', ' ', cleaned)
        cleaned = re.sub(r'Time \d+\.\d+s:\s*', ' ', cleaned)
        cleaned = re.sub(r'\[\d+:\d+:\d+\]', '', cleaned)
        cleaned = re.sub(r'>>>+', ' ', cleaned)
        
        # Remove transcript markers
        cleaned = re.sub(r'Transcript \d+:', '', cleaned)
        
        # Remove timestamp metadata sections
        cleaned = re.sub(r'Timestamps?:\.?\s*', '', cleaned)
        cleaned = re.sub(r'Word Start \(ms\) End \(ms\) Speaker\.?', '', cleaned)
        
        # Remove word timing data (format: "word 1234 5678 0")
        cleaned = re.sub(r'\b\w+\s+\d{3,}\s+\d{3,}\s+\d+\.?\b', '', cleaned)
        
        # Remove standalone numbers that look like timestamps
        cleaned = re.sub(r'\b\d{3,}\s+\d{3,}\s+\d+\.?\b', '', cleaned)
        
        # Remove "Speaker N" patterns
        cleaned = re.sub(r'Speaker\.?\s*\d+', '', cleaned, flags=re.IGNORECASE)
        
        # Clean up multiple spaces and newlines
        cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)  # Remove double+ newlines
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Normalize all spaces
        cleaned = re.sub(r'\.+', '.', cleaned)  # Remove multiple periods
        
        result = cleaned.strip()
        
        # Log the reduction
        original_length = len(text)
        cleaned_length = len(result)
        reduction_pct = ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        logger.info(f"Cleaned transcript: {original_length} → {cleaned_length} chars ({reduction_pct:.1f}% reduction)")
        
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
