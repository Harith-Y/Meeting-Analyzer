"""
Enhanced audio processing module with better conversion and validation
"""
import ffmpeg
import os
from pathlib import Path
from typing import Optional, Dict, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import AUDIO_SETTINGS, TEMP_DIR
from src.logger import setup_logger

logger = setup_logger(__name__)


class AudioProcessor:
    """Handle audio file conversion and validation"""
    
    def __init__(self):
        self.supported_formats = AUDIO_SETTINGS["supported_formats"]
        self.sample_rate = AUDIO_SETTINGS["sample_rate"]
        self.channels = AUDIO_SETTINGS["channels"]
        self.codec = AUDIO_SETTINGS["codec"]
    
    def validate_audio_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate audio file format and get metadata.
        
        Args:
            file_path: Path to audio file
        
        Returns:
            dict: Audio file metadata including duration, format, size
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            # Get file extension
            file_ext = os.path.splitext(file_path)[1].lower().replace('.', '')
            
            # Check if format is supported
            if file_ext not in self.supported_formats:
                raise ValueError(
                    f"Unsupported format: .{file_ext}. "
                    f"Supported formats: {', '.join(['.' + fmt for fmt in self.supported_formats])}"
                )
            
            # Get file size
            file_size_bytes = os.path.getsize(file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
            
            # Get audio metadata using ffprobe
            try:
                probe = ffmpeg.probe(file_path)
                audio_info = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
                
                duration_seconds = float(probe['format'].get('duration', 0))
                duration_minutes = duration_seconds / 60
                
                metadata = {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'format': file_ext,
                    'size_mb': round(file_size_mb, 2),
                    'duration_seconds': round(duration_seconds, 2),
                    'duration_minutes': round(duration_minutes, 2),
                    'duration_formatted': self._format_duration(duration_seconds),
                    'sample_rate': audio_info.get('sample_rate', 'Unknown'),
                    'channels': audio_info.get('channels', 'Unknown'),
                    'codec': audio_info.get('codec_name', 'Unknown'),
                    'bit_rate': audio_info.get('bit_rate', 'Unknown')
                }
                
                logger.info(f"Audio file validated: {metadata['file_name']} - "
                          f"{metadata['duration_formatted']} - {metadata['size_mb']} MB")
                
                return metadata
                
            except Exception as e:
                logger.warning(f"Could not get detailed metadata: {str(e)}")
                # Return basic metadata if ffprobe fails
                return {
                    'file_path': file_path,
                    'file_name': os.path.basename(file_path),
                    'format': file_ext,
                    'size_mb': round(file_size_mb, 2),
                    'duration_seconds': 0,
                    'duration_minutes': 0,
                    'duration_formatted': 'Unknown',
                    'sample_rate': 'Unknown',
                    'channels': 'Unknown',
                    'codec': 'Unknown',
                    'bit_rate': 'Unknown'
                }
        
        except Exception as e:
            logger.error(f"Error validating audio file: {str(e)}")
            raise
    
    def convert_to_wav(self, input_file: str, output_file: Optional[str] = None) -> str:
        """
        Convert audio files to WAV format optimized for transcription.
        
        Args:
            input_file: Path to input audio file
            output_file: Optional path to output WAV file
        
        Returns:
            str: Path to the WAV file
        """
        try:
            file_ext = os.path.splitext(input_file)[1].lower()
            
            # If already WAV, return the original path
            if file_ext == '.wav':
                logger.info(f"File is already in WAV format: {input_file}")
                return input_file
            
            # Generate output filename if not provided
            if output_file is None:
                output_file = os.path.splitext(input_file)[0] + '.wav'
            
            logger.info(f"Converting {input_file} to WAV format...")
            
            # Convert using FFmpeg with optimized settings for speech recognition
            (
                ffmpeg
                .input(input_file)
                .output(
                    output_file,
                    acodec=self.codec,
                    ac=self.channels,
                    ar=self.sample_rate
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )
            
            logger.info(f"Conversion successful: {output_file}")
            return output_file
            
        except ffmpeg.Error as e:
            error_message = e.stderr.decode() if e.stderr else str(e)
            logger.error(f"FFmpeg error during conversion: {error_message}")
            raise ValueError(f"Audio conversion failed: {error_message}")
        
        except Exception as e:
            logger.error(f"Error converting audio: {str(e)}")
            raise
    
    def _format_duration(self, seconds: float) -> str:
        """
        Format duration in seconds to HH:MM:SS.
        
        Args:
            seconds: Duration in seconds
        
        Returns:
            str: Formatted duration string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def estimate_processing_time(self, duration_minutes: float, model: str = "nvidia/parakeet-ctc-1.1b-asr") -> str:
        """
        Estimate processing time based on audio duration.
        
        Args:
            duration_minutes: Audio duration in minutes
            model: Transcription model being used
        
        Returns:
            str: Estimated processing time description
        """
        # Rough estimates based on model
        if "parakeet" in model.lower():
            # Parakeet is faster, roughly 1:1 or better
            estimated_minutes = duration_minutes * 1.2
        else:
            # Whisper is slower, roughly 1:2 or 1:3
            estimated_minutes = duration_minutes * 2.5
        
        if estimated_minutes < 1:
            return "Less than 1 minute"
        elif estimated_minutes < 60:
            return f"Approximately {int(estimated_minutes)} minutes"
        else:
            hours = estimated_minutes / 60
            return f"Approximately {hours:.1f} hours"


def check_ffmpeg_installed() -> bool:
    """
    Check if FFmpeg is installed and accessible.
    
    Returns:
        bool: True if FFmpeg is installed, False otherwise
    """
    import subprocess
    
    try:
        subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            check=True
        )
        logger.info("FFmpeg is installed and accessible")
        return True
    except FileNotFoundError:
        logger.error("FFmpeg is not installed or not in PATH")
        return False
    except Exception as e:
        logger.error(f"Error checking FFmpeg: {str(e)}")
        return False


if __name__ == "__main__":
    # Test the audio processor
    processor = AudioProcessor()
    print("Audio Processor initialized successfully")
    print(f"Supported formats: {', '.join(['.' + fmt for fmt in processor.supported_formats])}")
    print(f"FFmpeg installed: {check_ffmpeg_installed()}")
