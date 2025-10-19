import ffmpeg
import os

def convert_to_wav(input_file):
    """
    Convert audio files to .wav format using FFmpeg.
    
    Args:
        input_file (str): Path to the input audio file (.mp3, .m4a, or .wav)
    
    Returns:
        str: Path to the .wav file (original if already .wav, converted otherwise)
    
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If input file doesn't exist
    """
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File not found: {input_file}")
    
    file_ext = os.path.splitext(input_file)[1].lower()
    
    # If already WAV, return the original path
    if file_ext == '.wav':
        return input_file
    
    # Generate output filename
    output_file = os.path.splitext(input_file)[0] + '.wav'
    
    try:
        # Use FFmpeg to convert the audio file to WAV
        (
            ffmpeg
            .input(input_file)
            .output(output_file, acodec='pcm_s16le', ac=1, ar='16k')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )
        
        return output_file
        
    except ffmpeg.Error as e:
        error_message = e.stderr.decode() if e.stderr else str(e)
        raise ValueError(f"FFmpeg error converting {file_ext} to WAV: {error_message}")
    except Exception as e:
        raise ValueError(f"Error converting {file_ext} to WAV: {str(e)}")


if __name__ == "__main__":
    print("This module should be imported, not run directly.")
    print("\nExample usage:")
    print("  from file_to_wav import convert_to_wav")
    print("  wav_path = convert_to_wav('audio.mp3')")