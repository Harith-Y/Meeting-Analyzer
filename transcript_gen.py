import subprocess
import os
from file_to_wav import convert_to_wav

def format_output_parakeet(text):
    """
    Format the transcription output text for better readability.
    Converts line-by-line actual-formatted-output (with ## markers) into a clean paragraph.
    Each original line becomes a sentence ending with a period.
    
    Args:
        text: Raw transcription text from NVIDIA Riva
    
    Returns:
        str: Formatted paragraph text with proper punctuation
    """
    # Remove ## markers and strip whitespace from each line
    lines = [line.strip().replace('##', '').strip() for line in text.split('\n')]
    
    # Filter out empty lines
    lines = [line for line in lines if line]
    
    # Add period to end of each line if not already present
    sentences = []
    for line in lines:
        if line and not line.endswith(('.', '!', '?')):
            sentences.append(line + '.')
        elif line:
            sentences.append(line)
    
    # Join all sentences into a single paragraph with spaces
    summary = ' '.join(sentences)
    
    # Clean up extra spaces
    summary = ' '.join(summary.split())
    
    # Capitalize the first letter if needed
    if summary:
        summary = summary[0].upper() + summary[1:]
    
    return summary

def format_output_whisper(text):
    """
    Format the transcription output text for better readability.
    Converts continuous text into properly capitalized sentences.
    
    Args:
        text: Raw transcription text from NVIDIA Riva
    
    Returns:
        str: Formatted text with proper capitalization and punctuation
    """
    # Clean up the text
    text = text.strip()
    
    # Capitalize the first letter if needed
    if text:
        text = text[0].upper() + text[1:]
    
    # Ensure it ends with a period if it doesn't have ending punctuation
    if text and not text.endswith(('.', '!', '?')):
        text = text + '.'
    
    return text


def generate_transcript(audio, api_key, function_id, language_code, client_file, model):
    """
    Generate transcript using NVIDIA Riva ASR service.
    
    Args:
        audio: Streamlit UploadedFile object
        api_key: NVIDIA API key
        function_id: NVIDIA function ID for the model
        language_code: Language code for transcription
        client_file: Name of the transcription script file
        model: Model identifier string
    
    Returns:
        dict: Dictionary with 'transcript', 'formatted_transcript', and 'errors' keys
    """
    # Save the uploaded audio file temporarily
    temp_audio_path = f"temp_audio{os.path.splitext(audio.name)[1]}"
    with open(temp_audio_path, "wb") as f:
        f.write(audio.getvalue())
    
    # Convert to WAV if necessary
    wav_audio_path = temp_audio_path  # Default to original file
    conversion_attempted = False
    
    try:
        if not temp_audio_path.endswith('.wav'):
            wav_audio_path = convert_to_wav(temp_audio_path)
            conversion_attempted = True
    except Exception as e:
        # If conversion fails, provide helpful error message
        import streamlit as st
        error_msg = str(e)
        
        if "ffmpeg" in error_msg.lower() or "not found" in error_msg.lower():
            st.error("❌ FFmpeg is not installed or not in PATH")
            st.info("""
            **To fix this:**
            1. Close this Streamlit app
            2. Restart your terminal/PowerShell (to load new PATH)
            3. Verify FFmpeg: Run `ffmpeg -version`
            4. Restart Streamlit: `streamlit run app.py`
            
            **Alternative:** Upload a WAV file instead of MP3/M4A
            """)
            return {
                'transcript': None,
                'formatted_transcript': None,
                'errors': f"FFmpeg not found. Please install FFmpeg or use WAV files.",
                'success': False
            }
        else:
            st.warning(f"⚠️ Could not convert audio to WAV format: {error_msg}")
            st.info("Attempting transcription with original format...")
            wav_audio_path = temp_audio_path
    
    try:
        # Get the Python executable from the current environment
        import sys
        python_executable = sys.executable
        
        # Run the NVIDIA Riva transcription script using the current Python environment
        result = subprocess.run(
            [
                python_executable,
                f"python-clients/scripts/asr/{client_file}",
                "--server", "grpc.nvcf.nvidia.com:443",
                "--use-ssl",
                "--metadata", "function-id", f"{function_id}",
                "--metadata", "authorization", f"Bearer {api_key}",
                "--language-code", f"{language_code}",
                "--input-file", wav_audio_path
            ],
            capture_output=True,
            text=True,
            check=True
        )

        transcript = None
        formatted_transcript = None
        
        if model == "openai/whisper-large-v3":
            # Extract the final transcript from the output
            output_lines = result.stdout.strip().split('\n')
            
            # Look for the line that starts with "Final transcript:"
            for line in output_lines:
                if line.startswith("Final transcript:"):
                    transcript = line.replace("Final transcript:", "").strip()
                    break
            
            if transcript:
                formatted_transcript = format_output_whisper(transcript)
            else:
                transcript = result.stdout
                formatted_transcript = transcript
        
        elif model == "nvidia/parakeet-ctc-1.1b-asr":
            transcript = result.stdout
            formatted_transcript = format_output_parakeet(transcript)
        
        return {
            'transcript': transcript,
            'formatted_transcript': formatted_transcript,
            'errors': result.stderr if result.stderr else None,
            'success': True
        }
            
    except subprocess.CalledProcessError as e:
        return {
            'transcript': None,
            'formatted_transcript': None,
            'errors': f"Error running transcription: {e}\nError output: {e.stderr}",
            'success': False
        }
    except Exception as e:
        return {
            'transcript': None,
            'formatted_transcript': None,
            'errors': f"An error occurred: {str(e)}",
            'success': False
        }
    finally:
        # Clean up temporary files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        # If a WAV conversion was created, clean it up too
        if 'wav_audio_path' in locals() and wav_audio_path != temp_audio_path and os.path.exists(wav_audio_path):
            os.remove(wav_audio_path)


if __name__ == "__main__":
    print("This module should be imported, not run directly.")