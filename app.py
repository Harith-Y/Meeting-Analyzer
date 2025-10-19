import streamlit as st
import os

import dotenv 
dotenv.load_dotenv()

from transcript_gen import generate_transcript
from summary_gen import generate_summary

st.title("Meeting Analyzer")
st.write("Welcome to the Meeting Analyzer app!")

nvidia_api_key = os.getenv("NVIDIA_API_KEY")
openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

if nvidia_api_key and openrouter_api_key:
    st.success("NVIDIA API Key and OpenRouter API Key loaded successfully.")
else:
    st.warning("NVIDIA API Key or OpenRouter API Key not found. Required for NVIDIA models.")

audio = st.file_uploader("Upload your meeting audio file", type=["mp3", "wav", "m4a"])
if audio:
    st.audio(audio, format="audio/wav")
    
    # Show file info
    file_extension = audio.name.split('.')[-1].lower()
    file_size_mb = audio.size / (1024 * 1024)
    st.caption(f"📁 File: {audio.name} ({file_size_mb:.2f} MB)")
    
    if file_extension != 'wav':
        st.info(f"ℹ️ Audio will be automatically converted from .{file_extension} to .wav format before transcription.")

model = st.selectbox("Select the model for transcription", options=["nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3"])

if st.button("Analyze Meeting"):
    if audio:
        with st.spinner("Processing audio... This may take a few minutes."):
            # Generate transcript based on selected model
            if model == "openai/whisper-large-v3":
                result = generate_transcript(
                    audio, 
                    nvidia_api_key, 
                    "b702f636-f60c-4a3d-a6f4-f3568c13bd7d", 
                    "en", 
                    "transcribe_file_offline.py", 
                    "openai/whisper-large-v3"
                )
            
            elif model == "nvidia/parakeet-ctc-1.1b-asr":
                result = generate_transcript(
                    audio, 
                    nvidia_api_key, 
                    "1598d209-5e27-4d3c-8079-4751568b1081", 
                    "en-US", 
                    "transcribe_file.py", 
                    "nvidia/parakeet-ctc-1.1b-asr"
                )
            
            # Display transcription results
            if result and result['success']:
                st.success("✅ Transcription completed successfully!")
                st.subheader("Transcription")
                st.write(result['formatted_transcript'])
                
                if result['errors']:
                    with st.expander("View Warnings/Debug Info"):
                        st.text(result['errors'])
                
                # Generate summary
                if result['transcript']:
                    with st.spinner("Generating AI summary..."):
                        summary = generate_summary(result['transcript'], openrouter_api_key)
                        
                        if summary:
                            st.subheader("Meeting Summary")
                            st.write(summary)
                        else:
                            st.warning("Could not generate summary. Please check your OpenRouter API key.")
            else:
                st.error("❌ Transcription failed")
                if result and result['errors']:
                    st.error("Error details:")
                    st.text(result['errors'])
                    
                    # Provide helpful suggestions
                    if "DEADLINE_EXCEEDED" in result['errors'] or "failed to establish link" in result['errors']:
                        st.info("""
                        **Suggestions:**
                        - The NVIDIA service might be experiencing high load. Try again in a few minutes.
                        - Check your internet connection.
                        - Try using the other transcription model.
                        - For large audio files, try splitting them into smaller chunks.
                        """)
    else:
        st.error("Please upload an audio file to analyze.")