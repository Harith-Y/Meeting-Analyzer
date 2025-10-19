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

model = st.selectbox("Select the model for transcription", options=["nvidia/parakeet-ctc-1.1b-asr", "openai/whisper-large-v3"])

if st.button("Analyze Meeting"):
    if audio:
        with st.spinner("Analyzing meeting..."):
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
                st.subheader("Transcription")
                st.write(result['formatted_transcript'])
                
                if result['errors']:
                    st.warning("Warnings/Info:")
                    st.text(result['errors'])
                
                # Generate summary
                if result['transcript']:
                    with st.spinner("Generating summary..."):
                        summary = generate_summary(result['transcript'], openrouter_api_key)
                        
                        if summary:
                            st.subheader("Meeting Summary")
                            st.write(summary)
            else:
                st.error("Transcription failed")
                if result and result['errors']:
                    st.text(result['errors'])
    else:
        st.error("Please upload an audio file to analyze.")