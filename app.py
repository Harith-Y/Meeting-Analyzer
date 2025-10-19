import streamlit as st
import os
import subprocess

import requests
import json

import dotenv 
dotenv.load_dotenv()

from transcript_gen import generate_transcript

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

model = st.selectbox("Enter the model ID for transcription", options=["openai/whisper-large-v3", "nvidia/parakeet-ctc-1.1b-asr"])

if st.button("Analyze Meeting"):
    if audio:
        st.success("Analyzing meeting...")

        if model == "openai/whisper-large-v3":
            generate_transcript(audio, nvidia_api_key, "b702f636-f60c-4a3d-a6f4-f3568c13bd7d", "en", "transcribe_file_offline.py")
        
        elif model == "nvidia/parakeet-ctc-1.1b-asr":
            generate_transcript(audio, nvidia_api_key, "1598d209-5e27-4d3c-8079-4751568b1081", "en-US", "transcribe_file.py")

    else:
        st.error("Please upload an audio file to analyze.")