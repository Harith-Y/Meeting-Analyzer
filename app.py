import streamlit as st
import os
import subprocess
from huggingface_hub import InferenceClient

import dotenv 
dotenv.load_dotenv()

from transcript_gen import generate_transcript

st.title("Meeting Analyzer")
st.write("Welcome to the Meeting Analyzer app!")

hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
nvidia_api_key = os.getenv("NVIDIA_API_KEY")

if hf_api_key:
    st.success("Hugging Face API Key loaded successfully.")
else:
    st.error("Hugging Face API Key not found. Please set it in the .env file.")

if nvidia_api_key:
    st.success("NVIDIA API Key loaded successfully.")
else:
    st.warning("NVIDIA API Key not found. Required for NVIDIA models.")

audio = st.file_uploader("Upload your meeting audio file", type=["mp3", "wav", "m4a"])
if audio:
    st.audio(audio, format="audio/wav")

model = st.selectbox("Enter the model ID for transcription", options=["openai/whisper-large-v3", "nvidia/parakeet-ctc-1.1b-asr"])

if st.button("Analyze Meeting"):
    if audio:
        st.success("Analyzing meeting...")

        if model == "openai/whisper-large-v3":
            pass
        
        elif model == "nvidia/parakeet-ctc-1.1b-asr":
            pass
    else:
        st.error("Please upload an audio file to analyze.")