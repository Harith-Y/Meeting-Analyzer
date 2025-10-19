import streamlit as st
import requests
import json

def generate_summary(transcript, openrouter_api_key):
    """
    Generate a summary of the meeting transcript using OpenRouter API.
    
    Args:
        transcript: The full meeting transcript text
        openrouter_api_key: API key for OpenRouter service
    
    Returns:
        str: Summary of the meeting
    """
         
    st.subheader("Meeting Summary")
            
    response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer " + openrouter_api_key,
        "Content-Type": "application/json",
    },
    data=json.dumps({
        "model": "deepseek/deepseek-chat-v3.1:free",
        "messages": [
        {
            "role": "user",
            "content": "Summarize the following meeting transcript:\n\n" + transcript
        }
        ],
        
    })
    )
    
    # Extract the summary from the JSON response
    if response.status_code == 200:
        response_data = response.json()
        summary = response_data["choices"][0]["message"]["content"]
        st.write(summary)
    else:
        st.error(f"Error getting summary: {response.status_code}")
        st.text(response.text)