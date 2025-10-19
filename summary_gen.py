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
        str: Summary of the meeting or None if error occurs
    """
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_api_key}",
                "Content-Type": "application/json",
            },
            data=json.dumps({
                "model": "deepseek/deepseek-chat-v3.1:free",
                "messages": [
                    {
                        "role": "user",
                        "content": f"Provide a clear summary of the following meeting transcript. Do not miss any important information. Focus on key topics discussed, decisions made, and action items if any:\n\n{transcript}"
                    }
                ],
            })
        )
        
        # Extract the summary from the JSON response
        if response.status_code == 200:
            response_data = response.json()
            summary = response_data["choices"][0]["message"]["content"]
            
            # Clean up any artifacts or special tokens
            summary = clean_summary(summary)
            
            return summary
        else:
            st.error(f"Error getting summary: {response.status_code}")
            st.text(response.text)
            return None
            
    except Exception as e:
        st.error(f"An error occurred while generating summary: {str(e)}")
        return None


def clean_summary(text):
    """
    Clean up the summary text by removing special tokens and artifacts.
    
    Args:
        text: Raw summary text
    
    Returns:
        str: Cleaned summary text
    """
    # Remove common AI model artifacts
    artifacts = [
        '<｜begin▁of▁sentence｜>',
        '<|begin_of_sentence|>',
        '<｜end▁of▁sentence｜>',
        '<|end_of_sentence|>',
        '<|im_start|>',
        '<|im_end|>',
    ]
    
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    # Remove extra whitespace and newlines
    text = text.strip()
    
    return text


if __name__ == "__main__":
    print("This module should be imported, not run directly.")