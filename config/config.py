"""
Configuration settings for Class Lecture Transcription System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist
OUTPUTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# API Keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Transcription Models Configuration
TRANSCRIPTION_MODELS = {
    "nvidia/parakeet-ctc-1.1b-asr": {
        "name": "NVIDIA Parakeet (Fast)",
        "function_id": "1598d209-5e27-4d3c-8079-4751568b1081",
        "language_code": "en-US",
        "client_file": "transcribe_file.py",
        "description": "High-performance real-time transcription",
        "best_for": "Quick transcriptions, real-time processing"
    },
    "openai/whisper-large-v3": {
        "name": "OpenAI Whisper Large V3 (Accurate)",
        "function_id": "b702f636-f60c-4a3d-a6f4-f3568c13bd7d",
        "language_code": "en",
        "client_file": "transcribe_file_offline.py",
        "description": "Industry-leading accuracy for complex audio",
        "best_for": "High-quality transcriptions, lectures with technical terms"
    }
}

# Summary Models Configuration
SUMMARY_MODELS = {
    "meta-llama/llama-3.2-3b-instruct:free": {
        "name": "Llama 3.2 3B (Free)",
        "max_tokens": 4096,
        "description": "Free, reliable summarization (Recommended)"
    },
    "google/gemini-2.0-flash-exp:free": {
        "name": "Google Gemini 2.0 Flash (Free)",
        "max_tokens": 8192,
        "description": "Free, fast (may be rate-limited)"
    },
    "nousresearch/hermes-3-llama-3.1-405b:free": {
        "name": "Hermes 3 Llama 405B (Free)",
        "max_tokens": 4096,
        "description": "Free, very capable model"
    },
    "microsoft/phi-3-mini-128k-instruct:free": {
        "name": "Microsoft Phi-3 Mini (Free)",
        "max_tokens": 4096,
        "description": "Free Microsoft model"
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B (Paid)",
        "max_tokens": 8192,
        "description": "Advanced, more reliable (requires credits)"
    },
    "anthropic/claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet (Paid)",
        "max_tokens": 8192,
        "description": "Best quality (requires credits)"
    }
}

# Audio Processing Settings
AUDIO_SETTINGS = {
    "supported_formats": ["mp3", "wav", "m4a", "flac", "ogg"],
    "convert_to_wav": True,
    "sample_rate": 16000,
    "channels": 1,
    "codec": "pcm_s16le"
}

# Summary Prompt Templates
SUMMARY_PROMPTS = {
    "class_lecture": """You are an expert educational assistant helping students prepare for exams.

Analyze this class lecture transcript and provide a comprehensive study guide with:

1. **Main Topics Covered**: List all major topics and concepts discussed
2. **Key Concepts & Definitions**: Define important terms and concepts
3. **Important Points**: Highlight critical information students need to remember
4. **Examples & Explanations**: Summarize any examples or detailed explanations given
5. **Potential Exam Questions**: Suggest 3-5 questions that might appear on an exam based on this content
6. **Study Tips**: Brief recommendations on how to study this material

Transcript:
{transcript}

Provide a clear, well-organized summary that helps with exam preparation.""",
    
    "brief_summary": """Provide a concise summary of this class lecture, focusing on:
- Main topics covered
- Key takeaways
- Important concepts to remember

Transcript:
{transcript}""",
    
    "detailed_notes": """Create detailed study notes from this lecture including:
- All topics covered with timestamps if available
- Definitions and explanations
- Examples and case studies
- Formulas or important facts
- Connections between concepts

Transcript:
{transcript}"""
}

# Export Settings
EXPORT_SETTINGS = {
    "default_format": "txt",
    "include_timestamp": True,
    "include_metadata": True,
    "pdf_font": "Helvetica",
    "pdf_font_size": 11
}

# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "simple": {
            "format": "%(levelname)s - %(message)s"
        }
    },
    "handlers": {
        "file": {
            "class": "logging.FileHandler",
            "filename": str(LOGS_DIR / "app.log"),
            "formatter": "detailed",
            "level": "INFO"
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "simple",
            "level": "INFO"
        }
    },
    "root": {
        "level": "INFO",
        "handlers": ["file", "console"]
    }
}

# UI Settings
UI_SETTINGS = {
    "page_title": "Class Lecture Transcription System",
    "page_icon": "🎓",
    "layout": "wide",
    "sidebar_state": "expanded"
}

# Processing Settings
PROCESSING_SETTINGS = {
    "max_file_size_mb": 500,  # Maximum file size in MB
    "chunk_duration_minutes": 30,  # For very long files, process in chunks
    "enable_progress_bar": True,
    "retry_attempts": 2,
    "timeout_seconds": 600  # 10 minutes timeout
}
