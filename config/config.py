"""
Configuration settings for Class Lecture Transcription System
"""
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Detect if running in Streamlit Cloud or deployment environment
IS_CLOUD_DEPLOYMENT = os.getenv("STREAMLIT_SHARING_MODE") or os.getenv("IS_DEPLOYMENT", "false").lower() == "true"

# Base paths
BASE_DIR = Path(__file__).parent.parent

# For cloud deployment, use system temp directory
if IS_CLOUD_DEPLOYMENT:
    TEMP_DIR = Path(tempfile.gettempdir()) / "meeting_analyzer"
    OUTPUTS_DIR = TEMP_DIR / "outputs"
    LOGS_DIR = TEMP_DIR / "logs"
else:
    OUTPUTS_DIR = BASE_DIR / "outputs"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"

# Create directories if they don't exist (with error handling for read-only filesystems)
try:
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
except (OSError, PermissionError):
    # If we can't create directories (read-only filesystem), use temp directory
    TEMP_DIR = Path(tempfile.gettempdir()) / "meeting_analyzer"
    OUTPUTS_DIR = TEMP_DIR / "outputs"
    LOGS_DIR = TEMP_DIR / "logs"
    TEMP_DIR.mkdir(exist_ok=True, parents=True)
    OUTPUTS_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

# API Keys - Support both .env and Streamlit secrets
def get_api_key(key_name: str, streamlit_key: str = None) -> str:
    """Get API key from environment or Streamlit secrets"""
    # First try environment variable
    env_value = os.getenv(key_name)
    if env_value:
        return env_value
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and (streamlit_key or key_name) in st.secrets:
            return st.secrets.get(streamlit_key or key_name, "")
    except (ImportError, FileNotFoundError, KeyError):
        pass
    
    return ""

NVIDIA_API_KEY = get_api_key("NVIDIA_API_KEY")
OPENROUTER_API_KEY = get_api_key("OPENROUTER_API_KEY")
GROQ_API_KEY = get_api_key("GROQ_API_KEY")

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
    # Groq Models (Fast, Free, Reliable - RECOMMENDED)
    "groq:llama-3.3-70b-versatile": {
        "name": "Groq: Llama 3.3 70B (Free, Fast)",
        "max_tokens": 8192,
        "description": "FREE & FAST - Best choice for quick, reliable summaries (Recommended)"
    },
    "groq:llama-3.1-70b-versatile": {
        "name": "Groq: Llama 3.1 70B (Free)",
        "max_tokens": 8192,
        "description": "Free alternative with good quality"
    },
    "groq:mixtral-8x7b-32768": {
        "name": "Groq: Mixtral 8x7B (Free)",
        "max_tokens": 32768,
        "description": "Free with very large context window"
    },
    # OpenRouter Free Models (Fallback)
    "openrouter:nousresearch/hermes-3-llama-3.1-405b:free": {
        "name": "Hermes 3 Llama 405B (Free)",
        "max_tokens": 8192,
        "description": "Free OpenRouter model (may be rate-limited)"
    },
    "openrouter:microsoft/phi-3-mini-128k-instruct:free": {
        "name": "Microsoft Phi-3 Mini (Free)",
        "max_tokens": 8192,
        "description": "Free Microsoft model with large context"
    },
    # Legacy (for backward compatibility)
    "meta-llama/llama-3.2-3b-instruct:free": {
        "name": "Llama 3.2 3B (Free, Legacy)",
        "max_tokens": 8192,
        "description": "Older free model - use Groq models instead"
    },
    # Paid Options (Better Quality)
    "openrouter:meta-llama/llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B (Paid)",
        "max_tokens": 16384,
        "description": "Advanced with comprehensive output (requires credits)"
    },
    "openrouter:anthropic/claude-3.5-sonnet": {
        "name": "Claude 3.5 Sonnet (Paid)",
        "max_tokens": 16384,
        "description": "Best quality with most detailed output (requires credits)"
    }
}

# Speaker Diarization Settings
DIARIZATION_SETTINGS = {
    "enabled": False,
    "max_speakers": 2,
    "speaker_labels": {
        0: "Professor",
        1: "Students"
    },
    "format_timestamps": True,
    "include_confidence": False
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
    "class_lecture": """You are an expert educational assistant and academic tutor helping students prepare for exams. Your goal is to create an extremely comprehensive and detailed study guide that captures ALL important information from the lecture.

IMPORTANT: Be thorough and detailed. Don't summarize - EXPAND and EXPLAIN everything discussed in the lecture.

Analyze this class lecture transcript and provide an extensive study guide with the following sections:

## 1. LECTURE OVERVIEW
- Course/Subject context (if mentioned)
- Date and topic of lecture
- Main themes and objectives
- How this lecture connects to previous/future topics

## 2. COMPREHENSIVE TOPIC BREAKDOWN
For EACH major topic discussed, provide:
- Detailed explanation of the concept
- Why it's important
- How it relates to other concepts
- Any subtopics or related ideas
- Context and background information

## 3. KEY CONCEPTS & DEFINITIONS (Detailed)
For EACH important term or concept:
- Clear, detailed definition
- Multiple examples if provided
- Practical applications
- Common misconceptions or clarifications
- Related terminology

## 4. DETAILED EXPLANATIONS & REASONING
- Elaborate on all explanations provided by the instructor
- Include all reasoning, logic, and thought processes discussed
- Document step-by-step processes or methodologies
- Capture all "why" and "how" explanations

## 5. EXAMPLES, CASE STUDIES & ILLUSTRATIONS
For EACH example mentioned:
- Describe the example in full detail
- Explain what concept it illustrates
- Break down the analysis or solution
- Note any variations or alternative approaches discussed

## 6. FORMULAS, EQUATIONS & TECHNICAL DETAILS
- List all formulas, equations, or technical procedures
- Explain when and how to use each one
- Include any special cases or limitations
- Note any tips or tricks mentioned

## 7. IMPORTANT FACTS & POINTS TO MEMORIZE
- All specific facts, dates, names, or figures mentioned
- Critical information emphasized by the instructor
- Information explicitly stated as "important for the exam"
- Any repetitions or emphasized points

## 8. CONNECTIONS & RELATIONSHIPS
- How different concepts connect to each other
- Cause-and-effect relationships
- Comparisons and contrasts made
- Hierarchies or categorizations discussed

## 9. INSTRUCTOR'S INSIGHTS & COMMENTARY
- Personal insights or perspectives shared
- Real-world applications mentioned
- Career or practical relevance discussed
- Any anecdotes or stories that illustrate concepts

## 10. POTENTIAL EXAM QUESTIONS (Detailed)
Create 5-8 comprehensive exam questions covering:
- Multiple choice questions with detailed explanations
- Short answer questions
- Essay/discussion questions
- Problem-solving questions (if applicable)
- Include sample answers or answer guidelines

## 11. STUDY RECOMMENDATIONS
- Prioritized list of what to focus on
- Suggested review activities
- Connections to textbook or readings (if mentioned)
- Topics that need extra practice or review
- Study techniques specific to this material

## 12. SUMMARY OF KEY TAKEAWAYS
- Comprehensive list of 10-15 key takeaways
- Most critical information to remember
- Main learning objectives achieved

Transcript:
{transcript}

---

Provide an EXTREMELY DETAILED and COMPREHENSIVE study guide. Your goal is to help the student understand everything as if they attended the lecture themselves. Be thorough, clear, and organized. Include ALL information from the transcript, not just highlights.""",
    
    "brief_summary": """Provide a structured overview of this class lecture, focusing on:

**Main Topics Covered:**
- List all major topics discussed in order
- Include subtopics and their relationships

**Key Concepts & Definitions:**
- Define all important terms with clear explanations
- Provide context for each concept

**Critical Takeaways:**
- Most important points students must remember
- Information likely to appear on exams
- Practical applications mentioned

**Notable Examples:**
- Key examples used to illustrate concepts
- Their significance and what they demonstrate

Transcript:
{transcript}

Be thorough but organized. Include all essential information.""",
    
    "detailed_notes": """Create extremely detailed and comprehensive study notes from this lecture. Your goal is to capture EVERYTHING discussed:

**COMPLETE TOPIC COVERAGE:**
- Document every topic, subtopic, and related concept
- Include full explanations, not summaries
- Preserve the logical flow and structure of the lecture
- Note connections between topics

**DEFINITIONS & EXPLANATIONS:**
- Provide complete definitions for all terminology
- Include detailed explanations of concepts
- Document all reasoning and logic discussed
- Add context and background information

**EXAMPLES & ILLUSTRATIONS:**
- Describe each example in full detail
- Explain what each example demonstrates
- Include any variations or alternative scenarios
- Note instructor's analysis and insights

**FORMULAS & PROCEDURES:**
- List all formulas, equations, or procedures
- Explain when and how to apply them
- Include any special cases or conditions
- Note any tips or shortcuts mentioned

**IMPORTANT FACTS & DETAILS:**
- All specific facts, figures, names, or dates
- Information emphasized by the instructor
- Points repeated or highlighted
- Any exam-relevant information mentioned

**INSTRUCTOR INSIGHTS:**
- Commentary and perspectives shared
- Real-world applications discussed
- Practical relevance and context
- Any stories or anecdotes that aid understanding

Transcript:
{transcript}

---

Create notes that are so detailed a student who missed the lecture could learn everything from them. Be thorough, clear, and preserve all important information."""
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
