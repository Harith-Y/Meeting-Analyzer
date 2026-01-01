"""
Class Lecture Transcription System - Source Package

This package contains all the core modules for transcribing and summarizing
class lecture recordings to help students prepare for exams.

Modules:
    - logger: Logging and error tracking
    - audio_processor: Audio file validation and conversion
    - transcription: Speech-to-text transcription engine
    - summarization: AI-powered summary generation
    - file_exporter: Export results to multiple formats
"""

__version__ = "2.0.0"
__author__ = "Your Name"
__description__ = "Class Lecture Transcription and Summarization System"

# Import main classes for easy access
from .transcription import TranscriptionEngine
from .summarization import SummaryGenerator
from .file_exporter import FileExporter
from .audio_processor import AudioProcessor
from .logger import setup_logger

__all__ = [
    'TranscriptionEngine',
    'SummaryGenerator',
    'FileExporter',
    'AudioProcessor',
    'setup_logger',
]
