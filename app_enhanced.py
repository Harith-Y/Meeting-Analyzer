"""
Enhanced Streamlit Application for Class Lecture Transcription and Summarization
"""
import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    UI_SETTINGS, TRANSCRIPTION_MODELS, PROCESSING_SETTINGS,
    NVIDIA_API_KEY, OPENROUTER_API_KEY
)
from src.transcription import TranscriptionEngine
from src.summarization import SummaryGenerator
from src.file_exporter import FileExporter
from src.audio_processor import check_ffmpeg_installed
from src.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title=UI_SETTINGS['page_title'],
    page_icon=UI_SETTINGS['page_icon'],
    layout=UI_SETTINGS['layout'],
    initial_sidebar_state=UI_SETTINGS['sidebar_state']
)

# Initialize session state
if 'transcript_result' not in st.session_state:
    st.session_state.transcript_result = None
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


def main():
    """Main application function"""
    
    # Title and description
    st.title("🎓 Class Lecture Transcription System")
    st.markdown("""
    Transform your class recordings into comprehensive study materials for exam preparation.
    Upload your lecture audio, get accurate transcripts, and generate AI-powered summaries.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key status
        st.subheader("API Keys")
        if NVIDIA_API_KEY:
            st.success("✅ NVIDIA API Key loaded")
        else:
            st.error("❌ NVIDIA API Key missing")
        
        if OPENROUTER_API_KEY:
            st.success("✅ OpenRouter API Key loaded")
        else:
            st.error("❌ OpenRouter API Key missing")
        
        # FFmpeg status
        st.subheader("System Check")
        if check_ffmpeg_installed():
            st.success("✅ FFmpeg installed")
        else:
            st.warning("⚠️ FFmpeg not found")
            st.caption("Required for converting MP3/M4A files")
        
        # Model selection
        st.subheader("Model Selection")
        
        transcription_model = st.selectbox(
            "Transcription Model",
            options=list(TRANSCRIPTION_MODELS.keys()),
            format_func=lambda x: TRANSCRIPTION_MODELS[x]['name'],
            help="Choose the model for speech-to-text conversion"
        )
        
        # Display model info
        model_info = TRANSCRIPTION_MODELS[transcription_model]
        st.caption(f"📝 {model_info['description']}")
        st.caption(f"✨ Best for: {model_info['best_for']}")
        
        # Summary options
        st.subheader("Summary Options")
        
        summary_type = st.selectbox(
            "Summary Type",
            options=['class_lecture', 'brief_summary', 'detailed_notes'],
            format_func=lambda x: {
                'class_lecture': '📚 Comprehensive Study Guide',
                'brief_summary': '📋 Brief Summary',
                'detailed_notes': '📖 Detailed Notes'
            }[x],
            help="Choose the type of summary to generate"
        )
        
        include_key_points = st.checkbox("Extract Key Points", value=True)
        include_exam_questions = st.checkbox("Generate Exam Questions", value=True)
        
        # Export options
        st.subheader("Export Options")
        auto_export = st.checkbox("Auto-export results", value=True)
        export_formats = st.multiselect(
            "Export formats",
            options=['TXT', 'Markdown', 'JSON'],
            default=['TXT', 'Markdown']
        )
    
    # Main content area
    st.divider()
    
    # File upload section
    st.header("📁 Upload Audio File")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        audio_file = st.file_uploader(
            "Select your class recording",
            type=["mp3", "wav", "m4a", "flac", "ogg"],
            help=f"Maximum file size: {PROCESSING_SETTINGS['max_file_size_mb']} MB"
        )
    
    with col2:
        if audio_file:
            st.audio(audio_file, format="audio/wav")
    
    # Display file info
    if audio_file:
        file_size_mb = audio_file.size / (1024 * 1024)
        file_ext = audio_file.name.split('.')[-1].lower()
        
        st.info(f"""
        **📄 File:** {audio_file.name}  
        **💾 Size:** {file_size_mb:.2f} MB  
        **🎵 Format:** .{file_ext}
        """)
        
        if file_ext != 'wav':
            st.caption("ℹ️ File will be converted to WAV format for optimal transcription")
        
        # Check file size
        if file_size_mb > PROCESSING_SETTINGS['max_file_size_mb']:
            st.error(f"⚠️ File size exceeds {PROCESSING_SETTINGS['max_file_size_mb']} MB limit")
            return
    
    st.divider()
    
    # Processing button
    if audio_file:
        process_button = st.button(
            "🚀 Start Processing",
            type="primary",
            use_container_width=True
        )
        
        if process_button:
            process_lecture(
                audio_file,
                transcription_model,
                summary_type,
                include_key_points,
                include_exam_questions,
                auto_export,
                export_formats
            )
    else:
        st.info("👆 Please upload an audio file to begin")
    
    # Display results if available
    if st.session_state.processing_complete:
        display_results(
            st.session_state.transcript_result,
            st.session_state.summary_result,
            include_key_points,
            include_exam_questions
        )


def process_lecture(
    audio_file,
    transcription_model: str,
    summary_type: str,
    include_key_points: bool,
    include_exam_questions: bool,
    auto_export: bool,
    export_formats: list
):
    """Process the lecture recording"""
    
    logger.info(f"Starting lecture processing: {audio_file.name}")
    
    # Reset session state
    st.session_state.processing_complete = False
    st.session_state.transcript_result = None
    st.session_state.summary_result = None
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Transcription
        status_text.text("🎤 Transcribing audio... This may take several minutes.")
        progress_bar.progress(10)
        
        transcription_engine = TranscriptionEngine(NVIDIA_API_KEY)
        
        def update_progress(message):
            status_text.text(f"🎤 {message}")
        
        transcript_result = transcription_engine.transcribe(
            audio_file,
            model=transcription_model,
            progress_callback=update_progress
        )
        
        if not transcript_result['success']:
            st.error("❌ Transcription failed")
            st.error(transcript_result['errors'])
            logger.error(f"Transcription failed: {transcript_result['errors']}")
            return
        
        progress_bar.progress(50)
        st.success("✅ Transcription completed!")
        
        # Step 2: Summary Generation
        status_text.text("🤖 Generating AI summary...")
        progress_bar.progress(60)
        
        summary_generator = SummaryGenerator(OPENROUTER_API_KEY)
        
        summary_result = summary_generator.generate_summary(
            transcript_result['formatted_transcript'],
            summary_type=summary_type
        )
        
        if not summary_result['success']:
            st.warning("⚠️ Summary generation failed, but transcript is available")
            logger.warning(f"Summary failed: {summary_result.get('errors')}")
        else:
            st.success("✅ Summary generated!")
        
        progress_bar.progress(75)
        
        # Step 3: Additional features
        key_points_result = None
        exam_questions_result = None
        
        if include_key_points:
            status_text.text("🔑 Extracting key points...")
            key_points_result = summary_generator.generate_key_points(
                transcript_result['formatted_transcript']
            )
            progress_bar.progress(85)
        
        if include_exam_questions:
            status_text.text("📝 Generating exam questions...")
            exam_questions_result = summary_generator.generate_exam_questions(
                transcript_result['formatted_transcript']
            )
            progress_bar.progress(90)
        
        # Step 4: Export if enabled
        if auto_export and summary_result['success']:
            status_text.text("💾 Exporting results...")
            
            exporter = FileExporter()
            base_filename = exporter.generate_filename(
                base_name=os.path.splitext(audio_file.name)[0]
            )
            
            exported_files = exporter.export_complete_session(
                transcript_result,
                summary_result,
                base_filename
            )
            
            st.success(f"✅ Exported {len(exported_files)} files!")
            logger.info(f"Exported files: {exported_files}")
        
        progress_bar.progress(100)
        status_text.text("✨ Processing complete!")
        
        # Store results in session state
        st.session_state.transcript_result = transcript_result
        st.session_state.summary_result = summary_result
        st.session_state.key_points_result = key_points_result
        st.session_state.exam_questions_result = exam_questions_result
        st.session_state.processing_complete = True
        
        logger.info("Lecture processing completed successfully")
        
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        logger.error(f"Processing error: {str(e)}", exc_info=True)
    
    finally:
        progress_bar.empty()
        status_text.empty()


def display_results(
    transcript_result: dict,
    summary_result: dict,
    show_key_points: bool,
    show_exam_questions: bool
):
    """Display processing results"""
    
    st.divider()
    st.header("📊 Results")
    
    # Tabs for different views
    tabs = st.tabs(["📝 Transcript", "📚 Summary", "🔑 Key Points", "❓ Exam Questions", "📈 Metadata"])
    
    # Transcript tab
    with tabs[0]:
        st.subheader("Transcription")
        
        if transcript_result and transcript_result.get('formatted_transcript'):
            st.markdown(transcript_result['formatted_transcript'])
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", transcript_result.get('word_count', 0))
            with col2:
                st.metric("Characters", transcript_result.get('char_count', 0))
            with col3:
                duration = transcript_result.get('audio_metadata', {}).get('duration_formatted', 'N/A')
                st.metric("Duration", duration)
            
            # Download button
            st.download_button(
                "⬇️ Download Transcript",
                transcript_result['formatted_transcript'],
                file_name=f"transcript_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No transcript available")
    
    # Summary tab
    with tabs[1]:
        st.subheader("AI-Generated Summary")
        
        if summary_result and summary_result.get('summary'):
            st.markdown(summary_result['summary'])
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", summary_result.get('word_count', 0))
            with col2:
                st.metric("Summary Type", summary_result.get('summary_type', 'N/A'))
            
            # Download button
            st.download_button(
                "⬇️ Download Summary",
                summary_result['summary'],
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
        else:
            st.info("No summary available")
    
    # Key Points tab
    with tabs[2]:
        st.subheader("Key Points")
        
        if show_key_points and hasattr(st.session_state, 'key_points_result'):
            key_points_result = st.session_state.key_points_result
            if key_points_result and key_points_result.get('success'):
                st.markdown(key_points_result.get('key_points_text', ''))
            else:
                st.info("Key points extraction was not enabled")
        else:
            st.info("Key points extraction was not enabled")
    
    # Exam Questions tab
    with tabs[3]:
        st.subheader("Potential Exam Questions")
        
        if show_exam_questions and hasattr(st.session_state, 'exam_questions_result'):
            exam_questions_result = st.session_state.exam_questions_result
            if exam_questions_result and exam_questions_result.get('success'):
                st.markdown(exam_questions_result.get('questions', ''))
            else:
                st.info("Exam question generation was not enabled")
        else:
            st.info("Exam question generation was not enabled")
    
    # Metadata tab
    with tabs[4]:
        st.subheader("Processing Metadata")
        
        if transcript_result:
            st.json({
                'Transcription': {
                    'Model': transcript_result.get('model_name', 'N/A'),
                    'Timestamp': transcript_result.get('timestamp', 'N/A'),
                    'Word Count': transcript_result.get('word_count', 0),
                    'Audio Duration': transcript_result.get('audio_metadata', {}).get('duration_formatted', 'N/A')
                },
                'Summary': {
                    'Model': summary_result.get('model_name', 'N/A') if summary_result else 'N/A',
                    'Type': summary_result.get('summary_type', 'N/A') if summary_result else 'N/A',
                    'Word Count': summary_result.get('word_count', 0) if summary_result else 0
                }
            })


if __name__ == "__main__":
    main()
