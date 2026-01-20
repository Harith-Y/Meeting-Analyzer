"""
Enhanced Streamlit Application for Class Lecture Transcription and Summarization
"""
import streamlit as st
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.config import (
    UI_SETTINGS, TRANSCRIPTION_MODELS, PROCESSING_SETTINGS,
    NVIDIA_API_KEY, OPENROUTER_API_KEY, GROQ_API_KEY
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
if 'audio_filename' not in st.session_state:
    st.session_state.audio_filename = None
if 'export_formats' not in st.session_state:
    st.session_state.export_formats = ['Markdown']
if 'key_points_result' not in st.session_state:
    st.session_state.key_points_result = None
if 'exam_questions_result' not in st.session_state:
    st.session_state.exam_questions_result = None
if 'meeting_minutes_result' not in st.session_state:
    st.session_state.meeting_minutes_result = None


def main():
    """Main application function"""
    
    # Title and description
    st.title("🎓 Class Lecture Transcription System")
    st.markdown("""
    Transform your class recordings into comprehensive study materials for exam preparation.
    Upload your lecture audio, get accurate transcripts, and generate AI-powered summaries.
    
    💡 **Tip:** For best results, keep recordings under 45 minutes. For longer files, use `split_audio.py` to divide them into segments.
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
        try:
            ffmpeg_available = check_ffmpeg_installed()
            if ffmpeg_available:
                st.success("✅ FFmpeg installed")
            else:
                st.warning("⚠️ FFmpeg not found")
                st.caption("Required for converting MP3/M4A files")
                st.info("💡 On Streamlit Cloud: Ensure `packages.txt` contains `ffmpeg` and redeploy")
        except Exception as e:
            st.error(f"⚠️ Error checking FFmpeg: {str(e)}")
            st.caption("Will attempt to use FFmpeg if available")
        
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
        
        # Speaker Diarization options
        st.subheader("🎤 Speaker Diarization")
        enable_diarization = st.checkbox(
            "Enable Speaker Separation",
            value=False,
            help="Separate Professor and Students in the transcript"
        )
        
        if enable_diarization:
            st.info("💡 The system will distinguish between 2 speakers: Professor and Students")
            st.warning("⏱️ Speaker diarization significantly increases processing time (up to 60 minutes for long lectures)")
            speaker_0_label = st.text_input("Speaker 0 Label", value="Professor", help="Usually the main instructor")
            speaker_1_label = st.text_input("Speaker 1 Label", value="Students", help="All student voices combined")
        else:
            speaker_0_label = "Professor"
            speaker_1_label = "Students"
        
        # Summary options
        st.subheader("Summary Options")
        
        summary_model = st.selectbox(
            "Summary Model",
            options=['groq:llama-3.3-70b-versatile',
                     'groq:llama-3.1-70b-versatile',
                     'groq:mixtral-8x7b-32768',
                     'openrouter:nousresearch/hermes-3-llama-3.1-405b:free',
                     'openrouter:microsoft/phi-3-mini-128k-instruct:free'],
            format_func=lambda x: {
                'groq:llama-3.3-70b-versatile': '⚡ Groq Llama 3.3 70B (FREE & FAST - Recommended)',
                'groq:llama-3.1-70b-versatile': '⚡ Groq Llama 3.1 70B (FREE & FAST)',
                'groq:mixtral-8x7b-32768': '⚡ Groq Mixtral 8x7B (FREE - Large Context)',
                'openrouter:nousresearch/hermes-3-llama-3.1-405b:free': '🧠 Hermes 405B (Free, may be limited)',
                'openrouter:microsoft/phi-3-mini-128k-instruct:free': '🔷 Phi-3 Mini (Free, may be limited)'
            }[x],
            help="Groq models are FREE, FAST, and RELIABLE (no rate limits with your API key)"
        )
        
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
        include_meeting_minutes = st.checkbox(
            "Generate Minutes of Meeting (MoM)",
            value=False,
            help="Comprehensive meeting minutes capturing all details, decisions, and action items. Note: When enabled, summary generation is skipped (MoM provides comprehensive coverage)"
        )
        
        if include_meeting_minutes:
            st.caption("💡 Summary will be skipped when MoM is enabled - Minutes of Meeting provides complete coverage")
        
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
        
        # Estimate duration (rough approximation: 1 MB ≈ 1 minute for typical audio)
        estimated_duration_minutes = file_size_mb  # Rough estimate
        
        st.info(f"""
        **📄 File:** {audio_file.name}  
        **💾 Size:** {file_size_mb:.2f} MB  
        **🎵 Format:** .{file_ext}  
        **⏱️ Estimated Duration:** ~{estimated_duration_minutes:.0f} minutes
        """)
        
        if file_ext != 'wav':
            st.caption("ℹ️ File will be converted to WAV format for optimal transcription")
        
        # Check file size
        if file_size_mb > PROCESSING_SETTINGS['max_file_size_mb']:
            st.error(f"⚠️ File size exceeds {PROCESSING_SETTINGS['max_file_size_mb']} MB limit")
            return
        
        # Warn about large files that may cause streaming errors
        if file_size_mb > 50:  # Files > 50MB (roughly 45+ minutes) are at risk
            st.warning("⚠️ **Large File Detected**")
            st.markdown(f"""
            Your file is **{file_size_mb:.1f} MB** (~{estimated_duration_minutes:.0f} min estimated).
            
            **To avoid gRPC streaming errors:**
            - ✅ **Disable speaker diarization** (recommended for files > 45 min)
            - ✅ Consider splitting file into smaller chunks if > 60 minutes
            - ✅ Ensure stable internet connection during processing
            
            Files over 45 minutes may encounter "RST_STREAM" errors during streaming.
            """)
        elif file_size_mb > 35:  # 30-45 minute range
            st.info(f"📊 Medium-sized file detected ({file_size_mb:.1f} MB). "
                   "Consider disabling speaker diarization if you encounter any errors.")
    
    st.divider()
    
    # Processing button
    if audio_file:
        process_button = st.button(
            "🚀 Start Processing",
            type="primary",
            use_container_width=True
        )
        
        if process_button:
            # Store settings in session state
            st.session_state.audio_filename = os.path.splitext(audio_file.name)[0]
            st.session_state.export_formats = export_formats
            
            process_lecture(
                audio_file,
                transcription_model,
                summary_model,
                summary_type,
                include_key_points,
                include_exam_questions,
                include_meeting_minutes,
                auto_export,
                export_formats,
                enable_diarization,
                {0: speaker_0_label, 1: speaker_1_label}
            )
    else:
        st.info("👆 Please upload an audio file to begin")
    
    # Display results if available
    if st.session_state.processing_complete:
        display_results(
            st.session_state.transcript_result,
            st.session_state.summary_result,
            include_key_points,
            include_exam_questions,
            include_meeting_minutes
        )


def process_lecture(
    audio_file,
    transcription_model: str,
    summary_model: str,
    summary_type: str,
    include_key_points: bool,
    include_exam_questions: bool,
    include_meeting_minutes: bool,
    auto_export: bool,
    export_formats: list,
    enable_diarization: bool = False,
    speaker_labels: Optional[Dict[int, str]] = None
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
            progress_callback=update_progress,
            enable_diarization=enable_diarization,
            speaker_labels=speaker_labels
        )
        
        # Store audio duration for error handling (even if transcription fails)
        audio_duration_minutes = 0
        if transcript_result.get('audio_metadata'):
            duration_seconds = transcript_result['audio_metadata'].get('duration_seconds', 0)
            audio_duration_minutes = duration_seconds / 60
        
        if not transcript_result['success']:
            st.error("❌ Transcription failed")
            error_msg = transcript_result.get('errors') or transcript_result.get('error', 'Unknown error')
            st.error(error_msg)
            logger.error(f"Transcription failed: {error_msg}")
            
            # Provide specific guidance for common errors
            if 'DEADLINE_EXCEEDED' in error_msg and 'failed to establish link to worker' in error_msg:
                st.error("### ⚠️ NVIDIA API Worker Connection Failed")
                
                st.info(f"📊 Audio duration: **{audio_duration_minutes:.1f} minutes**")
                
                st.markdown("""
                ### 🔍 What Happened?
                The NVIDIA API service couldn't establish a connection to a processing worker node. 
                This is typically an **infrastructure/availability issue** on NVIDIA's side, not your file.
                
                ### 💡 Immediate Solutions:
                
                **Option 1: Retry (Recommended)**
                - Simply try processing again - this often resolves itself
                - The API may have been temporarily unavailable
                - Wait 30-60 seconds before retrying
                
                **Option 2: Disable Speaker Diarization**
                - ✅ **Highly Recommended** - This reduces processing requirements significantly
                - Uncheck "Enable Speaker Separation" in the sidebar
                - You'll still get accurate transcripts, just without speaker labels
                - This often bypasses worker availability issues
                
                **Option 3: Try Different Transcription Model**
                - Switch to **"OpenAI Whisper Large V3"** in the sidebar
                - Uses different infrastructure (offline processing)
                - More stable but takes longer
                - Better for when streaming services are unavailable
                
                **Option 4: Try During Off-Peak Hours**
                - NVIDIA API may be under heavy load
                - Try early morning or late evening (your timezone)
                - Weekends typically have better availability
                
                ### 🎯 For Your 28.9-Minute Meeting:
                Your file duration is perfectly fine - this is **not** a file size issue.
                
                **Quick Fix (90% success rate):**
                1. ✅ **Uncheck "Enable Speaker Separation"** 
                2. Click "Start Processing" again
                3. If it fails again, wait 1-2 minutes and retry
                
                ### 🔧 If Problem Persists:
                - Check [NVIDIA API Status](https://status.nvidia.com/)
                - Verify your API key hasn't expired
                - Consider switching to Whisper model as alternative
                """)
                
                # Add quick action buttons
                col1, col2 = st.columns(2)
                with col1:
                    st.info("💡 **Most likely fix:** Disable speaker diarization and retry")
                with col2:
                    if st.button("🔄 Check API Status", use_container_width=True):
                        st.markdown("[🔗 NVIDIA API Status Page](https://status.nvidia.com/)")
            
            elif 'RST_STREAM' in error_msg or 'error code 2' in error_msg:
                st.error("### 🔧 gRPC Streaming Error Detected")
                
                # Calculate if file is very long
                if audio_duration_minutes > 60:
                    st.warning(f"⚠️ Your audio file is **{audio_duration_minutes:.1f} minutes** long. "
                              "Files over 60 minutes frequently encounter streaming errors.")
                elif audio_duration_minutes > 45:
                    st.warning(f"📊 Audio duration: **{audio_duration_minutes:.1f} minutes** - "
                              "This is at the upper limit for reliable streaming.")
                elif audio_duration_minutes > 0:
                    st.info(f"📊 Audio duration: {audio_duration_minutes:.1f} minutes")
                
                st.markdown("""
                ### 💡 Solutions to Fix This Error:
                
                **Option 1: Split Your Audio File (Recommended for large files)**
                - For files over 45 minutes, splitting is highly recommended
                - Split into 30-35 minute segments for optimal processing
                - Use the command:
                  ```bash
                  python split_audio.py your_file.m4a --chunk-minutes 35
                  ```
                - Then process each chunk separately
                
                **Option 2: Disable Speaker Diarization**
                - If you have "Enable Speaker Separation" checked, uncheck it
                - This significantly reduces processing requirements
                - You can still get accurate transcripts without speaker labels
                
                **Option 3: Try a Different Model**
                - Switch to "OpenAI Whisper Large V3" (slower but more stable for large files)
                - This model handles longer files better
                
                **Option 4: Check Your Setup**
                - Ensure stable internet connection (required for streaming)
                - Verify API key is valid and has remaining credits
                - Try again in a few minutes (temporary API issues)
                
                ### 📝 For Meeting Processing:
                Since you're using this for meetings (~30 min), this error suggests:
                - Your internet connection may have interrupted
                - Try uploading again with a stable connection
                - Consider disabling speaker diarization if enabled
                """)
                
                # Add helpful buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📖 Read Full Troubleshooting Guide", use_container_width=True):
                        st.info("""
                        **Full Troubleshooting Steps:**
                        
                        1. Check file duration - split if > 45 minutes
                        2. Disable diarization to reduce load
                        3. Ensure stable internet connection
                        4. Try different transcription model
                        5. Check NVIDIA API status
                        6. Verify API key has credits
                        7. Try during off-peak hours
                        """)
                with col2:
                    st.link_button("🔗 View FFMPEG Guide", 
                                  "https://github.com/yourusername/meeting-analyzer/blob/main/FFMPEG_TROUBLESHOOTING.md",
                                  use_container_width=True)
            
            # Generic gRPC error handling
            elif 'grpc' in error_msg.lower() or 'StatusCode' in error_msg:
                st.warning("### ⚠️ NVIDIA API Communication Error")
                st.markdown("""
                A communication error occurred with the NVIDIA API service.
                
                **Quick Solutions:**
                1. **Retry** - Often resolves temporary issues
                2. **Disable speaker diarization** - Reduces processing load
                3. **Check internet connection** - Ensure stable connectivity
                4. **Try Whisper model** - Alternative transcription service
                
                If the problem persists, the NVIDIA API may be experiencing issues.
                """)
            
            return
        
        progress_bar.progress(50)
        st.success("✅ Transcription completed!")
        
        # Step 2: Summary Generation (skip if MoM is enabled, as it's redundant for meetings)
        summary_result = None
        if include_meeting_minutes:
            # Skip summary generation for meetings - MoM provides comprehensive coverage
            st.info("ℹ️ Skipping summary generation - Minutes of Meeting will provide comprehensive coverage")
            progress_bar.progress(60)
            # Create a placeholder summary result
            summary_result = {
                'success': True,
                'summary': 'Summary skipped - Minutes of Meeting generated instead',
                'summary_type': 'meeting_minutes',
                'model_name': 'N/A',
                'word_count': 0,
                'timestamp': datetime.now().isoformat()
            }
        else:
            # Generate summary for lectures
            status_text.text("🤖 Generating AI summary...")
            progress_bar.progress(60)
            
            summary_generator = SummaryGenerator(OPENROUTER_API_KEY, GROQ_API_KEY)
            
            # Use clean_transcript (without speaker labels/timestamps) for AI processing
            text_for_ai = transcript_result.get('clean_transcript', transcript_result['formatted_transcript'])
            
            summary_result = summary_generator.generate_summary(
                text_for_ai,
                summary_type=summary_type,
                model=summary_model
            )
            
            if not summary_result['success']:
                st.warning("⚠️ Summary generation failed, but transcript is available")
                logger.warning(f"Summary failed: {summary_result.get('errors')}")
            else:
                st.success("✅ Summary generated!")
        
        # Initialize summary_generator if not already done (for features below)
        if include_meeting_minutes:
            summary_generator = SummaryGenerator(OPENROUTER_API_KEY, GROQ_API_KEY)
            text_for_ai = transcript_result.get('clean_transcript', transcript_result['formatted_transcript'])
        
        progress_bar.progress(75)
        
        # Step 3: Additional features
        key_points_result = None
        exam_questions_result = None
        meeting_minutes_result = None
        
        if include_key_points:
            status_text.text("🔑 Extracting key points...")
            key_points_result = summary_generator.generate_key_points(
                text_for_ai
            )
            if not key_points_result.get('success', False):
                st.warning(f"⚠️ Key points extraction failed: {key_points_result.get('error', 'Unknown error')}")
                logger.error(f"Key points extraction failed: {key_points_result.get('error')}")
            progress_bar.progress(85)
        
        if include_exam_questions:
            status_text.text("📝 Generating exam questions...")
            exam_questions_result = summary_generator.generate_exam_questions(
                text_for_ai,
                num_questions=15
            )
            if not exam_questions_result.get('success', False):
                st.warning(f"⚠️ Exam questions generation failed: {exam_questions_result.get('error', 'Unknown error')}")
                logger.error(f"Exam questions generation failed: {exam_questions_result.get('error')}")
            progress_bar.progress(85)
        
        if include_meeting_minutes:
            status_text.text("📋 Generating comprehensive Minutes of Meeting...")
            meeting_minutes_result = summary_generator.generate_meeting_minutes(
                text_for_ai
            )
            if not meeting_minutes_result.get('success', False):
                st.warning(f"⚠️ Meeting minutes generation failed: {meeting_minutes_result.get('error', 'Unknown error')}")
                logger.error(f"Meeting minutes generation failed: {meeting_minutes_result.get('error')}")
            progress_bar.progress(90)
        
        # Step 4: Export if enabled
        if auto_export and (summary_result['success'] or meeting_minutes_result):
            status_text.text("💾 Exporting results...")
            
            exporter = FileExporter()
            base_filename = exporter.generate_filename(
                base_name=os.path.splitext(audio_file.name)[0]
            )
            
            exported_files = exporter.export_complete_session(
                transcript_result,
                summary_result,
                base_filename,
                key_points_result=key_points_result,
                exam_questions_result=exam_questions_result,
                meeting_minutes_result=meeting_minutes_result
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
        st.session_state.meeting_minutes_result = meeting_minutes_result
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
    show_exam_questions: bool,
    show_meeting_minutes: bool = False
):
    """Display processing results"""
    
    st.divider()
    st.header("📊 Results")
    
    # Tabs for different views - add MoM tab dynamically
    tab_names = ["📝 Transcript", "📚 Summary", "🔑 Key Points", "❓ Exam Questions"]
    if show_meeting_minutes:
        tab_names.append("📋 Minutes of Meeting")
    tab_names.append("📈 Metadata")
    
    tabs = st.tabs(tab_names)
    
    # Transcript tab
    with tabs[0]:
        st.subheader("Transcription")
        
        if transcript_result and transcript_result.get('formatted_transcript'):
            # Get filename and format
            base_name = st.session_state.get('audio_filename', 'transcript')
            export_formats = st.session_state.get('export_formats', ['Markdown'])
            use_markdown = 'Markdown' in export_formats
            
            # Download button at top
            file_ext = 'md' if use_markdown else 'txt'
            mime_type = 'text/markdown' if use_markdown else 'text/plain'
            st.download_button(
                "⬇️ Download Transcript",
                transcript_result['formatted_transcript'],
                file_name=f"{base_name}_Transcript.{file_ext}",
                mime=mime_type,
                use_container_width=True,
                type="primary"
            )
            
            st.divider()
            
            # Stats
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Word Count", transcript_result.get('word_count', 0))
            with col2:
                st.metric("Characters", transcript_result.get('char_count', 0))
            with col3:
                duration = transcript_result.get('audio_metadata', {}).get('duration_formatted', 'N/A')
                st.metric("Duration", duration)
            
            st.divider()
            
            # Content
            st.markdown(transcript_result['formatted_transcript'])
        else:
            st.info("No transcript available")
    
    # Summary tab
    with tabs[1]:
        st.subheader("AI-Generated Summary")
        
        # Check if summary was skipped for MoM
        if show_meeting_minutes and summary_result and summary_result.get('summary_type') == 'meeting_minutes':
            st.info("ℹ️ **Summary generation was skipped for this meeting**")
            st.markdown("""
            Since you enabled **Minutes of Meeting (MoM)**, the summary was not generated separately.
            
            **Why?** Minutes of Meeting provides comprehensive coverage of all meeting content,
            making a separate summary redundant for meetings.
            
            📋 **View the Minutes of Meeting tab** for the full detailed output.
            """)
        elif summary_result and summary_result.get('summary'):
            # Get filename and format
            base_name = st.session_state.get('audio_filename', 'summary')
            export_formats = st.session_state.get('export_formats', ['Markdown'])
            use_markdown = 'Markdown' in export_formats
            
            # Download button at top
            file_ext = 'md' if use_markdown else 'txt'
            mime_type = 'text/markdown' if use_markdown else 'text/plain'
            st.download_button(
                "⬇️ Download Summary",
                summary_result['summary'],
                file_name=f"{base_name}_Summary.{file_ext}",
                mime=mime_type,
                use_container_width=True,
                type="primary"
            )
            
            st.divider()
            
            # Stats
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Word Count", summary_result.get('word_count', 0))
            with col2:
                st.metric("Summary Type", summary_result.get('summary_type', 'N/A'))
            
            st.divider()
            
            # Content
            st.markdown(summary_result['summary'])
        else:
            st.info("No summary available")
    
    # Key Points tab
    with tabs[2]:
        st.subheader("Key Points")
        
        if hasattr(st.session_state, 'key_points_result') and st.session_state.key_points_result:
            key_points_result = st.session_state.key_points_result
            if key_points_result.get('success'):
                # Get filename and format
                base_name = st.session_state.get('audio_filename', 'keypoints')
                export_formats = st.session_state.get('export_formats', ['Markdown'])
                use_markdown = 'Markdown' in export_formats
                
                # Download button at top
                file_ext = 'md' if use_markdown else 'txt'
                mime_type = 'text/markdown' if use_markdown else 'text/plain'
                st.download_button(
                    "⬇️ Download Key Points",
                    key_points_result.get('key_points_text', ''),
                    file_name=f"{base_name}_KeyPoints.{file_ext}",
                    mime=mime_type,
                    use_container_width=True,
                    type="primary"
                )
                
                st.divider()
                st.markdown(key_points_result.get('key_points_text', ''))
            else:
                st.warning("Key points extraction failed")
        elif not show_key_points:
            st.info("Key points extraction was not enabled")
        else:
            st.info("No key points available")
    
    # Exam Questions tab
    with tabs[3]:
        st.subheader("Potential Exam Questions")
        
        if hasattr(st.session_state, 'exam_questions_result') and st.session_state.exam_questions_result:
            exam_questions_result = st.session_state.exam_questions_result
            if exam_questions_result.get('success'):
                # Get filename and format
                base_name = st.session_state.get('audio_filename', 'examquestions')
                export_formats = st.session_state.get('export_formats', ['Markdown'])
                use_markdown = 'Markdown' in export_formats
                
                # Download button at top
                file_ext = 'md' if use_markdown else 'txt'
                mime_type = 'text/markdown' if use_markdown else 'text/plain'
                st.download_button(
                    "⬇️ Download Exam Questions",
                    exam_questions_result.get('questions', ''),
                    file_name=f"{base_name}_ExamQuestions.{file_ext}",
                    mime=mime_type,
                    use_container_width=True,
                    type="primary"
                )
                
                st.divider()
                st.markdown(exam_questions_result.get('questions', ''))
            else:
                st.warning("Exam question generation failed")
        elif not show_exam_questions:
            st.info("Exam question generation was not enabled")
        else:
            st.info("No exam questions available")
    
    # Minutes of Meeting tab (conditional)
    if show_meeting_minutes:
        mom_tab_index = 4
        with tabs[mom_tab_index]:
            st.subheader("Minutes of Meeting (MoM)")
            
            if hasattr(st.session_state, 'meeting_minutes_result') and st.session_state.meeting_minutes_result:
                meeting_minutes_result = st.session_state.meeting_minutes_result
                if meeting_minutes_result.get('success'):
                    # Get filename and format
                    base_name = st.session_state.get('audio_filename', 'meeting_minutes')
                    export_formats = st.session_state.get('export_formats', ['Markdown'])
                    use_markdown = 'Markdown' in export_formats
                    
                    # Download button at top
                    file_ext = 'md' if use_markdown else 'txt'
                    mime_type = 'text/markdown' if use_markdown else 'text/plain'
                    st.download_button(
                        "⬇️ Download Minutes of Meeting",
                        meeting_minutes_result.get('minutes', ''),
                        file_name=f"{base_name}_MeetingMinutes.{file_ext}",
                        mime=mime_type,
                        use_container_width=True,
                        type="primary"
                    )
                    
                    st.divider()
                    
                    # Info box about MoM
                    st.info("📋 **Comprehensive Meeting Minutes** - Includes all decisions, action items, discussions, and important details from the meeting.")
                    
                    st.divider()
                    st.markdown(meeting_minutes_result.get('minutes', ''))
                else:
                    st.warning("Meeting minutes generation failed")
            else:
                st.info("No meeting minutes available")
    
    # Metadata tab (adjust index based on whether MoM is shown)
    metadata_tab_index = 5 if show_meeting_minutes else 4
    with tabs[metadata_tab_index]:
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
