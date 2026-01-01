"""
Command-line interface for batch processing lectures
Perfect for processing multiple recordings at once
"""
import argparse
import sys
from pathlib import Path
from typing import List
import time

sys.path.insert(0, str(Path(__file__).parent))

from config.config import NVIDIA_API_KEY, OPENROUTER_API_KEY, OUTPUTS_DIR
from src.transcription import TranscriptionEngine
from src.summarization import SummaryGenerator
from src.file_exporter import FileExporter
from src.audio_processor import AudioProcessor
from src.logger import setup_logger

logger = setup_logger(__name__)


def process_single_file(
    file_path: str,
    transcription_model: str,
    summary_type: str,
    include_key_points: bool,
    include_exam_questions: bool,
    export: bool
):
    """Process a single audio file"""
    
    print(f"\n{'='*60}")
    print(f"Processing: {Path(file_path).name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Initialize engines
        transcription_engine = TranscriptionEngine(NVIDIA_API_KEY)
        summary_generator = SummaryGenerator(OPENROUTER_API_KEY)
        
        # Step 1: Transcribe
        print("\n📝 Step 1/4: Transcribing audio...")
        transcript_result = transcription_engine.transcribe(
            file_path,
            model=transcription_model
        )
        
        if not transcript_result['success']:
            print(f"❌ Transcription failed: {transcript_result['errors']}")
            return False
        
        print(f"✅ Transcription complete ({transcript_result['word_count']} words)")
        
        # Step 2: Generate summary
        print("\n📚 Step 2/4: Generating summary...")
        summary_result = summary_generator.generate_summary(
            transcript_result['formatted_transcript'],
            summary_type=summary_type
        )
        
        if not summary_result['success']:
            print(f"⚠️ Summary generation failed: {summary_result.get('errors')}")
        else:
            print(f"✅ Summary generated ({summary_result['word_count']} words)")
        
        # Step 3: Key points (optional)
        key_points_result = None
        if include_key_points:
            print("\n🔑 Step 3/4: Extracting key points...")
            key_points_result = summary_generator.generate_key_points(
                transcript_result['formatted_transcript']
            )
            if key_points_result['success']:
                print(f"✅ {key_points_result['count']} key points extracted")
        
        # Step 4: Exam questions (optional)
        exam_questions_result = None
        if include_exam_questions:
            print("\n❓ Step 4/4: Generating exam questions...")
            exam_questions_result = summary_generator.generate_exam_questions(
                transcript_result['formatted_transcript']
            )
            if exam_questions_result['success']:
                print("✅ Exam questions generated")
        
        # Export
        if export and summary_result['success']:
            print("\n💾 Exporting results...")
            exporter = FileExporter()
            
            base_name = Path(file_path).stem
            base_filename = exporter.generate_filename(base_name)
            
            exported_files = exporter.export_complete_session(
                transcript_result,
                summary_result,
                base_filename
            )
            
            print(f"✅ Exported {len(exported_files)} files to {OUTPUTS_DIR}")
            for file_type, file_path in exported_files.items():
                print(f"   - {file_type}: {Path(file_path).name}")
        
        # Summary
        elapsed_time = time.time() - start_time
        print(f"\n✨ Processing complete in {elapsed_time:.1f} seconds")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
        return False


def process_batch(
    file_paths: List[str],
    transcription_model: str,
    summary_type: str,
    include_key_points: bool,
    include_exam_questions: bool,
    export: bool
):
    """Process multiple audio files"""
    
    print(f"\n{'#'*60}")
    print(f"  BATCH PROCESSING: {len(file_paths)} files")
    print(f"{'#'*60}")
    
    successful = 0
    failed = 0
    
    start_time = time.time()
    
    for i, file_path in enumerate(file_paths, 1):
        print(f"\n[{i}/{len(file_paths)}]")
        
        if process_single_file(
            file_path,
            transcription_model,
            summary_type,
            include_key_points,
            include_exam_questions,
            export
        ):
            successful += 1
        else:
            failed += 1
        
        # Brief pause between files
        if i < len(file_paths):
            print("\nPausing for 5 seconds before next file...")
            time.sleep(5)
    
    # Final summary
    elapsed_time = time.time() - start_time
    
    print(f"\n{'#'*60}")
    print(f"  BATCH PROCESSING COMPLETE")
    print(f"{'#'*60}")
    print(f"\n✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    print(f"\n📁 Output files saved to: {OUTPUTS_DIR}")


def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(
        description="Class Lecture Transcription System - Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python cli.py lecture.mp3
  
  # Process with Whisper model and detailed notes
  python cli.py lecture.mp3 -m openai/whisper-large-v3 -s detailed_notes
  
  # Batch process multiple files
  python cli.py lecture1.mp3 lecture2.mp3 lecture3.mp3
  
  # Process all MP3 files in a directory
  python cli.py /path/to/lectures/*.mp3
  
  # Include key points and exam questions
  python cli.py lecture.mp3 -k -e
  
  # Don't auto-export (just display)
  python cli.py lecture.mp3 --no-export
        """
    )
    
    parser.add_argument(
        'files',
        nargs='+',
        help='Audio file(s) to process'
    )
    
    parser.add_argument(
        '-m', '--model',
        choices=['nvidia/parakeet-ctc-1.1b-asr', 'openai/whisper-large-v3'],
        default='nvidia/parakeet-ctc-1.1b-asr',
        help='Transcription model (default: parakeet)'
    )
    
    parser.add_argument(
        '-s', '--summary-type',
        choices=['class_lecture', 'brief_summary', 'detailed_notes'],
        default='class_lecture',
        help='Summary type (default: class_lecture)'
    )
    
    parser.add_argument(
        '-k', '--key-points',
        action='store_true',
        help='Extract key points'
    )
    
    parser.add_argument(
        '-e', '--exam-questions',
        action='store_true',
        help='Generate exam questions'
    )
    
    parser.add_argument(
        '--no-export',
        action='store_true',
        help='Don\'t export results to files'
    )
    
    args = parser.parse_args()
    
    # Validate API keys
    if not NVIDIA_API_KEY:
        print("❌ NVIDIA_API_KEY not found in .env file")
        sys.exit(1)
    
    if not OPENROUTER_API_KEY:
        print("⚠️ OPENROUTER_API_KEY not found. Summary generation will fail.")
    
    # Validate files
    valid_files = []
    audio_processor = AudioProcessor()
    
    for file_path in args.files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠️  File not found: {file_path}")
            continue
        
        ext = path.suffix.lower().replace('.', '')
        if ext not in audio_processor.supported_formats:
            print(f"⚠️  Unsupported format: {file_path}")
            continue
        
        valid_files.append(str(path.absolute()))
    
    if not valid_files:
        print("❌ No valid files to process")
        sys.exit(1)
    
    # Process files
    if len(valid_files) == 1:
        # Single file
        success = process_single_file(
            valid_files[0],
            args.model,
            args.summary_type,
            args.key_points,
            args.exam_questions,
            not args.no_export
        )
        sys.exit(0 if success else 1)
    else:
        # Batch processing
        process_batch(
            valid_files,
            args.model,
            args.summary_type,
            args.key_points,
            args.exam_questions,
            not args.no_export
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Processing cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        logger.error("CLI error", exc_info=True)
        sys.exit(1)
