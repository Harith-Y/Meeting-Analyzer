"""
Split large audio files into smaller chunks for processing
This helps with files that exceed the gRPC message size limit (67MB)
"""
import argparse
import sys
from pathlib import Path
from pydub import AudioSegment
import os


def split_audio_file(input_file: str, chunk_duration_minutes: int = 20, output_dir: str = None):
    """
    Split audio file into smaller chunks.
    
    Args:
        input_file: Path to input audio file
        chunk_duration_minutes: Duration of each chunk in minutes
        output_dir: Optional output directory (defaults to same as input)
    
    Returns:
        list: Paths to the created chunk files
    """
    print(f"Loading audio file: {input_file}")
    
    # Load audio file
    audio = AudioSegment.from_file(input_file)
    
    # Get file info
    duration_ms = len(audio)
    duration_minutes = duration_ms / (1000 * 60)
    
    print(f"Audio duration: {duration_minutes:.2f} minutes")
    print(f"File size: {os.path.getsize(input_file) / (1024 * 1024):.2f} MB")
    
    # Calculate number of chunks
    chunk_duration_ms = chunk_duration_minutes * 60 * 1000
    num_chunks = int(duration_ms / chunk_duration_ms) + (1 if duration_ms % chunk_duration_ms > 0 else 0)
    
    print(f"\nSplitting into {num_chunks} chunks of {chunk_duration_minutes} minutes each...")
    
    # Prepare output directory
    if output_dir is None:
        output_dir = Path(input_file).parent / "chunks"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Get base filename
    base_name = Path(input_file).stem
    file_ext = Path(input_file).suffix
    
    # Split into chunks
    chunk_files = []
    
    for i in range(num_chunks):
        start_ms = i * chunk_duration_ms
        end_ms = min((i + 1) * chunk_duration_ms, duration_ms)
        
        chunk = audio[start_ms:end_ms]
        
        # Create output filename
        output_file = output_dir / f"{base_name}_part{i+1:02d}{file_ext}"
        
        print(f"Creating chunk {i+1}/{num_chunks}: {output_file.name}")
        print(f"  Duration: {(end_ms - start_ms) / (1000 * 60):.2f} minutes")
        
        # Export chunk
        chunk.export(str(output_file), format=file_ext.replace('.', ''))
        
        chunk_size = os.path.getsize(output_file) / (1024 * 1024)
        print(f"  Size: {chunk_size:.2f} MB")
        
        chunk_files.append(str(output_file))
    
    print(f"\n✅ Successfully created {num_chunks} chunks in: {output_dir}")
    print("\nYou can now process each chunk separately and combine the transcripts.")
    
    return chunk_files


def main():
    parser = argparse.ArgumentParser(
        description="Split large audio files into smaller chunks for processing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Split into 20-minute chunks (default)
  python split_audio.py lecture.m4a
  
  # Split into 15-minute chunks
  python split_audio.py lecture.m4a -d 15
  
  # Specify output directory
  python split_audio.py lecture.m4a -o my_chunks/
        """
    )
    
    parser.add_argument('input_file', help='Audio file to split')
    parser.add_argument(
        '-d', '--duration',
        type=int,
        default=20,
        help='Duration of each chunk in minutes (default: 20)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for chunks (default: chunks/ in same directory)'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input_file).exists():
        print(f"❌ Error: File not found: {args.input_file}")
        sys.exit(1)
    
    try:
        chunk_files = split_audio_file(
            args.input_file,
            args.duration,
            args.output_dir
        )
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("\n1. Process each chunk using the CLI:")
        for chunk_file in chunk_files:
            print(f"   python cli.py {chunk_file}")
        
        print("\n2. Or use the web interface and upload each chunk")
        print("\n3. Combine the transcripts manually or use a text editor")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        # Check if pydub is installed
        try:
            import pydub
        except ImportError:
            print("❌ Error: pydub is not installed")
            print("\nInstall it with:")
            print("  pip install pydub")
            sys.exit(1)
        
        main()
    
    except KeyboardInterrupt:
        print("\n\n❌ Operation cancelled by user")
        sys.exit(1)
