"""
Simple script to check if FFmpeg is installed and accessible.
"""
import subprocess
import sys

def check_ffmpeg():
    """Check if FFmpeg is installed and accessible."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Extract version from output
        first_line = result.stdout.split('\n')[0]
        print("✅ FFmpeg is installed!")
        print(f"   {first_line}")
        return True
        
    except FileNotFoundError:
        print("❌ FFmpeg is NOT installed or not in PATH")
        print("\n📋 Installation instructions:")
        print("   Windows: winget install --id=Gyan.FFmpeg -e")
        print("   After installation, restart your terminal!")
        return False
        
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {e}")
        return False

if __name__ == "__main__":
    print("Checking FFmpeg installation...")
    print("-" * 50)
    
    if check_ffmpeg():
        print("\n✅ Your system is ready for audio conversion!")
        sys.exit(0)
    else:
        print("\n⚠️  Please install FFmpeg to convert MP3/M4A files")
        print("    Or use WAV files directly")
        sys.exit(1)
