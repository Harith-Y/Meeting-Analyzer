"""
Quick setup script for Class Lecture Transcription System
"""
import os
import sys
import subprocess
from pathlib import Path


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required!")
        print("   Please upgrade Python and try again.")
        return False
    
    print("✅ Python version is compatible")
    return True


def check_ffmpeg():
    """Check if FFmpeg is installed"""
    print_header("Checking FFmpeg Installation")
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            check=True
        )
        first_line = result.stdout.split('\n')[0]
        print(f"✅ {first_line}")
        return True
    except FileNotFoundError:
        print("❌ FFmpeg is NOT installed")
        print("\n📋 Installation instructions:")
        print("   Windows: winget install --id=Gyan.FFmpeg -e")
        print("   After installation, restart your terminal!")
        return False
    except Exception as e:
        print(f"❌ Error checking FFmpeg: {e}")
        return False


def check_venv():
    """Check if virtual environment exists"""
    print_header("Checking Virtual Environment")
    
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment found")
        return True
    else:
        print("❌ Virtual environment not found")
        return False


def create_venv():
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    
    try:
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✅ Virtual environment created successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False


def install_requirements():
    """Install required packages"""
    print_header("Installing Requirements")
    
    # Determine pip path based on OS
    if sys.platform == "win32":
        pip_path = Path("venv/Scripts/pip.exe")
    else:
        pip_path = Path("venv/bin/pip")
    
    if not pip_path.exists():
        print("❌ Virtual environment not properly set up")
        return False
    
    print("Installing packages... This may take a few minutes.")
    
    try:
        # Upgrade pip first
        subprocess.run(
            [str(pip_path), "install", "--upgrade", "pip"],
            check=True,
            capture_output=True
        )
        
        # Install requirements
        req_file = "requirements_new.txt" if Path("requirements_new.txt").exists() else "requirements.txt"
        
        subprocess.run(
            [str(pip_path), "install", "-r", req_file],
            check=True
        )
        
        print("✅ All packages installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False


def check_env_file():
    """Check if .env file exists"""
    print_header("Checking Configuration")
    
    env_path = Path(".env")
    if env_path.exists():
        print("✅ .env file found")
        
        # Check if keys are set
        with open(env_path, 'r') as f:
            content = f.read()
            has_nvidia = "NVIDIA_API_KEY=" in content
            has_openrouter = "OPENROUTER_API_KEY=" in content
        
        if has_nvidia:
            print("  ✓ NVIDIA_API_KEY configured")
        else:
            print("  ⚠️  NVIDIA_API_KEY not set")
        
        if has_openrouter:
            print("  ✓ OPENROUTER_API_KEY configured")
        else:
            print("  ⚠️  OPENROUTER_API_KEY not set")
        
        return has_nvidia and has_openrouter
    else:
        print("❌ .env file not found")
        return False


def create_env_template():
    """Create .env template file"""
    print_header("Creating .env Template")
    
    env_content = """# API Keys for Class Lecture Transcription System
# Get your keys from:
# NVIDIA: https://build.nvidia.com/
# OpenRouter: https://openrouter.ai/

NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
"""
    
    try:
        with open(".env", "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
        print("📝 Please edit .env and add your API keys")
        return True
    except Exception as e:
        print(f"❌ Failed to create .env: {e}")
        return False


def check_python_clients():
    """Check if NVIDIA Riva python-clients exists"""
    print_header("Checking NVIDIA Riva Client")
    
    clients_path = Path("python-clients/scripts/asr")
    
    if clients_path.exists():
        print("✅ NVIDIA Riva client scripts found")
        return True
    else:
        print("❌ NVIDIA Riva client scripts not found")
        print("\n📋 The python-clients folder should contain NVIDIA Riva scripts")
        print("   If missing, you may need to clone it separately")
        return False


def print_next_steps():
    """Print next steps for the user"""
    print_header("Next Steps")
    
    print("""
1. Activate the virtual environment:
   Windows PowerShell:  .\\venv\\Scripts\\Activate.ps1
   Windows CMD:         .\\venv\\Scripts\\activate.bat
   Linux/Mac:           source venv/bin/activate

2. Edit the .env file and add your API keys:
   - Get NVIDIA API key from: https://build.nvidia.com/
   - Get OpenRouter key from: https://openrouter.ai/

3. Run the application:
   streamlit run app_enhanced.py

4. Open your browser to: http://localhost:8501

For help, see README_NEW.md
""")


def main():
    """Main setup function"""
    print("\n" + "🎓 "*20)
    print("   CLASS LECTURE TRANSCRIPTION SYSTEM - SETUP")
    print("🎓 "*20)
    
    all_checks_passed = True
    
    # 1. Check Python version
    if not check_python_version():
        return
    
    # 2. Check FFmpeg
    if not check_ffmpeg():
        all_checks_passed = False
        print("\n⚠️  FFmpeg is required but not installed. Please install it first.")
    
    # 3. Check/Create virtual environment
    if not check_venv():
        print("\nCreating virtual environment...")
        if not create_venv():
            return
    
    # 4. Install requirements
    if not install_requirements():
        all_checks_passed = False
    
    # 5. Check/Create .env
    if not check_env_file():
        create_env_template()
        all_checks_passed = False
    
    # 6. Check python-clients
    if not check_python_clients():
        all_checks_passed = False
    
    # Summary
    print_header("Setup Summary")
    
    if all_checks_passed:
        print("✅ Setup completed successfully!")
        print("🎉 You're ready to transcribe lectures!")
        print_next_steps()
    else:
        print("⚠️  Setup completed with some issues")
        print("📝 Please address the issues above before running the application")
        print_next_steps()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
