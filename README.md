# Meeting Analyzer

A Streamlit-based web application for transcribing and summarizing meeting audio files using NVIDIA Riva ASR (Automatic Speech Recognition) and AI-powered summarization via OpenRouter.

## 🎯 Features

- **🎤 Audio Transcription**: Support for multiple state-of-the-art ASR models
  - **NVIDIA Parakeet CTC 1.1B**: High-performance real-time transcription
  - **OpenAI Whisper Large V3**: Industry-leading accuracy for complex audio
- **🤖 AI-Powered Summarization**: Automatic meeting summary generation using DeepSeek Chat V3.1
- **📁 Multiple Audio Formats**: Supports MP3, WAV, and M4A files
- **🧹 Clean Output**: Automatic formatting and artifact removal for professional results
- **💻 Modern UI**: Simple and intuitive Streamlit interface with real-time feedback
- **⚡ Fast Processing**: Optimized workflow with efficient error handling

## 📂 Project Structure

```
Meeting-Analyzer/
├── app.py                      # Main Streamlit application (UI layer)
├── transcript_gen.py           # Transcription logic and text formatting
├── summary_gen.py              # AI summary generation with artifact cleaning
├── .env                        # Environment variables (API keys)
├── python-clients/             # NVIDIA Riva client scripts (cloned separately)
│   └── scripts/asr/
│       ├── transcribe_file.py
│       └── transcribe_file_offline.py
└── README.md                   # This file
```

## 🔧 Module Descriptions

### `app.py` (UI Layer)
**Purpose**: Main application interface and workflow orchestration

**Key Features**:
- User file upload and model selection
- Progress indicators with spinners
- Success/error status display with emoji indicators (✅/❌)
- Collapsible debug information
- API key validation

**Responsibilities**:
- Handle user interactions
- Coordinate between transcription and summarization modules
- Display formatted results
- Provide helpful error messages and suggestions

### `transcript_gen.py` (Transcription Module)
**Purpose**: Audio-to-text conversion with model-specific formatting

**Functions**:
- `generate_transcript()`: Main transcription function
  - Saves uploaded audio temporarily
  - Executes NVIDIA Riva ASR via subprocess
  - Returns structured dict with transcript, formatted text, and errors
  - Automatic cleanup of temporary files
  
- `format_output_parakeet()`: Formats Parakeet model output
  - Removes `##` markers from line-by-line output
  - Adds proper punctuation (periods)
  - Capitalizes sentences
  
- `format_output_whisper()`: Formats Whisper model output
  - Extracts "Final transcript:" from verbose output
  - Ensures proper capitalization and punctuation

**Design Philosophy**:
- Pure business logic - no Streamlit dependencies
- Returns structured data for easy testing
- Comprehensive error handling with detailed messages

### `summary_gen.py` (Summarization Module)
**Purpose**: AI-powered meeting summarization with output cleaning

**Functions**:
- `generate_summary()`: Generates concise meeting summaries
  - Calls OpenRouter API with DeepSeek Chat V3.1 model
  - Enhanced prompt for focused summaries (key topics, decisions, action items)
  - Automatic artifact cleaning
  
- `clean_summary()`: Removes AI model artifacts
  - Strips special tokens like `<｜begin▁of▁sentence｜>`
  - Removes extra whitespace
  - Ensures clean, professional output

**API Integration**:
- RESTful API calls to OpenRouter
- JSON request/response handling
- Status code validation
- Error handling with user feedback

## 🚀 Setup

### Prerequisites
- **Python 3.8+** (tested on Python 3.13)
- **FFmpeg** (required for MP3/M4A conversion)
- **Virtual environment** (recommended)
- **Internet connection** (for API calls)

### Installation Steps

#### 1. Install FFmpeg (Required for MP3/M4A files)

**Windows:**
```powershell
# Using winget (recommended)
winget install --id=Gyan.FFmpeg -e

# After installation, restart your terminal!
# Verify installation:
ffmpeg -version
```

**Mac:**
```bash
brew install ffmpeg
```

**Linux:**
```bash
sudo apt update
sudo apt install ffmpeg
```

#### 2. Clone and Setup Python Environment

1. **Clone the repository**:
```powershell
git clone https://github.com/Harith-Y/Meeting-Analyzer.git
cd Meeting-Analyzer
```

2. **Create and activate virtual environment**:
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# If you get an execution policy error:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

3. **Install Python dependencies**:
```powershell
pip install streamlit nvidia-riva-client requests python-dotenv ffmpeg-python numpy soundfile
```

4. **Verify FFmpeg installation**:
```powershell
python check_ffmpeg.py
```

5. **Clone NVIDIA Riva Python clients**:
```powershell
git clone https://github.com/nvidia-riva/python-clients.git
```

6. **Create `.env` file** in the project root:
```env
NVIDIA_API_KEY=your_nvidia_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### Getting API Keys

#### NVIDIA API Key
1. Visit [NVIDIA Cloud](https://build.nvidia.com/)
2. Sign up or log in
3. Navigate to API Keys section
4. Generate a new API key
5. Copy to `.env` file

#### OpenRouter API Key
1. Visit [OpenRouter](https://openrouter.ai/)
2. Create an account
3. Go to Keys section
4. Create a new API key
5. Copy to `.env` file

## 📖 Usage

### Running the Application

```powershell
streamlit run app.py
```

The app will open in your default browser (usually at `http://localhost:8501`)

### Step-by-Step Guide

1. **Upload Audio File**
   - Click "Browse files" button
   - Select an audio file (MP3, WAV, or M4A)
   - Preview will appear below

2. **Select Transcription Model**
   - Choose from dropdown:
     - **NVIDIA Parakeet**: Fast, real-time transcription
     - **OpenAI Whisper**: Higher accuracy, better for complex audio

3. **Analyze Meeting**
   - Click "Analyze Meeting" button
   - Wait for transcription (may take 1-5 minutes depending on file size)
   - View results in two sections:
     - **Transcription**: Full text of the audio
     - **Meeting Summary**: AI-generated summary

4. **Review Results**
   - Copy transcript or summary as needed
   - Check debug info in expandable section if needed

## 🏗️ Architecture Benefits

### 🎯 Separation of Concerns
- **UI Layer** (`app.py`): Only handles display and user interaction
- **Business Logic** (`transcript_gen.py`, `summary_gen.py`): Pure functions, testable independently
- Clear boundaries between modules

### 🔄 Modularity
- Functions return structured data (`dict`) instead of displaying directly
- Easy to swap models or add new features
- Reusable components across different projects
- Each module can be tested independently

### 🛡️ Error Handling
- Comprehensive try-catch blocks in each module
- Structured error responses with detailed messages
- User-friendly error display in UI
- Helpful suggestions for common issues

### 📊 Data Flow
```
User Upload → app.py → transcript_gen.py → NVIDIA Riva API
                ↓
         Formatted Transcript
                ↓
         summary_gen.py → OpenRouter API
                ↓
         Cleaned Summary
                ↓
         Display to User
```

## ⚠️ Known Issues

### NVIDIA Parakeet Connectivity
The NVIDIA Parakeet service occasionally experiences connectivity issues with `DEADLINE_EXCEEDED` errors. This is a service-side issue.

**Solutions**:
- Switch to OpenAI Whisper model (more reliable)
- Wait 10-15 minutes and retry
- Check NVIDIA service status at [build.nvidia.com](https://build.nvidia.com/)

### Large Audio Files
Files over 10 minutes may timeout or fail.

**Solutions**:
- Split audio into smaller chunks
- Use audio editing software (Audacity, etc.)
- Compress audio to lower bitrate

## 🔮 Future Improvements

- [ ] **More Models**: Support for additional ASR models (Google Speech, Azure, AWS)
- [ ] **Batch Processing**: Upload and process multiple files at once
- [ ] **Export Options**: Download transcripts as TXT, DOCX, or PDF
- [ ] **Speaker Diarization**: Identify and label different speakers
- [ ] **Custom Prompts**: User-defined summary templates
- [ ] **Real-time Transcription**: Live audio streaming
- [ ] **Language Support**: Multi-language transcription
- [ ] **Audio Preview**: Playback controls with timestamp navigation
- [ ] **History**: Save and review past transcriptions
- [ ] **API Rate Limiting**: Better handling of API quotas

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines
- Follow existing code style
- Add docstrings to new functions
- Test your changes thoroughly
- Update README if adding new features

## 📝 License

MIT License - feel free to use this project for personal or commercial purposes.

## 👤 Author

**Harith-Y**
- GitHub: [@Harith-Y](https://github.com/Harith-Y)
- Repository: [Meeting-Analyzer](https://github.com/Harith-Y/Meeting-Analyzer)

## 🙏 Acknowledgments

- **NVIDIA Riva** for providing powerful ASR APIs
- **OpenRouter** for AI model access
- **Streamlit** for the excellent web framework
- **DeepSeek** for the chat model

---

⭐ If you find this project helpful, please consider giving it a star!