# Voice Coach Mirror

## Overview

Voice Coach Mirror is a free, offline, real-time public speaking and communication trainer built with Python and Streamlit. The application provides comprehensive feedback on facial expressions, voice quality, and presentation skills by analyzing video and audio data locally without requiring any cloud services or APIs.

The system captures live video and audio during practice sessions, processes them using computer vision and audio analysis techniques, and provides detailed scoring and feedback to help users improve their communication skills.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

The application follows a modular architecture with clear separation of concerns:

### Frontend Architecture
- **Framework**: Streamlit web application
- **User Interface**: Single-page application with sidebar navigation
- **Real-time Display**: Live video feed with overlay metrics
- **Visualization**: Plotly charts for progress tracking and score visualization

### Backend Architecture
- **Processing Pipeline**: Modular design with separate processors for video and audio
- **Data Flow**: Session state management through Streamlit's built-in state system
- **Scoring System**: Comprehensive scoring engine with weighted metrics
- **Data Persistence**: Local JSON file storage for session history

### Core Processing Components
1. **Video Processor**: Real-time facial analysis using OpenCV and dlib
2. **Audio Processor**: Voice analysis using PyAudio and librosa
3. **Scoring Engine**: Multi-dimensional scoring system
4. **Data Manager**: Session storage and retrieval
5. **Practice Prompts**: Curated prompts for different scenarios

## Key Components

### 1. Video Processing Module (`modules/video_processor.py`)
- **Purpose**: Analyzes facial expressions, eye contact, and head movements
- **Key Technologies**: OpenCV, dlib face detection
- **Features**: 
  - Face detection and landmark tracking
  - Eye contact percentage calculation
  - Head stability analysis
  - Smile detection
- **Dependencies**: Requires shape_predictor_68_face_landmarks.dat model

### 2. Audio Processing Module (`modules/audio_processor.py`)
- **Purpose**: Processes voice quality, speech patterns, and audio metrics
- **Key Technologies**: PyAudio, librosa, numpy
- **Features**:
  - Real-time audio capture
  - Speech rate analysis
  - Silence and filler detection
  - Pitch variation tracking
  - Volume level monitoring
- **Threading**: Uses separate thread for continuous audio processing

### 3. Scoring Engine (`modules/scoring_engine.py`)
- **Purpose**: Calculates comprehensive scores across multiple dimensions
- **Scoring Categories**:
  - Confidence (face detection, eye contact, head stability)
  - Clarity (speech rate, volume consistency, filler ratio)
  - Presence (energy level, smile frequency, posture)
  - Energy (voice energy, pitch variation, engagement)
- **Weighting System**: Configurable weights for different metrics

### 4. Data Manager (`modules/data_manager.py`)
- **Purpose**: Handles session storage and retrieval
- **Storage Format**: JSON files for structured data
- **Features**:
  - Session serialization
  - Progress tracking
  - Data summarization
  - Backup functionality

### 5. Practice Prompts (`modules/practice_prompts.py`)
- **Purpose**: Provides curated practice scenarios
- **Categories**:
  - Interview preparation
  - Academic presentations
  - General presentations
  - Conversational practice
- **Randomization**: Dynamic prompt selection

## Data Flow

1. **Session Initialization**: User selects practice type and prompt
2. **Data Capture**: Simultaneous video and audio recording
3. **Real-time Processing**: 
   - Video frames analyzed for facial metrics
   - Audio chunks processed for voice metrics
4. **Score Calculation**: Multi-dimensional scoring based on collected metrics
5. **Feedback Generation**: Visual and textual feedback presentation
6. **Data Persistence**: Session results saved to local JSON storage
7. **Progress Tracking**: Historical data visualization

## External Dependencies

### Required Python Packages
- **streamlit**: Web application framework
- **opencv-python**: Computer vision processing
- **dlib**: Face detection and landmark prediction
- **pyaudio**: Audio capture and processing
- **librosa**: Audio analysis
- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **plotly**: Interactive visualizations

### External Models
- **shape_predictor_68_face_landmarks.dat**: Dlib's face landmark predictor model
- **Download Strategy**: Automatic download from dlib-models repository if not present

### System Requirements
- **Camera**: Webcam for video capture
- **Microphone**: Audio input device
- **Storage**: Local file system for data persistence
- **Processing**: CPU-based processing (no GPU required)

## Deployment Strategy

### Local Development
- **Environment**: Python virtual environment recommended
- **Dependencies**: Install via requirements.txt
- **Launch**: `streamlit run app.py`
- **Data Directory**: Automatic creation of `data/` and `models/` folders

### Production Considerations
- **Offline Operation**: Complete functionality without internet after initial setup
- **Cross-platform**: Compatible with Windows, macOS, and Linux
- **Resource Management**: Efficient memory usage with threading
- **Error Handling**: Graceful degradation when hardware unavailable

### File Structure
```
voice-coach-mirror/
├── app.py                 # Main Streamlit application
├── data/
│   └── sessions.json      # Session history storage
├── models/                # ML model storage
├── modules/
│   ├── video_processor.py
│   ├── audio_processor.py
│   ├── scoring_engine.py
│   ├── data_manager.py
│   └── practice_prompts.py
└── requirements.txt       # Python dependencies
```

The architecture prioritizes simplicity, offline functionality, and user privacy by processing all data locally without external API dependencies.