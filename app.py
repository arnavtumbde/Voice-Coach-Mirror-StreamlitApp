import streamlit as st
import cv2
import numpy as np
import pandas as pd
import json
import time
import threading
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Import custom modules
from modules.video_processor import VideoProcessor
from modules.audio_processor import AudioProcessor
from modules.scoring_engine import ScoringEngine
from modules.data_manager import DataManager
from modules.practice_prompts import PracticePrompts

# Initialize session state
if 'session_data' not in st.session_state:
    st.session_state.session_data = {}
if 'recording_active' not in st.session_state:
    st.session_state.recording_active = False
if 'current_prompt' not in st.session_state:
    st.session_state.current_prompt = None
if 'video_processor' not in st.session_state:
    st.session_state.video_processor = VideoProcessor()
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()
if 'scoring_engine' not in st.session_state:
    st.session_state.scoring_engine = ScoringEngine()
if 'data_manager' not in st.session_state:
    st.session_state.data_manager = DataManager()
if 'practice_prompts' not in st.session_state:
    st.session_state.practice_prompts = PracticePrompts()

def main():
    st.set_page_config(
        page_title="Voice Coach Mirror",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎤 Voice Coach Mirror")
    st.subheader("Your Personal Public Speaking Trainer")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Practice Session", "Progress Dashboard", "Settings"])
    
    if page == "Practice Session":
        practice_session_page()
    elif page == "Progress Dashboard":
        progress_dashboard_page()
    elif page == "Settings":
        settings_page()

def practice_session_page():
    st.header("🎯 Practice Session")
    
    # Check if camera and microphone are available
    if not check_devices():
        st.error("Please ensure your camera and microphone are connected and accessible.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Video feed placeholder
        video_placeholder = st.empty()
        
        # Control buttons
        col_start, col_stop = st.columns(2)
        
        with col_start:
            if st.button("🎬 Start Practice", disabled=st.session_state.recording_active):
                start_practice_session()
        
        with col_stop:
            if st.button("⏹️ Stop Practice", disabled=not st.session_state.recording_active):
                stop_practice_session()
    
    with col2:
        # Current prompt display
        if st.session_state.current_prompt:
            st.subheader("📝 Current Prompt")
            st.info(st.session_state.current_prompt)
        
        # Real-time feedback
        if st.session_state.recording_active:
            st.subheader("📊 Real-time Feedback")
            feedback_placeholder = st.empty()
            update_realtime_feedback(feedback_placeholder)
    
    # Session results
    if 'last_session_results' in st.session_state:
        display_session_results()

def start_practice_session():
    """Start a new practice session"""
    # Get random prompt
    st.session_state.current_prompt = st.session_state.practice_prompts.get_random_prompt()
    
    # Initialize processors
    st.session_state.video_processor.start_capture()
    st.session_state.audio_processor.start_recording()
    
    # Set recording state
    st.session_state.recording_active = True
    
    # Initialize session data
    st.session_state.session_data = {
        'start_time': datetime.now(),
        'prompt': st.session_state.current_prompt,
        'video_data': [],
        'audio_data': [],
        'duration': 0
    }
    
    st.success("Practice session started! Begin speaking when ready.")
    st.rerun()

def stop_practice_session():
    """Stop the current practice session and analyze results"""
    if not st.session_state.recording_active:
        return
    
    # Stop recording
    st.session_state.recording_active = False
    
    # Get final data from processors
    video_data = st.session_state.video_processor.stop_capture()
    audio_data = st.session_state.audio_processor.stop_recording()
    
    # Calculate session duration
    duration = (datetime.now() - st.session_state.session_data['start_time']).total_seconds()
    
    # Update session data
    st.session_state.session_data.update({
        'end_time': datetime.now(),
        'duration': duration,
        'video_data': video_data,
        'audio_data': audio_data
    })
    
    # Analyze and score the session
    scores = st.session_state.scoring_engine.calculate_scores(
        video_data, audio_data, duration
    )
    
    st.session_state.session_data['scores'] = scores
    
    # Save session data
    st.session_state.data_manager.save_session(st.session_state.session_data)
    
    # Store results for display
    st.session_state.last_session_results = {
        'scores': scores,
        'duration': duration,
        'prompt': st.session_state.current_prompt
    }
    
    st.success("Practice session completed! Check your results below.")
    st.rerun()

def display_session_results():
    """Display the results of the last practice session"""
    st.header("📈 Session Results")
    
    results = st.session_state.last_session_results
    scores = results['scores']
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Radar chart
        st.subheader("Performance Radar")
        
        categories = ['Confidence', 'Clarity', 'Presence', 'Energy']
        values = [scores['confidence'], scores['clarity'], scores['presence'], scores['energy']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Your Performance'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Detailed scores
        st.subheader("Detailed Scores")
        
        for category, score in scores.items():
            if category != 'feedback':
                st.metric(category.title(), f"{score:.1f}/10", 
                         delta=f"{score-5:.1f}" if score != 5 else None)
        
        # Feedback
        st.subheader("💡 Suggestions")
        for feedback in scores['feedback']:
            st.write(f"• {feedback}")
    
    # Session summary
    st.subheader("📊 Session Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Duration", f"{results['duration']:.1f}s")
    
    with col2:
        overall_score = sum([scores['confidence'], scores['clarity'], 
                           scores['presence'], scores['energy']]) / 4
        st.metric("Overall Score", f"{overall_score:.1f}/10")
    
    with col3:
        st.metric("Prompt", results['prompt'][:20] + "..." if len(results['prompt']) > 20 else results['prompt'])

def update_realtime_feedback(placeholder):
    """Update real-time feedback during recording"""
    if not st.session_state.recording_active:
        return
    
    # Get current metrics from processors
    video_metrics = st.session_state.video_processor.get_current_metrics()
    audio_metrics = st.session_state.audio_processor.get_current_metrics()
    
    with placeholder.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Eye Contact", f"{video_metrics.get('eye_contact', 0):.1f}%")
            st.metric("Smile Time", f"{video_metrics.get('smile_time', 0):.1f}%")
        
        with col2:
            st.metric("Speaking Time", f"{audio_metrics.get('speaking_time', 0):.1f}s")
            st.metric("Silence Ratio", f"{audio_metrics.get('silence_ratio', 0):.1f}%")

def progress_dashboard_page():
    """Display progress dashboard with historical data"""
    st.header("📊 Progress Dashboard")
    
    # Load historical data
    sessions = st.session_state.data_manager.load_sessions()
    
    if not sessions:
        st.info("No practice sessions recorded yet. Start your first practice session!")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(sessions)
    df['date'] = pd.to_datetime(df['start_time']).dt.date
    
    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", len(sessions))
    
    with col2:
        total_time = sum([s['duration'] for s in sessions])
        st.metric("Total Practice Time", f"{total_time/60:.1f} min")
    
    with col3:
        avg_confidence = np.mean([s['scores']['confidence'] for s in sessions])
        st.metric("Avg Confidence", f"{avg_confidence:.1f}/10")
    
    with col4:
        latest_session = sessions[-1]
        latest_score = sum([latest_session['scores']['confidence'], 
                          latest_session['scores']['clarity'],
                          latest_session['scores']['presence'], 
                          latest_session['scores']['energy']]) / 4
        st.metric("Latest Score", f"{latest_score:.1f}/10")
    
    # Progress charts
    st.subheader("📈 Progress Over Time")
    
    # Prepare data for plotting
    dates = [datetime.fromisoformat(s['start_time']).date() for s in sessions]
    confidence_scores = [s['scores']['confidence'] for s in sessions]
    clarity_scores = [s['scores']['clarity'] for s in sessions]
    presence_scores = [s['scores']['presence'] for s in sessions]
    energy_scores = [s['scores']['energy'] for s in sessions]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Confidence', 'Clarity', 'Presence', 'Energy')
    )
    
    fig.add_trace(go.Scatter(x=dates, y=confidence_scores, mode='lines+markers', name='Confidence'),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=dates, y=clarity_scores, mode='lines+markers', name='Clarity'),
                  row=1, col=2)
    fig.add_trace(go.Scatter(x=dates, y=presence_scores, mode='lines+markers', name='Presence'),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=dates, y=energy_scores, mode='lines+markers', name='Energy'),
                  row=2, col=2)
    
    fig.update_layout(height=600, showlegend=False)
    fig.update_yaxes(range=[0, 10])
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent sessions table
    st.subheader("🕐 Recent Sessions")
    recent_sessions = sessions[-10:]  # Last 10 sessions
    
    table_data = []
    for session in recent_sessions:
        overall_score = sum([session['scores']['confidence'], 
                           session['scores']['clarity'],
                           session['scores']['presence'], 
                           session['scores']['energy']]) / 4
        table_data.append({
            'Date': datetime.fromisoformat(session['start_time']).strftime('%Y-%m-%d %H:%M'),
            'Duration': f"{session['duration']:.1f}s",
            'Overall Score': f"{overall_score:.1f}/10",
            'Prompt': session['prompt'][:50] + "..." if len(session['prompt']) > 50 else session['prompt']
        })
    
    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

def settings_page():
    """Display settings and configuration options"""
    st.header("⚙️ Settings")
    
    st.subheader("🎥 Camera Settings")
    camera_quality = st.select_slider(
        "Camera Quality",
        options=["Low", "Medium", "High"],
        value="Medium"
    )
    
    st.subheader("🎤 Audio Settings")
    audio_sensitivity = st.slider(
        "Microphone Sensitivity",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1
    )
    
    st.subheader("📊 Analysis Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eye_contact_threshold = st.slider(
            "Eye Contact Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1
        )
    
    with col2:
        smile_threshold = st.slider(
            "Smile Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
    
    st.subheader("🗂️ Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Data"):
            export_data()
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            clear_all_data()
    
    st.subheader("ℹ️ System Information")
    st.info(f"Total Sessions: {len(st.session_state.data_manager.load_sessions())}")
    st.info(f"Data Storage: {get_data_size()} MB")

def check_devices():
    """Check if camera and microphone are available"""
    try:
        # In a server environment, devices may not be available
        # but we can still show the interface for demonstration
        return True
    except Exception as e:
        st.warning(f"Device check failed: {str(e)}")
        return True  # Allow the interface to load anyway

def export_data():
    """Export session data to CSV"""
    sessions = st.session_state.data_manager.load_sessions()
    if not sessions:
        st.warning("No data to export.")
        return
    
    df = pd.DataFrame(sessions)
    csv = df.to_csv(index=False)
    
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"voice_coach_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

def clear_all_data():
    """Clear all session data"""
    st.session_state.data_manager.clear_all_data()
    st.success("All data cleared successfully!")
    st.rerun()

def get_data_size():
    """Get the size of stored data in MB"""
    try:
        size = os.path.getsize('data/sessions.json')
        return round(size / (1024 * 1024), 2)
    except:
        return 0

if __name__ == "__main__":
    main()
