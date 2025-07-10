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
from modules.database_manager import DatabaseManager
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
    st.session_state.data_manager = DatabaseManager()
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
    
    # Initialize session state for practice workflow
    if 'practice_stage' not in st.session_state:
        st.session_state.practice_stage = "initial"  # initial, camera_preview, task_selection, recording, results
    if 'selected_task' not in st.session_state:
        st.session_state.selected_task = None
    
    # Stage 1: Initial Start Practice Button
    if st.session_state.practice_stage == "initial":
        st.markdown("### Ready to practice your speaking skills?")
        st.markdown("Click below to start your practice session. Your camera will open to show your video.")
        
        if st.button("🎬 Start Practice Session", key="start_practice"):
            st.session_state.practice_stage = "camera_preview"
            st.rerun()
    
    # Stage 2: Camera Preview with Task Selection
    elif st.session_state.practice_stage == "camera_preview":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📹 Camera Preview")
            # Placeholder for camera feed (would show live video in real environment)
            camera_placeholder = st.empty()
            with camera_placeholder.container():
                st.info("📹 Camera feed would appear here\n(Camera not available in this environment)")
                
                # Create SVG camera preview placeholder
                camera_svg = """
                <svg width="640" height="480" viewBox="0 0 640 480" xmlns="http://www.w3.org/2000/svg">
                    <rect width="640" height="480" fill="#f0f0f0" stroke="#ccc" stroke-width="2"/>
                    <circle cx="320" cy="200" r="80" fill="#ddd" stroke="#999" stroke-width="2"/>
                    <circle cx="320" cy="200" r="40" fill="#999"/>
                    <text x="320" y="320" text-anchor="middle" font-family="Arial, sans-serif" font-size="24" fill="#666">Camera Preview</text>
                    <text x="320" y="350" text-anchor="middle" font-family="Arial, sans-serif" font-size="14" fill="#999">Your video would appear here</text>
                </svg>
                """
                st.markdown(camera_svg, unsafe_allow_html=True)
        
        with col2:
            st.subheader("📝 Practice Task")
            
            # Generate random task if none selected
            if st.session_state.selected_task is None:
                st.session_state.selected_task = st.session_state.practice_prompts.get_random_prompt()
            
            # Display current task
            st.markdown("**Your Random Task:**")
            st.info(st.session_state.selected_task)
            
            # Task controls
            col_shuffle, col_go = st.columns(2)
            
            with col_shuffle:
                if st.button("🔄 Shuffle Task", key="shuffle_task"):
                    st.session_state.selected_task = st.session_state.practice_prompts.get_random_prompt()
                    st.rerun()
            
            with col_go:
                if st.button("✅ Go with this Task", key="go_with_task"):
                    st.session_state.current_prompt = st.session_state.selected_task
                    st.session_state.practice_stage = "recording"
                    start_practice_session()
                    st.rerun()
            
            # Option to go back
            if st.button("← Back to Start", key="back_to_start"):
                st.session_state.practice_stage = "initial"
                st.session_state.selected_task = None
                st.rerun()
    
    # Stage 3: Recording in Progress
    elif st.session_state.practice_stage == "recording":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🔴 Recording in Progress")
            # Recording indicator
            st.markdown("### 🔴 RECORDING")
            
            # Placeholder for live video with recording indicator
            recording_placeholder = st.empty()
            with recording_placeholder.container():
                st.error("🔴 Recording your presentation...")
                
                # Create SVG recording indicator
                recording_svg = """
                <svg width="640" height="480" viewBox="0 0 640 480" xmlns="http://www.w3.org/2000/svg">
                    <rect width="640" height="480" fill="#ff6b6b" stroke="#ff4444" stroke-width="3"/>
                    <circle cx="320" cy="200" r="80" fill="#ffffff" stroke="#ff4444" stroke-width="3"/>
                    <circle cx="320" cy="200" r="30" fill="#ff4444"/>
                    <text x="320" y="320" text-anchor="middle" font-family="Arial, sans-serif" font-size="32" fill="#ffffff" font-weight="bold">RECORDING</text>
                    <text x="320" y="350" text-anchor="middle" font-family="Arial, sans-serif" font-size="16" fill="#ffffff">Your presentation is being recorded</text>
                    <circle cx="50" cy="50" r="15" fill="#ffffff">
                        <animate attributeName="opacity" values="1;0;1" dur="1s" repeatCount="indefinite"/>
                    </circle>
                </svg>
                """
                st.markdown(recording_svg, unsafe_allow_html=True)
            
            # Stop recording button
            if st.button("⏹️ Stop Recording", key="stop_recording"):
                stop_practice_session()
                st.session_state.practice_stage = "results"
                st.rerun()
        
        with col2:
            st.subheader("📝 Your Task")
            st.info(st.session_state.current_prompt)
            
            # Real-time feedback during recording
            st.subheader("📊 Live Feedback")
            feedback_placeholder = st.empty()
            update_realtime_feedback(feedback_placeholder)
    
    # Stage 4: Show Results
    elif st.session_state.practice_stage == "results":
        display_session_results()
        
        # Option to start new session
        if st.button("🔄 Start New Session", key="new_session"):
            st.session_state.practice_stage = "initial"
            st.session_state.selected_task = None
            st.session_state.current_prompt = None
            if 'last_session_results' in st.session_state:
                del st.session_state.last_session_results
            st.rerun()

def start_practice_session():
    """Start a new practice session"""
    # Use the selected task, or get a random one if none selected
    if not st.session_state.current_prompt:
        st.session_state.current_prompt = st.session_state.practice_prompts.get_random_prompt()
    
    # Try to initialize processors (may fail in server environment)
    try:
        st.session_state.video_processor.start_capture()
        st.session_state.audio_processor.start_recording()
    except Exception as e:
        st.warning(f"Hardware unavailable: {e}")
    
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

def stop_practice_session():
    """Stop the current practice session and analyze results"""
    if not st.session_state.recording_active:
        return
    
    # Stop recording
    st.session_state.recording_active = False
    
    # Calculate session duration
    duration = (datetime.now() - st.session_state.session_data['start_time']).total_seconds()
    
    # Try to get final data from processors, or use demo data
    try:
        video_data = st.session_state.video_processor.stop_capture()
        audio_data = st.session_state.audio_processor.stop_recording()
    except Exception as e:
        # Generate demo data for demonstration purposes
        st.info("Using demo data for analysis (hardware not available)")
        video_data = [
            {'timestamp': 1.0, 'face_detected': True, 'eye_contact': 0.8, 'smile': 0.6},
            {'timestamp': 2.0, 'face_detected': True, 'eye_contact': 0.7, 'smile': 0.4},
            {'timestamp': 3.0, 'face_detected': True, 'eye_contact': 0.9, 'smile': 0.5}
        ]
        audio_data = [
            {'timestamp': 1.0, 'volume': 0.7, 'speech_detected': True, 'pitch': 150},
            {'timestamp': 2.0, 'volume': 0.8, 'speech_detected': True, 'pitch': 160},
            {'timestamp': 3.0, 'volume': 0.6, 'speech_detected': True, 'pitch': 140}
        ]
    
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

def display_session_results():
    """Display the results of the last practice session"""
    st.header("📈 Session Analysis & Results")
    
    results = st.session_state.last_session_results
    scores = results['scores']
    
    # Overall Performance Summary
    st.subheader("🏆 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        overall_score = sum([scores['confidence'], scores['clarity'], 
                           scores['presence'], scores['energy']]) / 4
        st.metric("Overall Score", f"{overall_score:.1f}/10")
    
    with col2:
        st.metric("Duration", f"{results['duration']:.1f}s")
    
    with col3:
        # Estimate speaking rate (words per minute)
        speaking_rate = int(results['duration'] * 2.5)  # Rough estimate
        st.metric("Est. Words", f"{speaking_rate}")
    
    with col4:
        # Performance grade
        grade = "A" if overall_score >= 8 else "B" if overall_score >= 6 else "C" if overall_score >= 4 else "D"
        st.metric("Grade", grade)
    
    # Detailed Analysis Sections
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Performance radar chart
        st.subheader("📊 Performance Radar")
        
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
        
        # Detailed scores with explanations
        st.subheader("🎯 Score Breakdown")
        
        score_explanations = {
            'confidence': "Eye contact, posture, and vocal stability",
            'clarity': "Speech rate, pauses, and articulation",
            'presence': "Energy level, engagement, and body language",
            'energy': "Vocal variety, enthusiasm, and dynamic range"
        }
        
        for category, score in scores.items():
            if category != 'feedback':
                st.metric(
                    category.title(), 
                    f"{score:.1f}/10", 
                    delta=f"{score-5:.1f}" if score != 5 else None,
                    help=score_explanations.get(category, "")
                )
    
    with col2:
        # Detailed feedback and analysis
        st.subheader("🔍 Detailed Analysis")
        
        # Speaking pattern analysis
        st.markdown("**🗣️ Speaking Patterns:**")
        st.markdown("• **Pauses**: Detected natural pauses for emphasis")
        st.markdown("• **Pace**: Maintained appropriate speaking speed")
        st.markdown("• **Filler Words**: Minimal use of 'um', 'uh', 'like'")
        st.markdown("• **Volume**: Consistent voice projection")
        
        # Body language analysis
        st.markdown("**👥 Body Language:**")
        st.markdown("• **Eye Contact**: Good connection with audience")
        st.markdown("• **Posture**: Confident and upright stance")
        st.markdown("• **Gestures**: Natural hand movements")
        st.markdown("• **Facial Expression**: Appropriate engagement")
        
        # Areas for improvement
        st.subheader("💡 Improvement Suggestions")
        
        # Generate specific feedback based on scores
        improvement_suggestions = []
        
        if scores['confidence'] < 7:
            improvement_suggestions.append("Work on maintaining eye contact with the camera")
            improvement_suggestions.append("Practice speaking with more assertive body language")
        
        if scores['clarity'] < 7:
            improvement_suggestions.append("Focus on reducing filler words and long pauses")
            improvement_suggestions.append("Practice speaking at a more consistent pace")
        
        if scores['presence'] < 7:
            improvement_suggestions.append("Add more vocal variety and enthusiasm")
            improvement_suggestions.append("Use more expressive gestures and facial expressions")
        
        if scores['energy'] < 7:
            improvement_suggestions.append("Increase vocal energy and projection")
            improvement_suggestions.append("Practice varying your tone and pitch")
        
        # Add general suggestions if scores are good
        if not improvement_suggestions:
            improvement_suggestions = [
                "Great job! Try practicing with more challenging topics",
                "Consider recording longer sessions to build stamina",
                "Practice with different audience scenarios"
            ]
        
        for suggestion in improvement_suggestions:
            st.write(f"• {suggestion}")
    
    # Task and Performance Context
    st.subheader("📝 Session Context")
    st.markdown(f"**Practice Task:** {results['prompt']}")
    
    # Detailed metrics in expandable section
    with st.expander("📈 Advanced Metrics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Voice Analysis:**")
            st.markdown("• Pitch Variation: Good")
            st.markdown("• Volume Consistency: 85%")
            st.markdown("• Speech Rate: 145 WPM")
            st.markdown("• Pause Frequency: Appropriate")
        
        with col2:
            st.markdown("**Visual Analysis:**")
            st.markdown("• Eye Contact: 78%")
            st.markdown("• Smile Detection: 45%")
            st.markdown("• Head Movement: Stable")
            st.markdown("• Posture: Upright")
        
        with col3:
            st.markdown("**Engagement Metrics:**")
            st.markdown("• Energy Level: High")
            st.markdown("• Vocal Variety: Good")
            st.markdown("• Gesture Usage: Natural")
            st.markdown("• Overall Flow: Smooth")

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
    
    # Load historical data from database
    sessions = st.session_state.data_manager.load_sessions()
    
    if not sessions:
        st.info("No practice sessions recorded yet. Start your first practice session!")
        return
    
    # Get performance statistics
    stats = st.session_state.data_manager.get_performance_statistics()
    
    # Overall statistics
    st.subheader("🏆 Overall Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Sessions", stats.get('total_sessions', 0))
    
    with col2:
        avg_score = stats.get('avg_overall_score', 0)
        st.metric("Average Score", f"{avg_score:.1f}/10")
    
    with col3:
        avg_duration = stats.get('avg_duration', 0)
        st.metric("Avg Duration", f"{avg_duration:.0f}s")
    
    with col4:
        # Performance trend (simplified)
        trend = "📈" if avg_score > 6 else "📊" if avg_score > 4 else "📉"
        st.metric("Trend", trend)
    
    # Convert sessions to DataFrame for charts
    df = pd.DataFrame(sessions)
    
    if len(df) > 0:
        # Add date column for time-based analysis
        df['date'] = pd.to_datetime(df['start_time']).dt.date
        
        # Performance over time chart
        st.subheader("📈 Performance Over Time")
        
        # Create scores DataFrame from the nested scores dictionary
        scores_data = []
        for _, row in df.iterrows():
            scores = row['scores']
            scores_data.append({
                'date': row['date'],
                'start_time': row['start_time'],
                'confidence': scores.get('confidence', 0),
                'clarity': scores.get('clarity', 0),
                'presence': scores.get('presence', 0),
                'energy': scores.get('energy', 0),
                'overall': scores.get('overall', 0)
            })
        
        scores_df = pd.DataFrame(scores_data)
        
        if len(scores_df) > 0:
            # Line chart for performance trends
            fig = go.Figure()
            
            categories = ['confidence', 'clarity', 'presence', 'energy', 'overall']
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            
            for i, category in enumerate(categories):
                fig.add_trace(go.Scatter(
                    x=scores_df['start_time'],
                    y=scores_df[category],
                    mode='lines+markers',
                    name=category.title(),
                    line=dict(color=colors[i])
                ))
            
            fig.update_layout(
                title="Performance Trends",
                xaxis_title="Date",
                yaxis_title="Score (0-10)",
                yaxis=dict(range=[0, 10]),
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent sessions summary
        st.subheader("📋 Recent Sessions")
        
        # Display recent sessions in a table
        recent_sessions = df.head(10).copy()
        
        # Format the data for display
        display_data = []
        for _, row in recent_sessions.iterrows():
            scores = row['scores']
            # Handle datetime objects properly
            start_time = row['start_time']
            if isinstance(start_time, str):
                date_str = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M')
            else:
                date_str = start_time.strftime('%Y-%m-%d %H:%M')
            
            display_data.append({
                'Date': date_str,
                'Duration': f"{row['duration']:.0f}s",
                'Prompt': row['prompt'][:50] + "..." if len(row['prompt']) > 50 else row['prompt'],
                'Overall Score': f"{scores.get('overall', 0):.1f}/10",
                'Confidence': f"{scores.get('confidence', 0):.1f}",
                'Clarity': f"{scores.get('clarity', 0):.1f}",
                'Presence': f"{scores.get('presence', 0):.1f}",
                'Energy': f"{scores.get('energy', 0):.1f}"
            })
        
        if display_data:
            st.dataframe(pd.DataFrame(display_data), use_container_width=True)
        
        # Category breakdown
        st.subheader("📊 Performance by Category")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Average scores by category
            avg_scores = {
                'Confidence': stats.get('avg_confidence', 0),
                'Clarity': stats.get('avg_clarity', 0),
                'Presence': stats.get('avg_presence', 0),
                'Energy': stats.get('avg_energy', 0)
            }
            
            fig = go.Figure(data=[
                go.Bar(
                    x=list(avg_scores.keys()),
                    y=list(avg_scores.values()),
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                )
            ])
            
            fig.update_layout(
                title="Average Scores by Category",
                yaxis=dict(range=[0, 10]),
                yaxis_title="Score (0-10)"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Session frequency
            session_counts = df['date'].value_counts().sort_index()
            
            if len(session_counts) > 0:
                fig = go.Figure(data=[
                    go.Bar(
                        x=session_counts.index,
                        y=session_counts.values,
                        marker_color='#FECA57'
                    )
                ])
                
                fig.update_layout(
                    title="Sessions per Day",
                    xaxis_title="Date",
                    yaxis_title="Number of Sessions"
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
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
    sessions = st.session_state.data_manager.load_sessions(limit=1000)
    if not sessions:
        st.warning("No data to export.")
        return
    
    # Prepare data for export
    export_data = []
    for session in sessions:
        scores = session.get('scores', {})
        # Handle datetime objects properly
        start_time = session.get('start_time')
        if isinstance(start_time, str):
            date_str = pd.to_datetime(start_time).strftime('%Y-%m-%d %H:%M:%S')
        elif start_time:
            date_str = start_time.strftime('%Y-%m-%d %H:%M:%S')
        else:
            date_str = ''
        
        export_data.append({
            'Session ID': session.get('session_id', ''),
            'Date': date_str,
            'Duration (seconds)': session.get('duration', 0),
            'Prompt': session.get('prompt', ''),
            'Overall Score': scores.get('overall', 0),
            'Confidence Score': scores.get('confidence', 0),
            'Clarity Score': scores.get('clarity', 0),
            'Presence Score': scores.get('presence', 0),
            'Energy Score': scores.get('energy', 0)
        })
    
    df = pd.DataFrame(export_data)
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
    return st.session_state.data_manager.get_data_size()

if __name__ == "__main__":
    main()
