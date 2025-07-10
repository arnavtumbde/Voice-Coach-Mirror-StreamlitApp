import json
import os
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import shutil

class DataManager:
    def __init__(self, data_file="data/sessions.json"):
        self.data_file = data_file
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        
        # Create empty sessions file if it doesn't exist
        if not os.path.exists(self.data_file):
            self.save_sessions([])
    
    def save_session(self, session_data: Dict[str, Any]):
        """Save a single session to the data file"""
        try:
            # Load existing sessions
            sessions = self.load_sessions()
            
            # Prepare session data for storage
            session_record = {
                'session_id': self.generate_session_id(),
                'start_time': session_data['start_time'].isoformat(),
                'end_time': session_data['end_time'].isoformat(),
                'duration': session_data['duration'],
                'prompt': session_data['prompt'],
                'scores': session_data['scores'],
                'video_summary': self.summarize_video_data(session_data.get('video_data', [])),
                'audio_summary': self.summarize_audio_data(session_data.get('audio_data', []))
            }
            
            # Add to sessions list
            sessions.append(session_record)
            
            # Save updated sessions
            self.save_sessions(sessions)
            
            return session_record['session_id']
            
        except Exception as e:
            print(f"Error saving session: {e}")
            return None
    
    def load_sessions(self) -> List[Dict[str, Any]]:
        """Load all sessions from the data file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            print(f"Error loading sessions: {e}")
            return []
    
    def save_sessions(self, sessions: List[Dict[str, Any]]):
        """Save sessions list to the data file"""
        try:
            with open(self.data_file, 'w') as f:
                json.dump(sessions, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving sessions: {e}")
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    
    def summarize_video_data(self, video_data: List[Dict]) -> Dict[str, Any]:
        """Create a summary of video data for storage"""
        if not video_data:
            return {
                'total_frames': 0,
                'face_detection_rate': 0,
                'eye_contact_percentage': 0,
                'smile_percentage': 0,
                'average_confidence': 0
            }
        
        frames_with_face = sum(1 for frame in video_data if frame['frame_data']['face_detected'])
        frames_with_eye_contact = sum(1 for frame in video_data if frame['frame_data']['eye_contact'])
        frames_with_smile = sum(1 for frame in video_data if frame['frame_data']['smile_detected'])
        
        total_frames = len(video_data)
        
        # Calculate average confidence
        confidences = [frame['frame_data']['confidence'] for frame in video_data 
                      if frame['frame_data']['face_detected']]
        average_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        return {
            'total_frames': total_frames,
            'face_detection_rate': (frames_with_face / total_frames) * 100,
            'eye_contact_percentage': (frames_with_eye_contact / total_frames) * 100,
            'smile_percentage': (frames_with_smile / total_frames) * 100,
            'average_confidence': average_confidence
        }
    
    def summarize_audio_data(self, audio_data: List[Dict]) -> Dict[str, Any]:
        """Create a summary of audio data for storage"""
        if not audio_data:
            return {
                'total_chunks': 0,
                'average_volume': 0,
                'speech_segments': 0,
                'estimated_words': 0
            }
        
        total_chunks = len(audio_data)
        volumes = []
        speech_segments = 0
        
        for chunk in audio_data:
            audio_chunk = chunk['audio_chunk']
            volume = (sum(x**2 for x in audio_chunk) / len(audio_chunk))**0.5
            volumes.append(volume)
            
            if volume > 0.01:  # Speech threshold
                speech_segments += 1
        
        average_volume = sum(volumes) / len(volumes) if volumes else 0
        estimated_words = speech_segments / 3.5  # Rough estimation
        
        return {
            'total_chunks': total_chunks,
            'average_volume': average_volume,
            'speech_segments': speech_segments,
            'estimated_words': estimated_words
        }
    
    def get_session_by_id(self, session_id: str) -> Dict[str, Any]:
        """Get a specific session by ID"""
        sessions = self.load_sessions()
        for session in sessions:
            if session['session_id'] == session_id:
                return session
        return None
    
    def get_recent_sessions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent sessions"""
        sessions = self.load_sessions()
        return sessions[-limit:] if len(sessions) >= limit else sessions
    
    def get_sessions_by_date_range(self, start_date: str, end_date: str) -> List[Dict[str, Any]]:
        """Get sessions within a date range"""
        sessions = self.load_sessions()
        filtered_sessions = []
        
        for session in sessions:
            session_date = datetime.fromisoformat(session['start_time']).date()
            start_dt = datetime.fromisoformat(start_date).date()
            end_dt = datetime.fromisoformat(end_date).date()
            
            if start_dt <= session_date <= end_dt:
                filtered_sessions.append(session)
        
        return filtered_sessions
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        sessions = self.load_sessions()
        
        if not sessions:
            return {
                'total_sessions': 0,
                'total_practice_time': 0,
                'average_scores': {
                    'confidence': 0,
                    'clarity': 0,
                    'presence': 0,
                    'energy': 0
                },
                'improvement_trends': {}
            }
        
        # Calculate statistics
        total_sessions = len(sessions)
        total_practice_time = sum(session['duration'] for session in sessions)
        
        # Calculate average scores
        confidence_scores = [session['scores']['confidence'] for session in sessions]
        clarity_scores = [session['scores']['clarity'] for session in sessions]
        presence_scores = [session['scores']['presence'] for session in sessions]
        energy_scores = [session['scores']['energy'] for session in sessions]
        
        average_scores = {
            'confidence': sum(confidence_scores) / len(confidence_scores),
            'clarity': sum(clarity_scores) / len(clarity_scores),
            'presence': sum(presence_scores) / len(presence_scores),
            'energy': sum(energy_scores) / len(energy_scores)
        }
        
        # Calculate improvement trends
        improvement_trends = self.calculate_improvement_trends(sessions)
        
        return {
            'total_sessions': total_sessions,
            'total_practice_time': total_practice_time,
            'average_scores': average_scores,
            'improvement_trends': improvement_trends
        }
    
    def calculate_improvement_trends(self, sessions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate improvement trends over time"""
        if len(sessions) < 2:
            return {
                'confidence_trend': 0,
                'clarity_trend': 0,
                'presence_trend': 0,
                'energy_trend': 0
            }
        
        # Sort sessions by date
        sorted_sessions = sorted(sessions, key=lambda x: x['start_time'])
        
        # Calculate trends (simple linear trend)
        def calculate_trend(scores):
            if len(scores) < 2:
                return 0
            
            # Simple slope calculation
            n = len(scores)
            x_sum = sum(range(n))
            y_sum = sum(scores)
            xy_sum = sum(i * scores[i] for i in range(n))
            x_sq_sum = sum(i * i for i in range(n))
            
            if n * x_sq_sum - x_sum * x_sum == 0:
                return 0
            
            slope = (n * xy_sum - x_sum * y_sum) / (n * x_sq_sum - x_sum * x_sum)
            return slope
        
        confidence_scores = [session['scores']['confidence'] for session in sorted_sessions]
        clarity_scores = [session['scores']['clarity'] for session in sorted_sessions]
        presence_scores = [session['scores']['presence'] for session in sorted_sessions]
        energy_scores = [session['scores']['energy'] for session in sorted_sessions]
        
        return {
            'confidence_trend': calculate_trend(confidence_scores),
            'clarity_trend': calculate_trend(clarity_scores),
            'presence_trend': calculate_trend(presence_scores),
            'energy_trend': calculate_trend(energy_scores)
        }
    
    def export_to_csv(self, output_file: str = None) -> str:
        """Export sessions data to CSV format"""
        sessions = self.load_sessions()
        
        if not sessions:
            return None
        
        # Flatten the data for CSV export
        csv_data = []
        for session in sessions:
            row = {
                'session_id': session['session_id'],
                'start_time': session['start_time'],
                'end_time': session['end_time'],
                'duration': session['duration'],
                'prompt': session['prompt'],
                'confidence_score': session['scores']['confidence'],
                'clarity_score': session['scores']['clarity'],
                'presence_score': session['scores']['presence'],
                'energy_score': session['scores']['energy'],
                'face_detection_rate': session['video_summary']['face_detection_rate'],
                'eye_contact_percentage': session['video_summary']['eye_contact_percentage'],
                'smile_percentage': session['video_summary']['smile_percentage'],
                'average_confidence': session['video_summary']['average_confidence'],
                'speech_segments': session['audio_summary']['speech_segments'],
                'estimated_words': session['audio_summary']['estimated_words']
            }
            csv_data.append(row)
        
        # Create DataFrame and export
        df = pd.DataFrame(csv_data)
        
        if output_file is None:
            output_file = f"voice_coach_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df.to_csv(output_file, index=False)
        return output_file
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a specific session"""
        try:
            sessions = self.load_sessions()
            sessions = [s for s in sessions if s['session_id'] != session_id]
            self.save_sessions(sessions)
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def clear_all_data(self):
        """Clear all session data"""
        try:
            self.save_sessions([])
            return True
        except Exception as e:
            print(f"Error clearing data: {e}")
            return False
    
    def backup_data(self, backup_file: str = None):
        """Create a backup of session data"""
        try:
            if backup_file is None:
                backup_file = f"backup_sessions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            shutil.copy2(self.data_file, backup_file)
            return backup_file
        except Exception as e:
            print(f"Error creating backup: {e}")
            return None
    
    def restore_data(self, backup_file: str):
        """Restore session data from backup"""
        try:
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, self.data_file)
                return True
            return False
        except Exception as e:
            print(f"Error restoring data: {e}")
            return False
