"""
Database manager for Voice Coach Mirror using PostgreSQL
"""

import os
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd


class DatabaseManager:
    def __init__(self):
        """Initialize database connection"""
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self.default_user_id = 1  # Using default user for simplicity
    
    def save_session(self, session_data: Dict[str, Any]) -> str:
        """Save a complete practice session to the database"""
        try:
            session_id = str(uuid.uuid4())
            
            with self.engine.connect() as conn:
                # Insert practice session
                conn.execute(text("""
                    INSERT INTO practice_sessions 
                    (user_id, session_id, prompt, start_time, end_time, duration_seconds)
                    VALUES (:user_id, :session_id, :prompt, :start_time, :end_time, :duration)
                """), {
                    'user_id': self.default_user_id,
                    'session_id': session_id,
                    'prompt': session_data.get('prompt', ''),
                    'start_time': session_data.get('start_time', datetime.now()),
                    'end_time': session_data.get('end_time', datetime.now()),
                    'duration': session_data.get('duration', 0)
                })
                
                # Insert scores if available
                if 'scores' in session_data:
                    scores = session_data['scores']
                    overall_score = sum([
                        scores.get('confidence', 0),
                        scores.get('clarity', 0),
                        scores.get('presence', 0),
                        scores.get('energy', 0)
                    ]) / 4
                    
                    conn.execute(text("""
                        INSERT INTO session_scores 
                        (session_id, confidence_score, clarity_score, presence_score, energy_score, overall_score)
                        VALUES (:session_id, :confidence, :clarity, :presence, :energy, :overall)
                    """), {
                        'session_id': session_id,
                        'confidence': scores.get('confidence', 0),
                        'clarity': scores.get('clarity', 0),
                        'presence': scores.get('presence', 0),
                        'energy': scores.get('energy', 0),
                        'overall': overall_score
                    })
                    
                    # Insert feedback if available
                    if 'feedback' in scores:
                        for feedback_text in scores['feedback']:
                            conn.execute(text("""
                                INSERT INTO session_feedback 
                                (session_id, feedback_type, feedback_text, category)
                                VALUES (:session_id, :feedback_type, :feedback_text, :category)
                            """), {
                                'session_id': session_id,
                                'feedback_type': 'suggestion',
                                'feedback_text': feedback_text,
                                'category': 'general'
                            })
                
                # Insert detailed metrics
                self._save_metrics(conn, session_id, session_data.get('video_data', []), 'video')
                self._save_metrics(conn, session_id, session_data.get('audio_data', []), 'audio')
                
                conn.commit()
                return session_id
                
        except SQLAlchemyError as e:
            print(f"Database error saving session: {e}")
            return None
    
    def _save_metrics(self, conn, session_id: str, data: List[Dict], data_type: str):
        """Save detailed metrics data"""
        for item in data:
            timestamp = item.get('timestamp', 0)
            
            # Save different types of metrics based on data type
            if data_type == 'video':
                metrics = {
                    'eye_contact': item.get('eye_contact', 0),
                    'smile': item.get('smile', 0),
                    'face_detected': float(item.get('face_detected', False))
                }
            elif data_type == 'audio':
                metrics = {
                    'volume': item.get('volume', 0),
                    'pitch': item.get('pitch', 0),
                    'speech_detected': float(item.get('speech_detected', False))
                }
            else:
                continue
            
            for metric_name, metric_value in metrics.items():
                conn.execute(text("""
                    INSERT INTO session_metrics 
                    (session_id, metric_type, metric_value, timestamp_seconds)
                    VALUES (:session_id, :metric_type, :metric_value, :timestamp)
                """), {
                    'session_id': session_id,
                    'metric_type': metric_name,
                    'metric_value': metric_value,
                    'timestamp': timestamp
                })
    
    def load_sessions(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Load recent practice sessions"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        ps.session_id,
                        ps.prompt,
                        ps.start_time,
                        ps.end_time,
                        ps.duration_seconds,
                        ss.confidence_score,
                        ss.clarity_score,
                        ss.presence_score,
                        ss.energy_score,
                        ss.overall_score
                    FROM practice_sessions ps
                    LEFT JOIN session_scores ss ON ps.session_id = ss.session_id
                    WHERE ps.user_id = :user_id
                    ORDER BY ps.created_at DESC
                    LIMIT :limit
                """), {
                    'user_id': self.default_user_id,
                    'limit': limit
                })
                
                sessions = []
                for row in result:
                    session = {
                        'session_id': row[0],
                        'prompt': row[1],
                        'start_time': row[2],
                        'end_time': row[3],
                        'duration': row[4],
                        'scores': {
                            'confidence': row[5] or 0,
                            'clarity': row[6] or 0,
                            'presence': row[7] or 0,
                            'energy': row[8] or 0,
                            'overall': row[9] or 0
                        }
                    }
                    sessions.append(session)
                
                return sessions
                
        except SQLAlchemyError as e:
            print(f"Database error loading sessions: {e}")
            return []
    
    def get_session_by_id(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific session by ID"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        ps.session_id,
                        ps.prompt,
                        ps.start_time,
                        ps.end_time,
                        ps.duration_seconds,
                        ss.confidence_score,
                        ss.clarity_score,
                        ss.presence_score,
                        ss.energy_score,
                        ss.overall_score
                    FROM practice_sessions ps
                    LEFT JOIN session_scores ss ON ps.session_id = ss.session_id
                    WHERE ps.session_id = :session_id
                """), {'session_id': session_id})
                
                row = result.fetchone()
                if row:
                    return {
                        'session_id': row[0],
                        'prompt': row[1],
                        'start_time': row[2],
                        'end_time': row[3],
                        'duration': row[4],
                        'scores': {
                            'confidence': row[5] or 0,
                            'clarity': row[6] or 0,
                            'presence': row[7] or 0,
                            'energy': row[8] or 0,
                            'overall': row[9] or 0
                        }
                    }
                return None
                
        except SQLAlchemyError as e:
            print(f"Database error getting session: {e}")
            return None
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get overall performance statistics"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        COUNT(*) as total_sessions,
                        AVG(ss.overall_score) as avg_overall_score,
                        AVG(ss.confidence_score) as avg_confidence,
                        AVG(ss.clarity_score) as avg_clarity,
                        AVG(ss.presence_score) as avg_presence,
                        AVG(ss.energy_score) as avg_energy,
                        AVG(ps.duration_seconds) as avg_duration
                    FROM practice_sessions ps
                    LEFT JOIN session_scores ss ON ps.session_id = ss.session_id
                    WHERE ps.user_id = :user_id
                """), {'user_id': self.default_user_id})
                
                row = result.fetchone()
                if row:
                    return {
                        'total_sessions': row[0] or 0,
                        'avg_overall_score': row[1] or 0,
                        'avg_confidence': row[2] or 0,
                        'avg_clarity': row[3] or 0,
                        'avg_presence': row[4] or 0,
                        'avg_energy': row[5] or 0,
                        'avg_duration': row[6] or 0
                    }
                return {}
                
        except SQLAlchemyError as e:
            print(f"Database error getting statistics: {e}")
            return {}
    
    def get_progress_over_time(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get performance progress over time"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        DATE(ps.created_at) as session_date,
                        AVG(ss.overall_score) as avg_score,
                        COUNT(*) as session_count
                    FROM practice_sessions ps
                    LEFT JOIN session_scores ss ON ps.session_id = ss.session_id
                    WHERE ps.user_id = :user_id
                    AND ps.created_at >= CURRENT_DATE - INTERVAL '%s days'
                    GROUP BY DATE(ps.created_at)
                    ORDER BY session_date
                """ % days), {'user_id': self.default_user_id})
                
                progress = []
                for row in result:
                    progress.append({
                        'date': row[0],
                        'avg_score': row[1] or 0,
                        'session_count': row[2] or 0
                    })
                
                return progress
                
        except SQLAlchemyError as e:
            print(f"Database error getting progress: {e}")
            return []
    
    def clear_all_data(self):
        """Clear all session data for the user"""
        try:
            with self.engine.connect() as conn:
                conn.execute(text("""
                    DELETE FROM practice_sessions WHERE user_id = :user_id
                """), {'user_id': self.default_user_id})
                conn.commit()
                
        except SQLAlchemyError as e:
            print(f"Database error clearing data: {e}")
    
    def export_to_csv(self, output_file: str = None) -> str:
        """Export sessions data to CSV format"""
        try:
            sessions = self.load_sessions(limit=1000)  # Get more sessions for export
            
            if not sessions:
                return "No data to export"
            
            # Convert to DataFrame
            df = pd.DataFrame(sessions)
            
            # Flatten scores dictionary
            scores_df = pd.json_normalize(df['scores'])
            df = pd.concat([df.drop('scores', axis=1), scores_df], axis=1)
            
            # Generate filename if not provided
            if not output_file:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_file = f"voice_coach_sessions_{timestamp}.csv"
            
            df.to_csv(output_file, index=False)
            return output_file
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return None
    
    def generate_session_id(self) -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    def get_data_size(self) -> float:
        """Get approximate database size in MB"""
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('practice_sessions')) as sessions_size,
                        pg_size_pretty(pg_total_relation_size('session_scores')) as scores_size,
                        pg_size_pretty(pg_total_relation_size('session_feedback')) as feedback_size,
                        pg_size_pretty(pg_total_relation_size('session_metrics')) as metrics_size
                """))
                
                row = result.fetchone()
                if row:
                    # This is a simplified calculation - in reality, you'd parse the sizes
                    return 0.1  # Placeholder MB value
                return 0.0
                
        except SQLAlchemyError as e:
            print(f"Database error getting size: {e}")
            return 0.0