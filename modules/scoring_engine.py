import numpy as np
from typing import Dict, List, Any

class ScoringEngine:
    def __init__(self):
        # Scoring weights and thresholds
        self.weights = {
            'confidence': {
                'face_detection': 0.3,
                'eye_contact': 0.4,
                'head_stability': 0.3
            },
            'clarity': {
                'speech_rate': 0.4,
                'volume_consistency': 0.3,
                'filler_ratio': 0.3
            },
            'presence': {
                'energy_level': 0.4,
                'smile_frequency': 0.3,
                'posture': 0.3
            },
            'energy': {
                'voice_energy': 0.5,
                'pitch_variation': 0.3,
                'engagement': 0.2
            }
        }
        
        # Ideal ranges for different metrics
        self.ideal_ranges = {
            'speech_rate': (120, 180),  # words per minute
            'eye_contact': (60, 80),    # percentage
            'smile_frequency': (10, 30), # percentage
            'silence_ratio': (10, 25),  # percentage
            'pitch_variation': (20, 50), # Hz
            'volume_level': (0.1, 0.8)  # normalized
        }
    
    def calculate_scores(self, video_data: List[Dict], audio_data: List[Dict], duration: float) -> Dict[str, Any]:
        """Calculate comprehensive scores for a practice session"""
        
        # Extract metrics from video and audio data
        video_metrics = self.extract_video_metrics(video_data)
        audio_metrics = self.extract_audio_metrics(audio_data, duration)
        
        # Calculate individual scores
        confidence_score = self.calculate_confidence_score(video_metrics, audio_metrics)
        clarity_score = self.calculate_clarity_score(audio_metrics)
        presence_score = self.calculate_presence_score(video_metrics, audio_metrics)
        energy_score = self.calculate_energy_score(audio_metrics)
        
        # Generate feedback
        feedback = self.generate_feedback(video_metrics, audio_metrics, {
            'confidence': confidence_score,
            'clarity': clarity_score,
            'presence': presence_score,
            'energy': energy_score
        })
        
        return {
            'confidence': confidence_score,
            'clarity': clarity_score,
            'presence': presence_score,
            'energy': energy_score,
            'feedback': feedback
        }
    
    def extract_video_metrics(self, video_data: List[Dict]) -> Dict[str, float]:
        """Extract key metrics from video data"""
        if not video_data:
            return {
                'face_detection_rate': 0,
                'eye_contact_percentage': 0,
                'smile_percentage': 0,
                'head_stability': 0,
                'average_confidence': 0
            }
        
        frames_with_face = sum(1 for frame in video_data if frame['frame_data']['face_detected'])
        frames_with_eye_contact = sum(1 for frame in video_data if frame['frame_data']['eye_contact'])
        frames_with_smile = sum(1 for frame in video_data if frame['frame_data']['smile_detected'])
        
        total_frames = len(video_data)
        
        # Calculate head stability based on position changes
        head_stability = self.calculate_head_stability(video_data)
        
        # Calculate average confidence
        confidences = [frame['frame_data']['confidence'] for frame in video_data 
                      if frame['frame_data']['face_detected']]
        average_confidence = np.mean(confidences) if confidences else 0
        
        return {
            'face_detection_rate': (frames_with_face / total_frames) * 100,
            'eye_contact_percentage': (frames_with_eye_contact / total_frames) * 100,
            'smile_percentage': (frames_with_smile / total_frames) * 100,
            'head_stability': head_stability,
            'average_confidence': average_confidence
        }
    
    def extract_audio_metrics(self, audio_data: List[Dict], duration: float) -> Dict[str, float]:
        """Extract key metrics from audio data"""
        if not audio_data or duration == 0:
            return {
                'speech_rate': 0,
                'volume_consistency': 0,
                'pitch_variation': 0,
                'energy_level': 0,
                'silence_ratio': 0,
                'filler_count': 0
            }
        
        # Calculate speech-related metrics
        volumes = []
        pitches = []
        energies = []
        speech_segments = 0
        
        for chunk in audio_data:
            audio_chunk = chunk['audio_chunk']
            volume = np.sqrt(np.mean(audio_chunk**2))
            
            if volume > 0.01:  # Speech threshold
                volumes.append(volume)
                speech_segments += 1
                energies.append(np.sum(audio_chunk**2))
                
                # Simple pitch estimation
                pitch = self.estimate_pitch_simple(audio_chunk)
                if pitch > 0:
                    pitches.append(pitch)
        
        # Calculate metrics
        speech_rate = self.estimate_speech_rate(speech_segments, duration)
        volume_consistency = self.calculate_volume_consistency(volumes)
        pitch_variation = np.std(pitches) if pitches else 0
        energy_level = np.mean(energies) if energies else 0
        
        # Estimate silence ratio
        speaking_time = speech_segments * (1024 / 44100)  # Approximate
        silence_ratio = max(0, (duration - speaking_time) / duration * 100)
        
        # Simple filler detection
        filler_count = self.estimate_filler_count(audio_data)
        
        return {
            'speech_rate': speech_rate,
            'volume_consistency': volume_consistency,
            'pitch_variation': pitch_variation,
            'energy_level': energy_level,
            'silence_ratio': silence_ratio,
            'filler_count': filler_count
        }
    
    def calculate_confidence_score(self, video_metrics: Dict[str, float], audio_metrics: Dict[str, float]) -> float:
        """Calculate confidence score based on video and audio metrics"""
        
        # Face detection component
        face_score = min(video_metrics['face_detection_rate'] / 90, 1.0)  # Normalize to 90%
        
        # Eye contact component
        eye_contact_ideal = self.ideal_ranges['eye_contact']
        eye_contact_score = self.score_in_range(
            video_metrics['eye_contact_percentage'], 
            eye_contact_ideal[0], 
            eye_contact_ideal[1]
        )
        
        # Head stability component
        head_stability_score = video_metrics['head_stability']
        
        # Combine components
        confidence_score = (
            face_score * self.weights['confidence']['face_detection'] +
            eye_contact_score * self.weights['confidence']['eye_contact'] +
            head_stability_score * self.weights['confidence']['head_stability']
        )
        
        return min(confidence_score * 10, 10)  # Scale to 0-10
    
    def calculate_clarity_score(self, audio_metrics: Dict[str, float]) -> float:
        """Calculate clarity score based on audio metrics"""
        
        # Speech rate component
        speech_rate_ideal = self.ideal_ranges['speech_rate']
        speech_rate_score = self.score_in_range(
            audio_metrics['speech_rate'],
            speech_rate_ideal[0],
            speech_rate_ideal[1]
        )
        
        # Volume consistency component
        volume_score = audio_metrics['volume_consistency']
        
        # Filler ratio component (inverse - fewer fillers is better)
        filler_score = max(0, 1 - (audio_metrics['filler_count'] / 20))  # Normalize to 20 fillers
        
        # Combine components
        clarity_score = (
            speech_rate_score * self.weights['clarity']['speech_rate'] +
            volume_score * self.weights['clarity']['volume_consistency'] +
            filler_score * self.weights['clarity']['filler_ratio']
        )
        
        return min(clarity_score * 10, 10)  # Scale to 0-10
    
    def calculate_presence_score(self, video_metrics: Dict[str, float], audio_metrics: Dict[str, float]) -> float:
        """Calculate presence score based on video and audio metrics"""
        
        # Energy level component
        energy_score = min(audio_metrics['energy_level'] * 1000, 1.0)  # Normalize
        
        # Smile frequency component
        smile_ideal = self.ideal_ranges['smile_frequency']
        smile_score = self.score_in_range(
            video_metrics['smile_percentage'],
            smile_ideal[0],
            smile_ideal[1]
        )
        
        # Posture component (based on face confidence and stability)
        posture_score = (video_metrics['average_confidence'] + video_metrics['head_stability']) / 2
        
        # Combine components
        presence_score = (
            energy_score * self.weights['presence']['energy_level'] +
            smile_score * self.weights['presence']['smile_frequency'] +
            posture_score * self.weights['presence']['posture']
        )
        
        return min(presence_score * 10, 10)  # Scale to 0-10
    
    def calculate_energy_score(self, audio_metrics: Dict[str, float]) -> float:
        """Calculate energy score based on audio metrics"""
        
        # Voice energy component
        voice_energy_score = min(audio_metrics['energy_level'] * 1000, 1.0)  # Normalize
        
        # Pitch variation component
        pitch_ideal = self.ideal_ranges['pitch_variation']
        pitch_score = self.score_in_range(
            audio_metrics['pitch_variation'],
            pitch_ideal[0],
            pitch_ideal[1]
        )
        
        # Engagement component (based on speaking time)
        silence_ideal = self.ideal_ranges['silence_ratio']
        engagement_score = 1 - self.score_in_range(
            audio_metrics['silence_ratio'],
            silence_ideal[0],
            silence_ideal[1]
        )
        
        # Combine components
        energy_score = (
            voice_energy_score * self.weights['energy']['voice_energy'] +
            pitch_score * self.weights['energy']['pitch_variation'] +
            engagement_score * self.weights['energy']['engagement']
        )
        
        return min(energy_score * 10, 10)  # Scale to 0-10
    
    def score_in_range(self, value: float, min_ideal: float, max_ideal: float) -> float:
        """Score a value based on how close it is to an ideal range"""
        if min_ideal <= value <= max_ideal:
            return 1.0
        elif value < min_ideal:
            # Score decreases as value gets further from minimum
            return max(0, 1 - (min_ideal - value) / min_ideal)
        else:
            # Score decreases as value gets further from maximum
            return max(0, 1 - (value - max_ideal) / max_ideal)
    
    def calculate_head_stability(self, video_data: List[Dict]) -> float:
        """Calculate head stability based on position changes"""
        if not video_data:
            return 0
        
        stability_scores = []
        
        for i in range(1, len(video_data)):
            current_frame = video_data[i]['frame_data']
            previous_frame = video_data[i-1]['frame_data']
            
            if (current_frame['head_position'] and 
                previous_frame['head_position'] and
                current_frame['face_detected'] and 
                previous_frame['face_detected']):
                
                # Calculate position change
                curr_pos = current_frame['head_position']['center']
                prev_pos = previous_frame['head_position']['center']
                
                distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                
                # Normalize distance (smaller movement = higher stability)
                stability = max(0, 1 - (distance / 50))  # Normalize to 50 pixels
                stability_scores.append(stability)
        
        return np.mean(stability_scores) if stability_scores else 0
    
    def estimate_speech_rate(self, speech_segments: int, duration: float) -> float:
        """Estimate speech rate in words per minute"""
        if duration == 0:
            return 0
        
        # Rough estimation: 3-4 audio chunks per word
        estimated_words = speech_segments / 3.5
        words_per_minute = (estimated_words / duration) * 60
        
        return min(words_per_minute, 300)  # Cap at 300 WPM
    
    def calculate_volume_consistency(self, volumes: List[float]) -> float:
        """Calculate volume consistency (lower standard deviation = higher consistency)"""
        if not volumes:
            return 0
        
        if len(volumes) < 2:
            return 1.0
        
        std_dev = np.std(volumes)
        mean_vol = np.mean(volumes)
        
        if mean_vol == 0:
            return 0
        
        # Coefficient of variation (lower = more consistent)
        cv = std_dev / mean_vol
        
        # Convert to score (0-1, where 1 is most consistent)
        return max(0, 1 - cv)
    
    def estimate_pitch_simple(self, audio_chunk: np.ndarray) -> float:
        """Simple pitch estimation using zero crossing rate"""
        try:
            # Zero crossing rate method (very basic)
            zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_chunk))))
            sample_rate = 44100
            
            if zero_crossings > 0:
                # Rough pitch estimation
                pitch = (zero_crossings * sample_rate) / (2 * len(audio_chunk))
                
                # Filter reasonable pitch range
                if 80 <= pitch <= 800:
                    return pitch
            
        except Exception as e:
            print(f"Error estimating pitch: {e}")
        
        return 0
    
    def estimate_filler_count(self, audio_data: List[Dict]) -> int:
        """Estimate number of filler words/sounds"""
        filler_count = 0
        
        try:
            # Look for patterns that might indicate fillers
            for i in range(len(audio_data) - 1):
                current_chunk = audio_data[i]['audio_chunk']
                next_chunk = audio_data[i + 1]['audio_chunk']
                
                current_volume = np.sqrt(np.mean(current_chunk**2))
                next_volume = np.sqrt(np.mean(next_chunk**2))
                
                # Look for brief, low-volume sounds followed by pauses
                if (0.005 < current_volume < 0.02 and  # Low volume
                    next_volume < 0.005):  # Followed by silence
                    filler_count += 1
        
        except Exception as e:
            print(f"Error estimating fillers: {e}")
        
        return filler_count
    
    def generate_feedback(self, video_metrics: Dict[str, float], audio_metrics: Dict[str, float], 
                         scores: Dict[str, float]) -> List[str]:
        """Generate specific feedback based on performance"""
        feedback = []
        
        # Confidence feedback
        if scores['confidence'] < 6:
            if video_metrics['eye_contact_percentage'] < 50:
                feedback.append("Try to maintain more eye contact with the camera")
            if video_metrics['face_detection_rate'] < 80:
                feedback.append("Make sure you're clearly visible in the frame")
            if video_metrics['head_stability'] < 0.7:
                feedback.append("Try to keep your head more stable while speaking")
        
        # Clarity feedback
        if scores['clarity'] < 6:
            if audio_metrics['speech_rate'] < 100:
                feedback.append("Try to speak a bit faster - aim for 120-150 words per minute")
            elif audio_metrics['speech_rate'] > 200:
                feedback.append("Try to slow down your speech for better clarity")
            if audio_metrics['filler_count'] > 5:
                feedback.append("Reduce filler words like 'um', 'uh', and 'like'")
            if audio_metrics['volume_consistency'] < 0.6:
                feedback.append("Try to maintain consistent volume throughout")
        
        # Presence feedback
        if scores['presence'] < 6:
            if video_metrics['smile_percentage'] < 10:
                feedback.append("Smile more to appear more engaging and confident")
            if audio_metrics['energy_level'] < 0.01:
                feedback.append("Put more energy and enthusiasm into your voice")
        
        # Energy feedback
        if scores['energy'] < 6:
            if audio_metrics['pitch_variation'] < 20:
                feedback.append("Vary your pitch more to sound more dynamic")
            if audio_metrics['silence_ratio'] > 30:
                feedback.append("Reduce long pauses - keep your speech flowing")
        
        # Positive reinforcement
        if scores['confidence'] >= 8:
            feedback.append("Excellent confidence and presence!")
        if scores['clarity'] >= 8:
            feedback.append("Your speech clarity is very good!")
        if scores['energy'] >= 8:
            feedback.append("Great energy and enthusiasm!")
        
        return feedback if feedback else ["Good job! Keep practicing to improve further."]
