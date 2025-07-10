import pyaudio
import numpy as np
import threading
import time
import queue
from collections import deque

class AudioProcessor:
    def __init__(self, sample_rate=44100, chunk_size=1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.audio = None
        self.stream = None
        self.recording = False
        self.audio_data = []
        self.current_metrics = {}
        self.lock = threading.Lock()
        self.audio_queue = queue.Queue()
        self.thread = None
        
        # Initialize PyAudio
        self.initialize_audio()
        
        # Metrics tracking
        self.reset_metrics()
    
    def initialize_audio(self):
        """Initialize PyAudio"""
        try:
            self.audio = pyaudio.PyAudio()
            print("Audio system initialized successfully")
        except Exception as e:
            print(f"Error initializing audio: {e}")
            # Don't fail completely if audio init fails in server environment
            self.audio = None
    
    def reset_metrics(self):
        """Reset all metrics for a new session"""
        with self.lock:
            self.current_metrics = {
                'speaking_time': 0.0,
                'silence_time': 0.0,
                'silence_ratio': 0.0,
                'average_pitch': 0.0,
                'pitch_variation': 0.0,
                'volume_level': 0.0,
                'filler_count': 0,
                'speech_rate': 0.0,
                'energy_level': 0.0
            }
            self.audio_data = []
    
    def start_recording(self):
        """Start audio recording"""
        try:
            if not self.audio:
                self.initialize_audio()
            
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self.audio_callback
            )
            
            self.recording = True
            self.reset_metrics()
            
            # Start processing thread
            self.thread = threading.Thread(target=self.process_audio)
            self.thread.start()
            
            self.stream.start_stream()
            return True
            
        except Exception as e:
            print(f"Error starting audio recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording and return session data"""
        self.recording = False
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.thread:
            self.thread.join()
        
        return self.audio_data.copy()
    
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Audio input callback"""
        if self.recording:
            audio_chunk = np.frombuffer(in_data, dtype=np.float32)
            self.audio_queue.put((time.time(), audio_chunk))
        
        return (in_data, pyaudio.paContinue)
    
    def process_audio(self):
        """Main audio processing loop"""
        audio_buffer = deque(maxlen=self.sample_rate * 5)  # 5 second buffer
        
        while self.recording:
            try:
                # Get audio chunk from queue
                timestamp, audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Add to buffer
                audio_buffer.extend(audio_chunk)
                
                # Store raw audio data
                self.audio_data.append({
                    'timestamp': timestamp,
                    'audio_chunk': audio_chunk
                })
                
                # Process audio chunk
                if len(audio_buffer) >= self.chunk_size:
                    chunk_data = np.array(list(audio_buffer)[-self.chunk_size:])
                    metrics = self.analyze_audio_chunk(chunk_data)
                    self.update_metrics(metrics)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def analyze_audio_chunk(self, audio_chunk):
        """Analyze a single audio chunk"""
        chunk_metrics = {
            'volume': 0.0,
            'pitch': 0.0,
            'energy': 0.0,
            'is_speech': False,
            'is_silence': False
        }
        
        try:
            # Calculate volume (RMS)
            chunk_metrics['volume'] = np.sqrt(np.mean(audio_chunk**2))
            
            # Detect speech vs silence
            volume_threshold = 0.01  # Adjust based on microphone sensitivity
            chunk_metrics['is_speech'] = chunk_metrics['volume'] > volume_threshold
            chunk_metrics['is_silence'] = not chunk_metrics['is_speech']
            
            # Calculate energy
            chunk_metrics['energy'] = np.sum(audio_chunk**2)
            
            # Estimate pitch using autocorrelation
            if chunk_metrics['is_speech']:
                chunk_metrics['pitch'] = self.estimate_pitch(audio_chunk)
            
        except Exception as e:
            print(f"Error analyzing audio chunk: {e}")
        
        return chunk_metrics
    
    def estimate_pitch(self, audio_chunk):
        """Estimate pitch of audio chunk using autocorrelation"""
        try:
            # Apply window to reduce artifacts
            windowed = audio_chunk * np.hanning(len(audio_chunk))
            
            # Calculate autocorrelation
            autocorr = np.correlate(windowed, windowed, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            
            # Find the peak (excluding the first peak at lag=0)
            min_lag = int(self.sample_rate / 800)  # Minimum 800 Hz
            max_lag = int(self.sample_rate / 80)   # Maximum 80 Hz
            
            if max_lag < len(autocorr):
                peak_idx = np.argmax(autocorr[min_lag:max_lag]) + min_lag
                
                if peak_idx > 0:
                    pitch = self.sample_rate / peak_idx
                    
                    # Filter out unrealistic pitch values
                    if 80 <= pitch <= 800:
                        return pitch
            
        except Exception as e:
            print(f"Error estimating pitch: {e}")
        
        return 0.0
    
    def update_metrics(self, chunk_metrics):
        """Update current metrics based on chunk data"""
        with self.lock:
            # Update speaking/silence time
            chunk_duration = self.chunk_size / self.sample_rate
            
            if chunk_metrics['is_speech']:
                self.current_metrics['speaking_time'] += chunk_duration
            else:
                self.current_metrics['silence_time'] += chunk_duration
            
            # Update silence ratio
            total_time = self.current_metrics['speaking_time'] + self.current_metrics['silence_time']
            if total_time > 0:
                self.current_metrics['silence_ratio'] = (
                    self.current_metrics['silence_time'] / total_time * 100
                )
            
            # Update average volume
            if chunk_metrics['volume'] > 0:
                current_volume = self.current_metrics['volume_level']
                self.current_metrics['volume_level'] = (
                    current_volume * 0.9 + chunk_metrics['volume'] * 0.1
                )
            
            # Update pitch metrics
            if chunk_metrics['pitch'] > 0:
                current_pitch = self.current_metrics['average_pitch']
                if current_pitch == 0:
                    self.current_metrics['average_pitch'] = chunk_metrics['pitch']
                else:
                    self.current_metrics['average_pitch'] = (
                        current_pitch * 0.9 + chunk_metrics['pitch'] * 0.1
                    )
                
                # Update pitch variation
                pitch_diff = abs(chunk_metrics['pitch'] - current_pitch)
                self.current_metrics['pitch_variation'] = (
                    self.current_metrics['pitch_variation'] * 0.9 + pitch_diff * 0.1
                )
            
            # Update energy level
            if chunk_metrics['energy'] > 0:
                current_energy = self.current_metrics['energy_level']
                self.current_metrics['energy_level'] = (
                    current_energy * 0.9 + chunk_metrics['energy'] * 0.1
                )
    
    def get_current_metrics(self):
        """Get current metrics (thread-safe)"""
        with self.lock:
            return self.current_metrics.copy()
    
    def detect_fillers(self, audio_data):
        """Detect filler words like 'um', 'uh', 'like', etc."""
        # This is a simplified approach - in practice, you'd use speech recognition
        # or machine learning models to detect specific filler words
        filler_count = 0
        
        try:
            # Concatenate all audio chunks
            full_audio = np.concatenate([chunk['audio_chunk'] for chunk in audio_data])
            
            # Simple pattern detection based on audio characteristics
            # Fillers typically have specific frequency patterns and durations
            
            # Split into segments
            segment_length = int(self.sample_rate * 0.5)  # 0.5 second segments
            segments = [full_audio[i:i+segment_length] 
                       for i in range(0, len(full_audio), segment_length)]
            
            for segment in segments:
                if len(segment) < segment_length:
                    continue
                
                # Check for filler patterns
                if self.is_likely_filler(segment):
                    filler_count += 1
            
        except Exception as e:
            print(f"Error detecting fillers: {e}")
        
        return filler_count
    
    def is_likely_filler(self, audio_segment):
        """Check if audio segment is likely a filler word"""
        try:
            # Simple heuristics for filler detection
            # Fillers typically have:
            # 1. Lower energy than regular speech
            # 2. Specific frequency characteristics
            # 3. Shorter duration
            
            energy = np.mean(audio_segment**2)
            volume = np.sqrt(np.mean(audio_segment**2))
            
            # Low energy and volume might indicate hesitation/filler
            return energy < 0.01 and volume < 0.1
            
        except Exception as e:
            print(f"Error checking filler: {e}")
            return False
    
    def calculate_speech_rate(self, audio_data, duration):
        """Calculate speech rate (words per minute estimate)"""
        try:
            # Estimate speech rate based on audio characteristics
            # This is a simplified approach
            
            speaking_time = self.current_metrics['speaking_time']
            if speaking_time == 0:
                return 0
            
            # Count speech segments (rough word estimation)
            segment_count = 0
            for chunk in audio_data:
                audio_chunk = chunk['audio_chunk']
                volume = np.sqrt(np.mean(audio_chunk**2))
                
                if volume > 0.01:  # Speech threshold
                    segment_count += 1
            
            # Estimate words per minute
            # Assuming average of 3-4 chunks per word
            estimated_words = segment_count / 3.5
            words_per_minute = (estimated_words / speaking_time) * 60
            
            return min(words_per_minute, 300)  # Cap at 300 WPM
            
        except Exception as e:
            print(f"Error calculating speech rate: {e}")
            return 0
    
    def get_session_summary(self):
        """Get summary of the entire session"""
        try:
            total_duration = self.current_metrics['speaking_time'] + self.current_metrics['silence_time']
            
            summary = {
                'total_duration': total_duration,
                'speaking_time': self.current_metrics['speaking_time'],
                'silence_time': self.current_metrics['silence_time'],
                'silence_ratio': self.current_metrics['silence_ratio'],
                'average_pitch': self.current_metrics['average_pitch'],
                'pitch_variation': self.current_metrics['pitch_variation'],
                'volume_level': self.current_metrics['volume_level'],
                'energy_level': self.current_metrics['energy_level'],
                'filler_count': self.detect_fillers(self.audio_data),
                'speech_rate': self.calculate_speech_rate(self.audio_data, total_duration)
            }
            
            return summary
            
        except Exception as e:
            print(f"Error getting session summary: {e}")
            return {}
    
    def cleanup(self):
        """Clean up audio resources"""
        if self.audio:
            self.audio.terminate()
