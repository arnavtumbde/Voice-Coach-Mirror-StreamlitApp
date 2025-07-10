import cv2
import numpy as np
import time
from threading import Thread, Lock
import os
import urllib.request

class VideoProcessor:
    def __init__(self):
        self.cap = None
        self.detector = None
        self.predictor = None
        self.recording = False
        self.current_metrics = {}
        self.session_data = []
        self.lock = Lock()
        self.thread = None
        
        # Initialize face detection
        self.initialize_face_detection()
        
        # Metrics tracking
        self.reset_metrics()
    
    def initialize_face_detection(self):
        """Initialize face detection using OpenCV"""
        try:
            # Initialize OpenCV face detector (Haar cascade)
            face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(face_cascade_path)
            
            # Initialize eye detector for eye contact detection
            eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
            self.eye_detector = cv2.CascadeClassifier(eye_cascade_path)
            
            # Initialize smile detector
            smile_cascade_path = cv2.data.haarcascades + 'haarcascade_smile.xml'
            self.smile_detector = cv2.CascadeClassifier(smile_cascade_path)
            
            print("OpenCV face detection initialized successfully")
                
        except Exception as e:
            print(f"Error initializing face detection: {e}")
    
    def download_face_predictor(self, predictor_path):
        """Download the face landmarks predictor model - Not needed for OpenCV version"""
        # This method is no longer needed since we're using OpenCV's built-in cascades
        pass
    
    def reset_metrics(self):
        """Reset all metrics for a new session"""
        with self.lock:
            self.current_metrics = {
                'eye_contact': 0.0,
                'smile_time': 0.0,
                'head_movement': 0.0,
                'face_detected': 0.0,
                'total_frames': 0,
                'frames_with_face': 0,
                'frames_with_eye_contact': 0,
                'frames_with_smile': 0
            }
            self.session_data = []
    
    def start_capture(self):
        """Start video capture and processing"""
        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("Cannot open camera")
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.recording = True
            self.reset_metrics()
            
            # Start processing thread
            self.thread = Thread(target=self.process_video)
            self.thread.start()
            
            return True
            
        except Exception as e:
            print(f"Error starting video capture: {e}")
            return False
    
    def stop_capture(self):
        """Stop video capture and return session data"""
        self.recording = False
        
        if self.thread:
            self.thread.join()
        
        if self.cap:
            self.cap.release()
        
        return self.session_data.copy()
    
    def process_video(self):
        """Main video processing loop"""
        while self.recording and self.cap:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame
            frame_data = self.analyze_frame(frame)
            
            # Update metrics
            self.update_metrics(frame_data)
            
            # Store frame data
            self.session_data.append({
                'timestamp': time.time(),
                'frame_data': frame_data
            })
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.033)  # ~30 FPS
    
    def analyze_frame(self, frame):
        """Analyze a single frame for facial features"""
        frame_data = {
            'face_detected': False,
            'eye_contact': False,
            'smile_detected': False,
            'head_position': None,
            'confidence': 0.0
        }
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces using OpenCV Haar cascades
            faces = self.detector.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0:
                frame_data['face_detected'] = True
                face = faces[0]  # Use first detected face (x, y, w, h)
                x, y, w, h = face
                
                # Extract face region
                face_roi = gray[y:y+h, x:x+w]
                
                # Detect eyes within face region
                eyes = self.eye_detector.detectMultiScale(face_roi)
                frame_data['eye_contact'] = len(eyes) >= 2  # Both eyes detected
                
                # Detect smile within face region
                smiles = self.smile_detector.detectMultiScale(face_roi, 1.8, 20)
                frame_data['smile_detected'] = len(smiles) > 0
                
                # Calculate head position (center of face)
                center_x = x + w // 2
                center_y = y + h // 2
                frame_data['head_position'] = {
                    'center': (center_x, center_y),
                    'face_rect': (x, y, w, h)
                }
                
                # Calculate confidence based on face size and position
                face_area = w * h
                frame_area = frame.shape[0] * frame.shape[1]
                face_ratio = face_area / frame_area
                
                # Confidence increases with larger, more centered faces
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2
                
                distance_from_center = np.sqrt((center_x - frame_center_x)**2 + (center_y - frame_center_y)**2)
                max_distance = np.sqrt(frame_center_x**2 + frame_center_y**2)
                center_score = 1 - (distance_from_center / max_distance)
                
                frame_data['confidence'] = min(face_ratio * 10, 1.0) * center_score
                
        except Exception as e:
            print(f"Error analyzing frame: {e}")
        
        return frame_data
    
    def detect_eye_contact(self, face_roi):
        """Detect if person is making eye contact with camera"""
        try:
            # Detect eyes within the face region
            eyes = self.eye_detector.detectMultiScale(face_roi)
            
            # Eye contact is detected if both eyes are detected
            # This is a simplified approach - more sophisticated methods
            # would analyze gaze direction
            return len(eyes) >= 2
            
        except Exception as e:
            print(f"Error detecting eye contact: {e}")
            return False
    
    def eye_aspect_ratio(self, eye_points):
        """Calculate eye aspect ratio - Not needed for OpenCV version"""
        # This method is no longer needed since we're using OpenCV's built-in detection
        pass
    
    def detect_smile(self, face_roi):
        """Detect if person is smiling"""
        try:
            # Detect smile within the face region
            smiles = self.smile_detector.detectMultiScale(face_roi, 1.8, 20)
            
            # Smile is detected if at least one smile is found
            return len(smiles) > 0
            
        except Exception as e:
            print(f"Error detecting smile: {e}")
            return False
    
    def get_head_position(self, face_rect):
        """Get head position/orientation"""
        try:
            x, y, w, h = face_rect
            
            # Calculate center of face
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Simple head position based on face center
            return {
                'center': (center_x, center_y),
                'face_rect': face_rect,
                'tilt': 0  # Simplified - no tilt detection without landmarks
            }
            
        except Exception as e:
            print(f"Error getting head position: {e}")
            return None
    
    def update_metrics(self, frame_data):
        """Update current metrics based on frame data"""
        with self.lock:
            self.current_metrics['total_frames'] += 1
            
            if frame_data['face_detected']:
                self.current_metrics['frames_with_face'] += 1
                
                if frame_data['eye_contact']:
                    self.current_metrics['frames_with_eye_contact'] += 1
                
                if frame_data['smile_detected']:
                    self.current_metrics['frames_with_smile'] += 1
            
            # Calculate percentages
            if self.current_metrics['total_frames'] > 0:
                self.current_metrics['face_detected'] = (
                    self.current_metrics['frames_with_face'] / self.current_metrics['total_frames'] * 100
                )
                
                self.current_metrics['eye_contact'] = (
                    self.current_metrics['frames_with_eye_contact'] / self.current_metrics['total_frames'] * 100
                )
                
                self.current_metrics['smile_time'] = (
                    self.current_metrics['frames_with_smile'] / self.current_metrics['total_frames'] * 100
                )
    
    def get_current_metrics(self):
        """Get current metrics (thread-safe)"""
        with self.lock:
            return self.current_metrics.copy()
    
    def get_session_summary(self):
        """Get summary of the entire session"""
        summary = {
            'total_frames': len(self.session_data),
            'face_detection_rate': 0,
            'eye_contact_percentage': 0,
            'smile_percentage': 0,
            'average_confidence': 0
        }
        
        if not self.session_data:
            return summary
        
        frames_with_face = sum(1 for data in self.session_data if data['frame_data']['face_detected'])
        frames_with_eye_contact = sum(1 for data in self.session_data if data['frame_data']['eye_contact'])
        frames_with_smile = sum(1 for data in self.session_data if data['frame_data']['smile_detected'])
        
        total_frames = len(self.session_data)
        
        summary['face_detection_rate'] = frames_with_face / total_frames * 100
        summary['eye_contact_percentage'] = frames_with_eye_contact / total_frames * 100
        summary['smile_percentage'] = frames_with_smile / total_frames * 100
        
        # Calculate average confidence
        confidences = [data['frame_data']['confidence'] for data in self.session_data 
                      if data['frame_data']['face_detected']]
        if confidences:
            summary['average_confidence'] = np.mean(confidences)
        
        return summary
