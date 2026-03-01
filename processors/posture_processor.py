import time
import math
import numpy as np
from collections import deque
from game.game_state import GameState
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

class PostureProcessor:
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        
        # Inactivity tracking
        self.movement_history = deque(maxlen=30) # approx 5 seconds at 6 fps
        self.last_keypoints = None
        self.last_movement_time = time.time()
        
        # Thresholds
        self.fwd_head_thresh = 25.0 # degrees
        self.shoulder_imbalance_thresh = 25.0 # degrees, increased for webcam tilt tolerance
        self.slouch_thresh = 15.0 # degrees
        
        self.base_hip_cx = None
        
        # Mediapipe for Sleep Detection (Task API)
        base_options = mp_python.BaseOptions(model_asset_path='face_landmarker.task')
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        
    def _compute_angle(self, p1, p2, p3=None):
        # Angle between line p1-p2 and vertical (if p3 is None)
        # or angle between p1-p2 and p2-p3
        if p3 is None:
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            return math.degrees(math.atan2(abs(dx), abs(dy)))
        else:
            # Angle between vectors
            v1 = (p1[0]-p2[0], p1[1]-p2[1])
            v2 = (p3[0]-p2[0], p3[1]-p2[1])
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
            mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
            if mag1*mag2 == 0: return 0
            val = max(-1.0, min(1.0, dot / (mag1*mag2)))
            return math.degrees(math.acos(val))
            
    def _check_inactivity(self, keypoints, current_time):
        if not keypoints:
            return self.game_state.inactive_seconds

        # Compute variance using only Torso and Head points (indices 0, 5, 6, 11, 12)
        # to ignore waving hands/typing.
        torso_indices = {0, 5, 6, 11, 12}
        pts = np.array([kp[:2] for i, kp in enumerate(keypoints) if i in torso_indices and kp[2] > 0.3])
        if len(pts) == 0:
            return self.game_state.inactive_seconds
            
        center = np.mean(pts, axis=0)
        self.movement_history.append(center)
        
        if len(self.movement_history) == self.movement_history.maxlen:
            history_pts = np.array(self.movement_history)
            variance = np.var(history_pts, axis=0).sum()
            
            # If torso variance is very high, user shifted physically; reset inactivity
            if variance > 200.0:
                self.last_movement_time = current_time
                self.movement_history.clear()
        
        inactive_secs = current_time - self.last_movement_time
        return inactive_secs

    def process(self, keypoints, frame_array=None):
        """
        Takes YOLO keypoints list, e.g. [[x,y,conf], ...] 17 keypoints
        Returns a posture dictionary.
        """
        current_time = time.time()
        
        # 1. Check inactivity
        inactive_seconds = self._check_inactivity(keypoints, current_time)
        
        if not keypoints or len(keypoints) < 13:
            # Update state with no pose
            self.game_state.update("unknown", 0.0, inactive_seconds)
            return {
                "posture_state": "unknown",
                "severity": 0.0,
                "keypoints": keypoints
            }
            
        def pt(idx):
             if idx < len(keypoints) and keypoints[idx][2] > 0.5:
                 return (keypoints[idx][0], keypoints[idx][1])
             return None
             
        nose = pt(0)
        l_shoulder = pt(5)
        r_shoulder = pt(6)
        
        state = "good"
        severity = 0.0

        # EAR (Eye Aspect Ratio) Calculation via Mediapipe Task API
        # Only triggers if a face is actually detected (prevents false positives on empty frames)
        if frame_array is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_array)
            try:
                results = self.face_landmarker.detect(mp_image)
            except Exception as e:
                results = None
                
            # Only check for sleeping if a face was actually detected
            if results and results.face_landmarks:
                landmarks = results.face_landmarks[0]
                h, w, _ = frame_array.shape
                
                def get_pt(idx):
                    return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
                    
                def calc_ear(eye_indices):
                    # Eye indices: [p1_left, p2_top1, p3_top2, p4_right, p5_bottom2, p6_bottom1]
                    p1, p2, p3, p4, p5, p6 = [get_pt(i) for i in eye_indices]
                    v1 = np.linalg.norm(p2 - p6)
                    v2 = np.linalg.norm(p3 - p5)
                    h_dist = np.linalg.norm(p1 - p4)
                    return (v1 + v2) / (2.0 * h_dist) if h_dist > 0 else 0.5  # default OPEN if no eye width
                
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                
                left_ear = calc_ear(left_eye_indices)
                right_ear = calc_ear(right_eye_indices)
                avg_ear = (left_ear + right_ear) / 2.0
                
                # If average EAR drops below 0.15, eyes are genuinely closed (not just a blink)
                if avg_ear < 0.15:
                    state = "eyes_closed"
                    severity = 1.0
        
        # If not sleeping, check posture
        if state != "eyes_closed":
            # We now rely only on upper body (shoulders and nose) because webcams rarely capture hips.
            if l_shoulder and r_shoulder and nose:
                shoulder_cx = (l_shoulder[0] + r_shoulder[0]) / 2
                shoulder_cy = (l_shoulder[1] + r_shoulder[1]) / 2
                shoulder_width = max(1.0, abs(l_shoulder[0] - r_shoulder[0]))
                
                # Lean Drift (Horizontal drift of nose relative to shoulders over time)
                if self.base_hip_cx is None:
                    # repurpose base_hip_cx to base_nose_cx
                    self.base_hip_cx = nose[0]
                else:
                    drift = abs(nose[0] - self.base_hip_cx)
                
                # Forward Head / Slouching (Nose dropping too close to the shoulders vertically)
                # A good posture implies the head is reasonably high above the shoulders.
                head_height = shoulder_cy - nose[1]
                head_ratio = head_height / shoulder_width
                
                # If head ratio drops heavily, they're slouching forward or looking down.
                if head_ratio < 0.65:
                    state = "slouching"
                    severity = max(severity, min(1.0, (0.65 - head_ratio) / 0.5))
                elif head_ratio < 0.8:
                    state = "forward_head"
                    severity = max(severity, min(1.0, (0.8 - head_ratio) / 0.5))
    
                # Shoulder Imbalance (Relaxed to relative vertical offset instead of strict angle)
                # Webcams are rarely perfectly horizontal.
                shoulder_y_diff = abs(l_shoulder[1] - r_shoulder[1])
                imbalance_ratio = shoulder_y_diff / shoulder_width
                
                if imbalance_ratio > 0.2: # 20% vertical difference relative to width
                    state = "imbalance"
                    severity = max(severity, min(1.0, imbalance_ratio / 0.5))
            else:
                state = "unknown"
            
        # Update Game State
        self.game_state.update(state, severity, inactive_seconds)
        
        return {
            "posture_state": state,
            "severity": severity,
            "keypoints": keypoints
        }
