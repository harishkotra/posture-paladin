import cv2
import numpy as np

class OverlayRenderer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Colors (BGR)
        self.color_green = (0, 255, 0)
        self.color_red = (0, 0, 255)
        self.color_yellow = (0, 255, 255)
        self.color_blue = (255, 0, 0)
        self.color_bg = (50, 50, 50)
        self.color_text = (255, 255, 255)

    def draw(self, frame, state, posture_data, fps=0, latency=0):
        # Draw skeleton and spine if pose data is available
        self._draw_pose(frame, posture_data)
        
        # Draw HUD elements
        self._draw_hud(frame, state, fps, latency, posture_data)
        
        return frame

    def _draw_pose(self, frame, posture_data):
        if not posture_data:
            return

        keypoints = posture_data.get('keypoints', [])
        if not keypoints:
            return

        # Assuming keypoints format [x, y, conf]
        # Common indices: 0:Nose, 5:LShoulder, 6:RShoulder, 11:LHip, 12:RHip
        
        def pt(idx):
             if idx < len(keypoints) and keypoints[idx][2] > 0.3:
                 return (int(keypoints[idx][0]), int(keypoints[idx][1]))
             return None
        
        nose = pt(0)
        l_shoulder = pt(5)
        r_shoulder = pt(6)
        l_hip = pt(11)
        r_hip = pt(12)

        color = self.color_green
        if posture_data.get('posture_state') != 'good':
            color = self.color_red if posture_data.get('severity', 0) > 0.7 else self.color_yellow

        # Draw spine proxy focusing on upper body
        if l_shoulder and r_shoulder:
            shoulder_cx = int((l_shoulder[0] + r_shoulder[0]) / 2)
            shoulder_cy = int((l_shoulder[1] + r_shoulder[1]) / 2)
            
            # Shoulders
            cv2.line(frame, l_shoulder, r_shoulder, color, 4)

            # Neck line if nose is visible
            if nose:
                cv2.line(frame, (shoulder_cx, shoulder_cy), nose, color, 2)
            
            if l_hip and r_hip:
                hip_cx = int((l_hip[0] + r_hip[0]) / 2)
                hip_cy = int((l_hip[1] + r_hip[1]) / 2)
                
                # Full Spine
                cv2.line(frame, (shoulder_cx, shoulder_cy), (hip_cx, hip_cy), color, 4)
                # Hips
                cv2.line(frame, l_hip, r_hip, color, 2)
            else:
                # Draw a simple vertical anchor representing the expected spine
                cv2.line(frame, (shoulder_cx, shoulder_cy), (shoulder_cx, shoulder_cy + 150), color, 4)

        # Draw head point
        if nose:
            cv2.circle(frame, nose, 6, color, -1)


    def _draw_hud(self, frame, state, fps, latency, posture_data):
        h, w, _ = frame.shape
        
        # Draw semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Helper to draw text with drop shadow
        def draw_text(text, position, scale=0.6, color=self.color_text, thickness=2):
            cv2.putText(frame, text, (position[0]+2, position[1]+2), self.font, scale, (0,0,0), thickness, cv2.LINE_AA)
            cv2.putText(frame, text, position, self.font, scale, color, thickness, cv2.LINE_AA)

        # ----- TOP LEFT: Health & Posture status -----
        draw_text("PosturePaladin", (10, 25), 0.7, self.color_yellow, 2)
        
        health_color = self.color_green if state.health > 50 else (self.color_yellow if state.health > 20 else self.color_red)
        draw_text(f"Health: {state.health}/100", (10, 50), 0.6, health_color, 2)
        
        p_state = posture_data.get('posture_state', 'unknown') if posture_data else "unknown"
        p_color = self.color_green if p_state == "good" else self.color_red
        draw_text(f"State: {p_state.upper()}", (200, 50), 0.6, p_color, 2)

        # ----- TOP RIGHT: XP, Level & Streak -----
        xp_text = f"XP: {state.xp}  | Lvl: {state.level}"
        draw_text(xp_text, (w - 250, 30), 0.6, self.color_blue, 2)
        draw_text(f"Streak: {state.streak_minutes}m", (w - 250, 55), 0.5, self.color_text, 1)

        # ----- BOTTOM: Metrics & Inactivity -----
        bar_height = int(h * 0.08)
        bl_overlay = frame.copy()
        cv2.rectangle(bl_overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(bl_overlay, 0.5, frame, 0.5, 0, frame)
        
        font_scale = max(0.4, w / 2000.0)
        
        active_color = self.color_red if state.inactive_seconds > 40 * 60 else self.color_text
        draw_text(f"Inactive: {int(state.inactive_seconds//60)}m {int(state.inactive_seconds%60)}s", (10, h - int(bar_height/3)), font_scale, active_color, 1)
        
        metrics_text = f"FPS: {fps:.1f} | Latency: {latency}ms"
        draw_text(metrics_text, (w - int(w * 0.15), h - int(bar_height/3)), font_scale, self.color_text, 1)

        privacy_mode = getattr(state, "privacy_mode_name", "Local Processing")
        draw_text(f"[SECURE] {privacy_mode}: Video NEVER leaves your machine.", (w//2 - int(w*0.2), h - int(bar_height/3)), font_scale, self.color_green, 1)
        
        # Boss mode indicator
        if getattr(state, "boss_mode_active", False):
            draw_text("! BOSS MODE ACTIVE !", (w//2 - 120, h//2), 1.0, self.color_red, 3)
