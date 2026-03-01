import time

class GameState:
    def __init__(self):
        self.xp = 0
        self.level = 1
        self.health = 100
        self.posture_score = 1.0 # 0.0 to 1.0
        self.streak_minutes = 0
        self.penalties = 0

        self.last_update_time = time.time()
        self.good_posture_seconds = 0
        self.bad_posture_seconds = 0
        self.inactive_seconds = 0
        self.total_active_minutes = 0

        # Achievements
        self.iron_spine_unlocked = False
        self.desk_warrior_unlocked = False
        self.reborn_knight_unlocked = False

        self.boss_mode_active = False
        self.last_boss_mode_time = time.time()

        self.needs_coaching = False
        self.last_llm_trigger_time = 0

    def update(self, posture_state, severity, inactive_seconds):
        now = time.time()
        dt = now - self.last_update_time
        self.last_update_time = now

        self.inactive_seconds = inactive_seconds
        self.posture_score = 1.0 - severity

        # Update timings based on state
        if posture_state == "good":
            self.good_posture_seconds += dt
            if self.bad_posture_seconds > 0:
                # Correcting posture reward
                self.bad_posture_seconds = 0
                self.xp += 10
                self.trigger_coaching()

            if self.good_posture_seconds >= 60: # 1 min good posture
                self.xp += 5
                self.good_posture_seconds -= 60
                self.streak_minutes += 1
                self.total_active_minutes += 1

                # +1 health per 3 minutes aligned (checked every minute)
                if self.streak_minutes % 3 == 0:
                    self.health = min(100, self.health + 1)

        elif posture_state in ["slouching", "forward_head", "imbalance"]:
            self.bad_posture_seconds += dt
            self.good_posture_seconds = 0

            # Penalty: Slouch > 30s
            if self.bad_posture_seconds >= 30:
                self.xp = max(0, self.xp - 10)
                self.penalties += 1
                self.bad_posture_seconds -= 15 # reset partially so it doesn't spam every tick
                self.trigger_coaching()

            # Health drain: -2 health every 15s of slouch
            if self.bad_posture_seconds >= 15 and self.bad_posture_seconds % 15 < dt:
                 self.health = max(0, self.health - 2)
                 
        elif posture_state == "eyes_closed":
            self.bad_posture_seconds += dt
            self.good_posture_seconds = 0

            # Penalty: Sleeping > 5s
            if self.bad_posture_seconds >= 5:
                self.xp = max(0, self.xp - 20)
                self.penalties += 1
                self.bad_posture_seconds -= 5 
                self.trigger_coaching()

            # Health drain: -3 health every 5s of sleeping
            if self.bad_posture_seconds >= 5 and self.bad_posture_seconds % 5 < dt:
                 self.health = max(0, self.health - 3)
                 
        # Inactivity Penalty
        if self.inactive_seconds >= 40 * 60:
            if self.inactive_seconds % 60 < dt: # apply per minute after 40 mins
                self.xp = max(0, self.xp - 20)
                self.trigger_coaching()

        # Check Levels
        self._check_levels()
        
        # Check Achievements
        self._check_achievements()

        # Boss Mode: Every 45 minutes
        if now - self.last_boss_mode_time > 45 * 60:
            self.boss_mode_active = True
            self.last_boss_mode_time = now
            self.trigger_coaching()

        # Periodic LLM check (e.g., if health drops low)
        if self.health < 30 and now - self.last_llm_trigger_time > 300: # max trigger once per 5 mins for health
            self.trigger_coaching()

    def _check_levels(self):
        new_level = 1
        if self.xp >= 2500:
            new_level = 5
        elif self.xp >= 1200:
            new_level = 4
        elif self.xp >= 600:
            new_level = 3
        elif self.xp >= 200:
            new_level = 2
            
        if new_level > self.level:
            self.level = new_level
            self.trigger_coaching()

    def _check_achievements(self):
        if self.streak_minutes >= 60 and not self.iron_spine_unlocked:
            self.iron_spine_unlocked = True
            self.xp += 100
            self.trigger_coaching()
            
        if self.total_active_minutes >= 180 and not self.desk_warrior_unlocked:
            self.desk_warrior_unlocked = True
            self.xp += 200
            self.trigger_coaching()

        if self.health > 80 and not self.reborn_knight_unlocked:
            # simple mock, assuming they were low at some point
            if self.penalties > 5:
                self.reborn_knight_unlocked = True
                self.xp += 150
                self.trigger_coaching()

    def trigger_coaching(self):
        now = time.time()
        # Debounce coaching to avoid spamming the LLM
        if now - self.last_llm_trigger_time > 30: # 30s cooldown
            self.needs_coaching = True

    def get_summary_state(self):
        return {
            "posture_score": round(self.posture_score, 2),
            "inactive_minutes": round(self.inactive_seconds / 60, 1),
            "xp": self.xp,
            "level": self.level,
            "health": self.health,
            "streak": self.streak_minutes
        }
