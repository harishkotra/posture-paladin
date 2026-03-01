import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from vision_agents.core import Agent, User
from vision_agents.plugins import getstream, gemini, elevenlabs, deepgram, ultralytics

from game.game_state import GameState
from processors.posture_processor import PostureProcessor
from ui.overlay_renderer import OverlayRenderer

import cv2
import time
import av
import aiortc
from typing import Optional
import functools
from concurrent.futures import ThreadPoolExecutor
from vision_agents.core.processors import VideoProcessorPublisher
from vision_agents.core.utils.video_forwarder import VideoForwarder
from vision_agents.core.utils.video_track import QueuedVideoTrack

class PosturePaladinProcessor(VideoProcessorPublisher):
    def __init__(self, game_state, posture_processor, overlay_renderer, yolo_processor):
        self.game_state = game_state
        self.posture_processor = posture_processor
        self.overlay_renderer = overlay_renderer
        self.yolo_processor = yolo_processor
        self.frame_count = 0
        self.start_time = time.time()
        self._video_forwarder = None
        self._shutdown = False
        self._executor = ThreadPoolExecutor(max_workers=2)  # for CPU-heavy inference
        self._agent = None
        self._video_track = QueuedVideoTrack(fps=30)
        
    @property
    def name(self) -> str:
        return "posture_paladin_processor"

    def attach_agent(self, agent: "Agent") -> None:
        self._agent = agent
        # Also attach the inner processor just in case
        self.yolo_processor.attach_agent(agent)

    def publish_video_track(self) -> aiortc.VideoStreamTrack:
        return self._video_track

    async def _process_frame(self, frame: av.VideoFrame):
        if self._shutdown:
            return

        frame_array = frame.to_ndarray(format="rgb24")
        
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            print(f">>> Processed {self.frame_count} frames, pushed to video_track queue of size {self._video_track.frame_queue.qsize()}")
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # 1. Ask YOLO to detect and annotate on the frame array
        annotated_array, pose_data = await self.yolo_processor.add_pose_to_ndarray(frame_array)
        
        # 2. Extract Keypoints for our Game logic
        keypoints = []
        if pose_data and "persons" in pose_data:
            for person in pose_data["persons"]:
                if "keypoints" in person:
                    keypoints = person["keypoints"]
                    break # just take the first person for paladin logic
                    
        # 3. Analyze Posture - run in thread so it doesn't block the event loop
        loop = asyncio.get_event_loop()
        posture_result = await loop.run_in_executor(
            self._executor,
            functools.partial(self.posture_processor.process, keypoints, frame_array)
        )
        
        # 4. Update Game State
        if posture_result:
            self.game_state.update(
                posture_result.get("state", "unknown"),
                posture_result.get("health_delta", 0.0),
                inactive_seconds=posture_result.get("inactive_seconds", 0)
            )
        else:
            self.game_state.update("unknown", 0.0, inactive_seconds=0) 
            
        # 5. Draw Overlays on the annotated array
        self.overlay_renderer.draw(annotated_array, self.game_state, posture_result, fps, 0)
        
        # 6. Convert back to frame
        new_frame = av.VideoFrame.from_ndarray(annotated_array, format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        
        # Output frame back to the call
        await self._video_track.add_frame(new_frame)
        
        # Trigger LLM coaching (fire-and-forget in background - NEVER block the frame loop)
        if getattr(self.game_state, 'needs_coaching', False) and self._agent:
            self.game_state.needs_coaching = False
            posture_state = posture_result.get('posture_state', 'unknown') if posture_result else 'unknown'
            coaching_msgs = {
                "slouching": "Warrior, straighten your spine! Your health is fading!",
                "forward_head": "Pull your head back! You are leaning too far forward!",
                "imbalance": "Level your shoulders! You are tilting to one side!",
                "eyes_closed": "Wake up, warrior! Sleeping on duty costs you dearly!",
                "good": f"Well done! Keep it up! Health: {self.game_state.health}, XP: {self.game_state.xp}",
            }
            msg = coaching_msgs.get(posture_state, f"Check your posture! State: {posture_state}")

            async def _do_coach(m):
                try:
                    if hasattr(self._agent, 'llm') and self._agent.llm:
                        await self._agent.llm.simple_response(m)
                    else:
                        print(f"[Coach] {m}")
                except Exception as e:
                    print(f"Coaching error (non-fatal): {e}")

            asyncio.create_task(_do_coach(msg))  # fire-and-forget

        return new_frame

    async def process_video(
        self,
        incoming_track: aiortc.VideoStreamTrack,
        participant_id: Optional[str],
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        self._video_forwarder = shared_forwarder
        if self._video_forwarder is None:
             self._video_forwarder = VideoForwarder(
                incoming_track,
                max_buffer=1,
                fps=10.0,
                name="paladin_forwarder",
            )
            
        self._video_forwarder.add_frame_handler(
            self._handle_frame_wrapper, fps=10.0, name="paladin_drawer"
        )
        
    async def _handle_frame_wrapper(self, frame: av.VideoFrame):
        if self._shutdown:
            return
        await self._process_frame(frame)

    async def stop_processing(self) -> None:
        if self._video_forwarder is not None:
             await self._video_forwarder.remove_frame_handler(self._handle_frame_wrapper)
             self._video_forwarder = None

    async def close(self) -> None:
        self._shutdown = True
        self.yolo_processor._shutdown = True # Ensure nested processor closes
        await self.stop_processing()


from vision_agents.core.runner import Runner
from vision_agents.core.agents.agent_launcher import AgentLauncher
import argparse

def create_agent(privacy_mode="Cloud Voice Mode"):
    agent_user = User(name="Posture Paladin", id="paladin_agent")
    game_state = GameState()
    posture_proc = PostureProcessor(game_state)
    renderer = OverlayRenderer()
    
    # Initialize YOLO manually
    yolo_proc = ultralytics.YOLOPoseProcessor(model_path="yolo11n-pose.pt", device="cpu")
    custom_processor = PosturePaladinProcessor(game_state, posture_proc, renderer, yolo_proc)
    
    with open("instructions/posture_paladin.md", "r") as f:
        instructions = f.read()

    # Apply Privacy Toggle Logic
    # Use standard GeminiLLM (HTTP-only, no persistent WebSocket) instead of Realtime.
    # GeminiRealtime kept a permanent WebSocket that competed with Stream's WebSocket
    # on the same event loop, causing the video to freeze every 30-60 seconds.
    from vision_agents.plugins.gemini.gemini_llm import GeminiLLM
    llm = GeminiLLM()
    if privacy_mode == "Cloud Voice Mode":
        print("🟡 Cloud Voice Mode (local HUD, text LLM coaching only)")
    elif privacy_mode == "Local Processing Mode":
        print("🟢 Local Processing Mode (Screen Only)")
    elif privacy_mode == "Voice Disabled":
        print("🔴 Voice Disabled Mode")
        
    # Inject mode onto game state for rendering it onto the UI
    game_state.privacy_mode_name = privacy_mode

    agent = Agent(
        edge=getstream.Edge(),
        agent_user=agent_user,
        instructions=instructions,
        llm=llm,
        processors=[
            custom_processor,
        ],
    )
    return agent

async def join_call(agent: Agent, call_type: str, call_id: str):
    call = await agent.edge.create_call(call_id, type=call_type)
    async with agent.join(call):
        await agent._call_ended_event.wait()

def main():
    import sys
    print("Initializing PosturePaladin...")
    
    # Parse CLI for privacy mode before handing to the Runner
    parser = argparse.ArgumentParser(description="PosturePaladin Privacy Configurator", add_help=False)
    parser.add_argument("--privacy", type=str, choices=["local", "cloud", "disabled"], default="cloud")
    args, remaining_argv = parser.parse_known_args()
    
    # Map CLI arg to internal state string
    pm_mapping = {
        "local": "Local Processing Mode",
        "cloud": "Cloud Voice Mode",
        "disabled": "Voice Disabled"
    }
    active_privacy_mode = pm_mapping[args.privacy]
    
    sys.argv = [sys.argv[0]] + remaining_argv

    launcher = AgentLauncher(
        create_agent=lambda: create_agent(privacy_mode=active_privacy_mode),
        join_call=join_call,
    )
    runner = Runner(launcher=launcher)
    runner.cli()

if __name__ == "__main__":
    main()
