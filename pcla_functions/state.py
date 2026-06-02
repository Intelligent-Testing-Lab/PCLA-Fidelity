from collections import deque
from .cosmos import generate_realistic_video, generate_request_params
from pathlib import Path
import numpy as np
import cv2
import copy
import uuid

class CameraState:
    working_dir: Path | None = None
    current_step: int = 0
    SAVE_CARLA_FRAMES: bool = False
    SAVE_COSMOS_FRAMES: bool = True
    _frame_buffer: deque[np.ndarray] = deque(maxlen=20)  # ring buffer, auto-evicts oldest
    
    @staticmethod
    def set_working_dir(name: str = "scenario") -> Path:
        frame_folder = Path.cwd() / "results" / f"{name}_{str(uuid.uuid4())}"
        frame_folder.mkdir(parents=True, exist_ok=True)
        CameraState.working_dir = frame_folder
        return frame_folder
        
    @staticmethod
    def get_working_dir() -> Path:
        return CameraState.working_dir if CameraState.working_dir is not None else CameraState.set_working_dir()
    
    @staticmethod
    def increment_step() -> None:
        CameraState.current_step += 1
    
    @staticmethod
    def save_numpy_frame(frame: np.ndarray, cosmos: bool = False) -> None:
        if CameraState.SAVE_CARLA_FRAMES and not cosmos:
            save_dir = CameraState.get_working_dir() / f"{CameraState.current_step}.png"
            cv2.imwrite(str(save_dir), frame)
        elif CameraState.SAVE_COSMOS_FRAMES and cosmos:
            save_dir = CameraState.get_working_dir() / f"cosmos_{CameraState.current_step}.png"
            cv2.imwrite(str(save_dir), frame)
            
    @staticmethod
    def add_frame(frame: np.ndarray) -> None:
        CameraState._frame_buffer.append(copy.deepcopy(frame))         

    @staticmethod
    def save_video_from_buffer(num_frames: int = 20, fps: float = 20.0) -> None:
        buffer = list(CameraState._frame_buffer)

        if not buffer:
            raise RuntimeError("Frame buffer is empty.")

        # Pad front with oldest frame if fewer than num_frames exist
        while len(buffer) < num_frames:
            buffer.insert(0, buffer[0])

        height, width = buffer[0].shape[:2]
        output_path = CameraState.get_working_dir() / f"{CameraState.current_step}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter at {output_path}")

        try:
            for frame in buffer:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                video_writer.write(frame)
        finally:
            video_writer.release()
            
    @staticmethod
    def get_cosmos_frame() -> np.ndarray:
        # get the last frames .mp4
        current_frame = CameraState.get_working_dir() / f"{CameraState.current_step}.mp4"

        req_params = generate_request_params(
            num_frames=20, fps=20, size="1024x512", input_reference=str(current_frame)
        )
        
        # block until receive raw bytes back from cosmos
        cosmos_raw = generate_realistic_video(req_params)
        
        tmp_path = "/dev/shm/_last_frame_tmp.mp4"
        with open(tmp_path, "wb") as f:
            f.write(cosmos_raw)  # video_bytes used directly here

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video from bytes.")

        # Seek to last frame
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            cap = cv2.VideoCapture(tmp_path)
            last_frame = None
            while True:
                ret, f = cap.read()
                if not ret:
                    break
                last_frame = f
            cap.release()
            if last_frame is None:
                raise RuntimeError("Could not read any frames from video.")
            CameraState.save_numpy_frame(last_frame, cosmos=True)
            return last_frame
        
        CameraState.save_numpy_frame(frame, cosmos=True)
            
        return frame

                
    