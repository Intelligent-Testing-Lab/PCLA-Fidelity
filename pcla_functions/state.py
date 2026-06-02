from collections import deque
from pathlib import Path
import numpy as np
import cv2
import copy
import uuid

class CameraState:
    working_dir: Path | None = None
    current_step: int = 0
    SAVE_CARLA_FRAMES: bool = True
    SAVE_COSMOS_FRAMES: bool = False
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
    def save_numpy_frame(frame: np.ndarray) -> None:
        if CameraState.SAVE_CARLA_FRAMES:
            save_dir = CameraState.get_working_dir() / f"{CameraState.current_step}.png"
            cv2.imwrite(str(save_dir), frame)
    
    @staticmethod
    def add_frame(frame: np.ndarray) -> None:
        # might need to adjust to a deep copy    
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
                
    