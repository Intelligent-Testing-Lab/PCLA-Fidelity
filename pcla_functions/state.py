from pathlib import Path
import numpy as np
import cv2
import uuid

class CameraState:
    working_dir: Path | None = None
    current_step: int = 0
    SAVE_CAMERA_FRAMES: bool = True
    
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
        print("Saving array to file")
        save_dir = CameraState.get_working_dir() / f"{CameraState.current_step}.png"
        cv2.imwrite(str(save_dir), frame)
        
        
                
    