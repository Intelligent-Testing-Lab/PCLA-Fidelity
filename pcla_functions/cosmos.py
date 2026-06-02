from typing import Optional
from pathlib import Path
import requests

DEFAULT_POSITIVE = "A photorealistic real-world dashcam video, highly detailed, cinematic lighting."
DEFAULT_NEGATIVE = "video game, CGI, CARLA simulation, low quality"

def generate_realistic_video(params: dict) -> bytes:
    url = params.get("url", "http://localhost:8000/v1/videos/sync")
    input_reference = params["input_reference"]

    form_data = {k: str(v) for k, v in params.items() if k not in ("url", "input_reference")}

    with open(input_reference, "rb") as f:
        response = requests.post(
            url,
            files={"input_reference": (Path(input_reference).name, f, "video/mp4")},
            data=form_data,
        )

    response.raise_for_status()
    return response.content

def generate_request_params(
        num_frames: int = 20, 
        fps: int = 20, 
        size: str = "1024x512",
        input_reference: str = "",
        prompts : tuple = (DEFAULT_POSITIVE, DEFAULT_NEGATIVE),
    ) -> dict:
    
    return {
        "input_reference": input_reference,
        "prompt": prompts[0],
        "negative_prompt": prompts[1],
        "size": size,
        "num_frames": num_frames,
        "fps": fps,
    }