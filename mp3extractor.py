import os
import json
import logging
import logging.handlers
import subprocess
from moviepy import VideoFileClip
from pathlib import Path
from typing import Optional

# Environment setting
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config["paths"]
ffmpeg_conf = config.get("ffmpeg", {})
logging_config = config["logging"]
use_ffmpeg = config.get("audio_extraction", {}).get("use_ffmpeg", False)

# Log settings
os.makedirs(paths["logs_dir"], exist_ok=True)
log_file = os.path.join(paths["logs_dir"], logging_config.get("log_file", "pipeline.log"))

handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=int(logging_config["max_file_size"].replace("MB", "")) * 1024 * 1024,
    backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=getattr(logging, logging_config["level"]),
    format=logging_config["format"]
)

def select_video_file() -> str:
    """Select video file from default directory."""
    video_files = list(Path(paths["video_dir"]).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {paths['video_dir']}")
    if len(video_files) > 1:
        logging.warning(f"Multiple MP4 files found, using the first one: {video_files[0]}")
    return video_files[0].resolve().as_posix()

def extract_audio_with_ffmpeg(video_file: str, output_audio: str) -> None:
    """Extract audio using FFmpeg."""
    try:
        command = ["ffmpeg"]
        if ffmpeg_conf.get("hwaccel", "none") != "none":
            command.extend(["-hwaccel", ffmpeg_conf["hwaccel"]])
        command.extend(["-i", video_file, "-vn", "-acodec", "mp3", "-y", output_audio])
        
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"FFmpeg audio extraction successful: {output_audio}")
        logging.debug(f"FFmpeg stdout: {result.stdout}")
        logging.debug(f"FFmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg error: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"FFmpeg audio extraction error: {e}")
        raise

def extract_audio_with_moviepy(video_file: str, output_audio: str) -> None:
    """Extract audio using MoviePy."""
    try:
        video = VideoFileClip(video_file)
        os.makedirs(os.path.dirname(output_audio), exist_ok=True)
        video.audio.write_audiofile(output_audio)
        video.close()
        logging.info(f"MoviePy audio extraction successful: {output_audio}")
    except Exception as e:
        logging.error(f"MoviePy audio extraction error: {e}")
        raise

def extract_audio_from_video(video_file: str, output_audio: str) -> None:
    """
    Extracts audio from video and saves it in the specified format.

    Args:
        video_file: Path to the video file (e.g., 'video.mp4').
        output_audio: Path to the output audio file (e.g., 'audio.mp3').
    """
    try:
        if not os.path.exists(video_file):
            raise FileNotFoundError(f"Video file not found: {video_file}")

        logging.info(f"Extracting audio from {video_file} to {output_audio}")
        if use_ffmpeg:
            extract_audio_with_ffmpeg(video_file, output_audio)
        else:
            extract_audio_with_moviepy(video_file, output_audio)
        print(f"Audio successfully extracted and saved as {output_audio}")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Audio extraction error: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    try:
        video_file = select_video_file()
        output_audio = (Path(paths["audio_dir"]) / paths["audio_file"]).as_posix()
        output_audio_folder = Path(paths['audio_dir'])
        os.makedirs(output_audio_folder,exist_ok=True)
        extract_audio_from_video(video_file, output_audio)
    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Main error: {e}")
        print(f"Error: {e}")
        raise