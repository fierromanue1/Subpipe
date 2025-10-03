import os
import subprocess
import logging
import sys
import json
from pathlib import Path

# Config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config["paths"]
log_dir = paths["logs_dir"]
ffmpeg_conf = config["ffmpeg"]

# Log settings
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "embed_subtitles.log")

logging.basicConfig(
    filename=log_file,
    level=getattr(logging, config["logging"]["level"]),
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def add_soft_subs(video, subs, output):
    """
    Adds soft subtitles to the video.
    """
    try:
        if not os.path.exists(video):
            raise FileNotFoundError(f"Not found: {video}")
        if not os.path.exists(subs):
            raise FileNotFoundError(f"SRT file not found: {subs}")

        output_dir = os.path.dirname(output)
        os.makedirs(output_dir, exist_ok=True)

        logging.info(f"Starting soft subtitles: {video} -> {output}")

        command = [
            "ffmpeg",
            "-i", video,
            "-i", subs,
            "-c:v", "copy",
            "-c:a", "copy",
            "-c:s", "mov_text",
            "-y",
            output
        ]
        result = subprocess.run(command, check=True, capture_output=True, text=True)

        logging.info(f"Succsess soft subtitles to the video.: {output}")
        logging.debug(f"FFmpeg stdout: {result.stdout}")
        logging.debug(f"FFmpeg stderr: {result.stderr}")
        print(f"Soft subtitles saved: {output}")

    except Exception as e:
        logging.error(f"Soft subs error: {str(e)}")
        print(f"Soft sub Error: {str(e)}")
        raise

def add_burned_subs(video: str, subs: str, output: str) -> None:
    try:
        if not os.path.exists(video):
            raise FileNotFoundError(f"Video not found: {video}")
        if not os.path.exists(subs):
            raise FileNotFoundError(f"SRT file not found: {subs}")

        os.makedirs(os.path.dirname(output), exist_ok=True)
        logging.info(f"Starting burned subtitles: {video} -> {output}")

        # Map subtitle position to FFmpeg Alignment
        position_map: dict[str, int] = {"bottom": 2, "top": 8, "center": 10}
        alignment = position_map.get(ffmpeg_conf["subtitle_position"], 2)  # Default to bottom

        style = (
            f"subtitles='{subs}':force_style="
            f"'FontName={ffmpeg_conf['font_name']},"
            f"FontSize={ffmpeg_conf['font_size']},"
            f"PrimaryColour={ffmpeg_conf['primary_colour']},"
            f"OutlineColour={ffmpeg_conf['outline_colour']},"
            f"BorderStyle={ffmpeg_conf['border_style']},"
            f"Outline={ffmpeg_conf['outline']},"
            f"Shadow={ffmpeg_conf['shadow']},"
            f"Alignment={alignment}'"
        )

        # 🔹 Burada sadece video input’u veriyoruz, subs değil
        command = ["ffmpeg", "-i", video, "-vf", style,
                   "-c:a", "copy",
                   "-c:v", "libx264",
                   "-preset", ffmpeg_conf["preset"],
                   "-crf", str(ffmpeg_conf["crf"]),
                   "-pix_fmt", ffmpeg_conf["pix_fmt"],
                   "-y", output]

        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Success: Burned subtitles added to {output}")
        logging.debug(f"FFmpeg stdout: {result.stdout}")
        logging.debug(f"FFmpeg stderr: {result.stderr}")
        print(f"Burned subtitles video saved: {output}")

    except Exception as e:
        logging.error(f"Burned subtitle embedding error: {e}")
        raise

if __name__ == "__main__":
    try:
        video_path = list(Path(paths["video_dir"]).glob("*.mp4"))[0].resolve().as_posix()
        subs_translated_path = (Path(paths["subs_translated_dir"])/ paths["subs_translated_file_srt"]).as_posix()
        output_soft = (Path(paths["video_with_subs_dir"])/ paths["output_soft"]).as_posix()
        output_burned = (Path(paths["video_with_subs_dir"])/ paths["output_burned"]).as_posix()

        mode = config["subtitles"].get("mode", "soft")  # Default soft

        if mode == "soft":
            add_soft_subs(video_path, subs_translated_path, output_soft)
        elif mode == "burned":
            add_burned_subs(video_path, subs_translated_path, output_burned)
        else:
            raise ValueError(f"Invailed subtitle mode: {mode}")

    except Exception as e:
        logging.error(f"Main error: {str(e)}")
        print(f"Main error: {str(e)}")
        sys.exit(1)