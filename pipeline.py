import os
import json
import logging
import logging.handlers
import argparse
from pathlib import Path
import torch
from mp3extractor import extract_audio_from_video
from transcribe_only import transcribe
from translate_only import load_segments, translate
from embed_subtitles import add_soft_subs, add_burned_subs

# Environment setting
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config['paths']
models = config['models']
subtitles = config['subtitles']
logging_config = config['logging']

# Log settings
os.makedirs(paths['logs_dir'], exist_ok=True)
log_file = os.path.join(paths['logs_dir'], 'pipeline.log')

handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=int(logging_config['max_file_size'].replace('MB', '')) * 1024 * 1024,
    backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=getattr(logging, logging_config['level']),
    format=logging_config['format']
)

def cleanup_gpu() -> None:
    """Utility to clean GPU memory."""
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("GPU cache cleared")
    except Exception as e:
        logging.warning(f"GPU cleanup skipped: {e}")

def select_video_file(video_arg: str | None) -> str:
    """Select video file from argument or default directory."""
    if video_arg:
        video_path = Path(video_arg).resolve()
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path.as_posix()
    
    video_files = list(Path(paths["video_dir"]).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No MP4 files found in {paths['video_dir']}")
    if len(video_files) > 1:
        logging.warning(f"Multiple MP4 files found, using the first one: {video_files[0]}")
    return video_files[0].resolve().as_posix()

def step_extract_audio(video_file: str, output_audio: Path) -> None:
    """Extract audio from video."""
    logging.info("Step 1: Extracting audio from video...")
    try:
        extract_audio_from_video(video_file, str(output_audio))
        logging.info(f"Successfully extracted audio file: {output_audio}")
    except Exception as e:
        logging.error(f"Audio extraction failed: {e}")
        raise

def step_transcribe(output_audio: Path, subs_original_file_json: Path, subs_original_file: Path) -> None:
    """Transcribe audio to segments."""
    logging.info("Step 2: Transcribing audio...")
    try:
        transcribe(output_audio, subs_original_file_json, subs_original_file)
        logging.info(f"Transcription completed: {subs_original_file_json}")
        cleanup_gpu()
    except Exception as e:
        logging.error(f"Transcription failed: {e}")
        raise

def step_translate(subs_original_file_json: Path, subs_translated_file_srt: str, subs_translated_file_txt: Path) -> None:
    """Translate transcription segments."""
    logging.info("Step 3: Translating transcription...")
    try:
        segments, source_lang = load_segments(subs_original_file_json)
        translate(
            segments,
            source_lang=source_lang,
            target_lang=models['target_lang'],
            output_srt=subs_translated_file_srt,
            output_txt=str(subs_translated_file_txt)
        )
        logging.info(f"Translation completed: {subs_translated_file_srt}")
        cleanup_gpu()
    except Exception as e:
        logging.error(f"Translation failed: {e}")
        raise

def step_subtitles(video_file: str, subs_translated_file_srt: str, output_soft: Path, output_burned: Path, mode: str) -> None:
    """Add subtitles to video."""
    logging.info(f"Step 4: Adding subtitles (mode: {mode})...")
    try:
        if mode == 'soft':
            add_soft_subs(video_file, subs_translated_file_srt, str(output_soft))
            logging.info(f"Soft subtitled video saved: {output_soft}")
        elif mode == 'burned':
            add_burned_subs(video_file, subs_translated_file_srt, str(output_burned))
            logging.info(f"Burned subtitled video saved: {output_burned}")
        else:
            raise ValueError(f"Invalid subtitle mode: {mode}")
    except Exception as e:
        logging.error(f"Subtitle embedding failed: {e}")
        raise

def run_pipeline(steps: list[str], mode: str | None = None, video_arg: str | None = None) -> None:
    """
    Run selected pipeline steps.

    Args:
        steps: List of steps to run (extract, transcribe, translate, subtitles).
        mode: Subtitle mode ('soft' or 'burned'), overrides config.json.
        video_arg: Custom video file path (optional).
    """
    try:
        # Select video file
        video_file = select_video_file(video_arg)

        # Paths
        output_audio = Path(paths['audio_dir']) / paths['audio_file']
        subs_original_file = Path(paths['subs_original_dir']) / paths['subs_original_file']
        subs_original_file_json = Path(paths['subs_original_dir']) / paths['subs_original_file_json']
        output_audio_folder = Path(paths['audio_dir'])
        subs_translated_folder = Path(paths["subs_translated_dir"])
        subs_translated_file_srt = (subs_translated_folder / paths['subs_translated_file_srt']).as_posix()
        subs_translated_file_txt = subs_translated_folder / paths['subs_translated_file_txt']
        output_soft = Path(paths['video_with_subs_dir']) / paths['output_soft']
        output_burned = Path(paths['video_with_subs_dir']) / paths['output_burned']
        os.makedirs(output_audio_folder,exist_ok=True)
        # os.makedirs(output_audio,exist_ok=True)
        final_mode = mode or subtitles.get('mode', 'burned')

        if "extract" in steps:
            step_extract_audio(video_file, output_audio)
        
        if "transcribe" in steps:
            step_transcribe(output_audio, subs_original_file_json, subs_original_file)
        
        if "translate" in steps:
            step_translate(subs_original_file_json, subs_translated_file_srt, subs_translated_file_txt)
        
        if "subtitles" in steps:
            step_subtitles(video_file, subs_translated_file_srt, output_soft, output_burned, final_mode)

        print("Pipeline completed successfully!")
        logging.info("Pipeline completed successfully.")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        print(f"Error: {e}")
        raise
    except ValueError as e:
        logging.error(f"Configuration error: {e}")
        print(f"Error: {e}")
        raise
    except Exception as e:
        logging.error(f"Pipeline error: {e}")
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Subtitle processing pipeline from video to subtitled video.",
        epilog="""Examples:
            python pipeline.py --video "FilePath"     # Run full pipeline for custom video.
            python pipeline.py                        # Run full pipeline (for : data/video/sample.mp4 | mode from config)
            python pipeline.py --steps transcribe     # Only transcribe
            python pipeline.py --steps subtitles --mode soft   # Add soft subtitles
            python pipeline.py --steps subtitles --mode burned # Add burned subtitles
            python pipeline.py --steps extract transcribe  # Extract + Transcribe""",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        default=["extract", "transcribe", "translate", "subtitles"],
        help="Pipeline steps to run. Options: extract, transcribe, translate, subtitles"
    )
    parser.add_argument(
        "--mode",
        choices=["soft", "burned"],
        help="Subtitle mode: soft or burned (overrides config.json)"
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Custom video file path (optional)"
    )
    args = parser.parse_args()
    run_pipeline(args.steps, mode=args.mode, video_arg=args.video)