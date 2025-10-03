import os
import time
import torch
import json
import logging
import logging.handlers
from faster_whisper import WhisperModel
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple

# Environment Settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config["paths"]
source_lang = config["models"]["source_lang"]
log_dir = paths["logs_dir"]
language_code_map = config["models"]["language_code_map"]
whisper_model = config["models"]["whisper_model"]
whisper_transcription = config["whisper_transcription"]

# Log settings
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "transcribe.log")

handler = logging.handlers.RotatingFileHandler(
    log_file,
    maxBytes=int(config["logging"]["max_file_size"].replace("MB", "")) * 1024 * 1024,
    backupCount=5
)
logging.basicConfig(
    handlers=[handler],
    level=getattr(logging, config["logging"]["level"]),
    format=config["logging"]["format"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

def cleanup_gpu() -> None:
    """Utility to clean GPU memory."""
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("GPU cache cleared")
    except Exception as e:
        logging.warning(f"GPU cleanup error: {e}")

def save_json_segments(segments: List[Tuple[float, float, str]], lang: str, output_json: Path) -> None:
    """Save transcription segments to JSON."""
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump({"segments": segments, "lang": lang}, f, ensure_ascii=False, indent=2)

def save_srt_segments(segments: List[Tuple[float, float, str]], output_srt: Path) -> None:
    """Save transcription segments to SRT."""
    os.makedirs(os.path.dirname(output_srt), exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, 1):
            start_time_str = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start * 1000) % 1000):03}"
            end_time_str = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end * 1000) % 1000):03}"
            f.write(f"{idx}\n{start_time_str} --> {end_time_str}\n{text}\n\n")

def transcribe(audio_file: Path, output_json: Path, output_srt: Path, source_lang: str = "auto") -> None:
    """
    Transcribes audio file using Whisper model and saves to JSON and SRT.
    
    Args:
        audio_file: Path to audio file.
        output_json: Path for JSON output.
        output_srt: Path for SRT output.
        source_lang: Source language code or 'auto' for auto-detection.
    """
    start_time = time.time()
    try:
        logging.info("Loading Whisper model...")
        model = WhisperModel(whisper_model, device=device, compute_type="int8_float16")

        segments, info = model.transcribe(
            str(audio_file),
            beam_size=whisper_transcription["beam_size"],
            language=None if source_lang == "auto" else source_lang,
            task="transcribe",
            vad_filter=whisper_transcription["vad_filter"],
            word_timestamps=whisper_transcription["word_timestamps"],
            vad_parameters=dict(min_silence_duration_ms=whisper_transcription["min_silence_duration_ms"]),
            max_initial_timestamp=whisper_transcription["max_initial_timestamp"]
        )

        transcription_segments: List[Tuple[float, float, str]] = []
        for segment in tqdm(segments, desc="Processing segments", unit="seg"):
            transcription_segments.append((segment.start, segment.end, segment.text.strip()))
        
        logging.info(f"Detected Lang: {info.language}, Prob: {info.language_probability:.2f}")
        logging.info(f"Total segments: {len(transcription_segments)}")

        # Convert Whisper language code to NLLB format
        nllb_lang = language_code_map.get(info.language, info.language)

        # Save outputs
        save_json_segments(transcription_segments, nllb_lang, output_json)
        save_srt_segments(transcription_segments, output_srt)

        print(f"Transcription completed. {len(transcription_segments)} segments found.")
        print(f"Duration: {time.time() - start_time:.2f} seconds")
        logging.info(f"Transcription completed. {len(transcription_segments)} segments found.")
    
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except Exception as e:
        logging.error(f"Transcription error: {e}")
        raise
    finally:
        cleanup_gpu()

if __name__ == "__main__":
    audio_file_path = Path(paths["audio_dir"]) / paths["audio_file"]
    subs_original_file_path = Path(paths["subs_original_dir"]) / paths["subs_original_file"]
    subs_original_file_json_path = Path(paths["subs_original_dir"]) / paths["subs_original_file_json"]
    
    transcribe(audio_file_path, subs_original_file_json_path, subs_original_file_path, source_lang)