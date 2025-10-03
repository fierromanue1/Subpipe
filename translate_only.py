import os
import time
import torch
import json
import logging
import logging.handlers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from tqdm import tqdm
import nltk
from nltk.tokenize import sent_tokenize
from pathlib import Path
from typing import List, Tuple
import re

# NLTK download (runtime, silent)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Environment settings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Config
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

paths = config["paths"]
log_dir = paths["logs_dir"]
target_lang = config["models"]["target_lang"]
translation_model = config["models"]["translation_model"]
max_line_length = config["subtitles"]["max_line_length"]
max_lines = config["subtitles"]["max_lines"]
language_code_map = config["models"]["language_code_map"]
use_regex_splitter = config.get("use_regex_splitter", False)  # Optional: Set to true for regex-based splitting

subs_original_file_json_path = Path(paths["subs_original_dir"]) / paths["subs_original_file_json"]
subs_translated_folder = Path(paths["subs_translated_dir"])
subs_translated_file_srt = subs_translated_folder / paths["subs_translated_file_srt"]
subs_translated_file_txt = subs_translated_folder / paths["subs_translated_file_txt"]
os.makedirs(subs_translated_folder ,exist_ok=True)

# Log settings
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, "translate.log")

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
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

def cleanup_gpu() -> None:
    """Utility to clean GPU memory."""
    try:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logging.info("GPU cache cleared")
    except Exception as e:
        logging.warning(f"GPU cleanup error: {e}")

def load_segments(json_file: Path) -> Tuple[List[Tuple[float, float, str]], str]:
    """Load transcription segments and language code from JSON."""
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data["segments"], data["lang"]
    except FileNotFoundError as e:
        logging.error(f"JSON file not found: {e}")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON format: {e}")
        raise

def split_sentences(text: str) -> List[str]:
    """Split text into sentences. Uses NLTK by default, or regex if configured."""
    if use_regex_splitter:
        # Simple regex splitter for multilingual support (splits on . ? ! followed by space/capital letter)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s(?=[A-Z])', text)
        return [s.strip() for s in sentences if s.strip()]
    else:
        # NLTK without language specification (defaults to English)
        try:
            return sent_tokenize(text)
        except LookupError as e:
            logging.error(f"NLTK resource error: {e}. Please run: import nltk; nltk.download('punkt_tab')")
            raise

def translate_segment(text: str, tokenizer: AutoTokenizer, model: AutoModelForSeq2SeqLM, target_lang: str, max_length: int) -> str:
    """Translate a single text segment with length and line limits."""
    if not text or len(text.strip()) < 3:
        return ""

    # Split into sentences
    sentences = split_sentences(text)
    if len(text) > max_line_length:
        sentences = [s[:max_line_length] for s in sentences][:max_lines]
    else:
        sentences = sentences[:max_lines]

    translated_sentences = []
    for sentence in sentences:
        encoded = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        with torch.no_grad():
            out = model.generate(
                **encoded,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(target_lang),
                max_length=max_length
            )
        translated_sentences.append(tokenizer.decode(out[0], skip_special_tokens=True))
    
    return " ".join(translated_sentences)

def save_srt_segments(segments: List[Tuple[float, float, str]], output_srt: Path) -> None:
    """Save translated segments to SRT."""
    os.makedirs(os.path.dirname(output_srt),exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for idx, (start, end, text) in enumerate(segments, 1):
            start_time_str = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start * 1000) % 1000):03}"
            end_time_str = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end * 1000) % 1000):03}"
            f.write(f"{idx}\n{start_time_str} --> {end_time_str}\n{text}\n\n")

def save_txt_segments(segments: List[Tuple[float, float, str]], output_txt: Path) -> None:
    """Save translated segments to TXT."""
    os.makedirs(os.path.dirname(output_txt),exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        for _, _, text in segments:
            f.write(text + "\n")

def translate(segments: List[Tuple[float, float, str]], source_lang: str, target_lang: str, output_srt: Path, output_txt: Path) -> None:
    """
    Translates segments using NLLB model and saves to SRT and TXT.
    
    Args:
        segments: List of (start, end, text) tuples from transcription.
        source_lang: Source language code (from Whisper auto-detection).
        target_lang: Target language code in NLLB format.
        output_srt: Path for SRT output.
        output_txt: Path for TXT output.
    """
    start_time = time.time()
    try:
        # Map Whisper's language code to NLLB format
        source_lang = language_code_map.get(source_lang, source_lang) or "eng_Latn"  # Fallback to English
        logging.info(f"Source Language: {source_lang}, Target Language: {target_lang}")
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(translation_model)
        logging.info(f"Loading {translation_model} model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(translation_model, torch_dtype=torch_dtype).to(device)

        translated_segments: List[Tuple[float, float, str]] = []
        max_length = 512
        tokenizer.src_lang = source_lang

        with tqdm(total=len(segments), desc="Translating segments", unit="seg") as pbar:
            for start, end, text in segments:
                translated_text = translate_segment(text, tokenizer, model, target_lang, max_length)
                translated_segments.append((start, end, translated_text))
                pbar.update(1)

        # Save outputs
        save_srt_segments(translated_segments, output_srt)
        save_txt_segments(translated_segments, output_txt)

        print(f"Translation completed. {len(translated_segments)} segments translated.")
        print(f"Duration: {time.time() - start_time:.2f} seconds")
        logging.info(f"Translation completed. {len(translated_segments)} segments translated.")

    except FileNotFoundError as e:
        logging.error(f"File error: {e}")
        raise
    except LookupError as e:
        logging.error(f"NLTK resource error: {e}. Please run: import nltk; nltk.download('punkt_tab')")
        raise
    except Exception as e:
        logging.error(f"Translation error: {e}")
        raise
    finally:
        cleanup_gpu()

if __name__ == "__main__":
    segments, source_lang = load_segments(subs_original_file_json_path)
    translate(segments, source_lang=source_lang, target_lang=target_lang, output_srt=subs_translated_file_srt, output_txt=subs_translated_file_txt)