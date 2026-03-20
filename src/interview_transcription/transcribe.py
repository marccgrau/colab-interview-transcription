import os

import torch
import whisperx


def setup_device():
    """Detect GPU and return (device, compute_type)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Device: {device} | Compute type: {compute_type}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram:.1f} GB")

    return device, compute_type


def prepare_audio(cfg):
    """Apply optional vocal isolation via Demucs. Returns path to audio target."""
    if not cfg.enable_stemming:
        return cfg.audio_path

    print("Running Demucs vocal isolation...")
    ret = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{cfg.audio_path}" -o "temp_outputs"'
    )
    if ret != 0:
        print("WARNING: Demucs failed, using original audio.")
        return cfg.audio_path

    stem = os.path.splitext(os.path.basename(cfg.audio_path))[0]
    return os.path.join("temp_outputs", "htdemucs", stem, "vocals.wav")


def transcribe(audio_path, cfg, device, compute_type):
    """Run WhisperX transcription. Returns (audio_array, result_dict, detected_language)."""
    print(f"Loading Whisper model: {cfg.whisper_model}")
    model = whisperx.load_model(
        cfg.whisper_model,
        device,
        compute_type=compute_type,
        language=cfg.language,
    )

    audio = whisperx.load_audio(audio_path)
    print("Transcribing...")
    result = model.transcribe(audio, batch_size=cfg.batch_size, language=cfg.language)

    detected_language = result["language"]
    print(f"Detected language: {detected_language}")
    print(f"Segments: {len(result['segments'])}")

    del model
    torch.cuda.empty_cache()

    return audio, result, detected_language


def align(audio, result, detected_language, device):
    """Run forced alignment with wav2vec2. Returns updated result dict."""
    print(f"Loading alignment model for '{detected_language}'...")
    align_model, align_metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    print("Aligning...")
    result = whisperx.align(
        result["segments"],
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    print(f"Aligned segments: {len(result['segments'])}")

    del align_model
    torch.cuda.empty_cache()

    return result


def diarize(audio, result, cfg, hf_token, device):
    """Run speaker diarization with pyannote.audio. Returns updated result dict."""
    print("Running speaker diarization...")
    diarize_model = whisperx.DiarizationPipeline(
        use_auth_token=hf_token,
        device=device,
    )

    kwargs = {}
    if cfg.num_speakers is not None:
        kwargs["num_speakers"] = cfg.num_speakers
    if cfg.min_speakers is not None:
        kwargs["min_speakers"] = cfg.min_speakers
    if cfg.max_speakers is not None:
        kwargs["max_speakers"] = cfg.max_speakers

    diarize_segments = diarize_model(audio, **kwargs)
    result = whisperx.assign_word_speakers(diarize_segments, result)

    speakers = sorted({seg["speaker"] for seg in result["segments"] if "speaker" in seg})
    print(f"Speakers found: {speakers}")

    del diarize_model
    torch.cuda.empty_cache()

    return result
