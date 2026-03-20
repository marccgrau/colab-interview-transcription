import os
import shutil

from .auth import load_secrets
from .format import (
    display_transcript,
    download_outputs,
    format_raw_transcript,
    save_outputs,
)
from .postprocess import postprocess_transcript
from .transcribe import align, diarize, prepare_audio, setup_device, transcribe


class TranscriptionResult:
    """Holds transcription results and provides save/display/download methods."""

    def __init__(self, transcript, segments, cfg):
        self.transcript = transcript
        self.segments = segments
        self.cfg = cfg
        self._output_paths = None

    def save(self, output_dir=None):
        """Save TXT, SRT, and JSON files. Returns dict of output paths."""
        self._output_paths = save_outputs(
            self.segments, self.transcript, self.cfg, output_dir
        )
        return self._output_paths

    def display(self):
        """Display color-coded transcript in a notebook."""
        display_transcript(self.transcript, self.cfg.speaker_labels)

    def download(self):
        """Download files in Colab, or print paths locally."""
        if not self._output_paths:
            self.save()
        download_outputs(self._output_paths)


class TranscriptionPipeline:
    """Interview transcription and speaker diarization pipeline.

    Usage::

        from interview_transcription import Config, TranscriptionPipeline

        cfg = Config(audio_path="interview.mp4", language="de", num_speakers=2)
        result = TranscriptionPipeline(cfg).run()
        result.save()
        result.display()
    """

    def __init__(self, cfg):
        self.cfg = cfg
        if not cfg.audio_path:
            raise ValueError("audio_path must be set in Config")

    def run(self):
        """Run the full pipeline. Returns a TranscriptionResult."""
        cfg = self.cfg

        if not os.path.isfile(cfg.audio_path):
            raise FileNotFoundError(f"Audio file not found: {cfg.audio_path}")

        # Secrets
        hf_token, llm_api_key = load_secrets(cfg)
        if not llm_api_key:
            cfg.enable_llm_postprocessing = False

        # Device
        device, compute_type = setup_device()

        # Pre-processing
        vocal_target = prepare_audio(cfg)

        # Transcribe -> Align -> Diarize
        audio, result, detected_language = transcribe(
            vocal_target, cfg, device, compute_type
        )
        result = align(audio, result, detected_language, device)
        result = diarize(audio, result, cfg, hf_token, device)

        # Format raw transcript
        raw_paragraphs = format_raw_transcript(result["segments"], cfg.speaker_labels)
        print(f"\nFormatted {len(raw_paragraphs)} speaker turns")

        # LLM post-processing
        transcript = postprocess_transcript(raw_paragraphs, cfg, llm_api_key)

        # Cleanup temp files from Demucs
        if os.path.isdir("temp_outputs"):
            shutil.rmtree("temp_outputs")

        return TranscriptionResult(transcript, result["segments"], cfg)
