from dataclasses import dataclass, field


@dataclass
class Config:
    audio_path: str = ""

    # Whisper
    whisper_model: str = "large-v3-turbo"  # or "large-v3" for max quality
    language: str | None = None  # None = auto-detect
    batch_size: int = 8  # reduce to 4 if OOM on T4

    # Diarization
    num_speakers: int | None = 2  # None = auto-detect
    min_speakers: int | None = None
    max_speakers: int | None = None

    # Pre-processing
    enable_stemming: bool = False  # vocal isolation via Demucs

    # LLM post-processing
    enable_llm_postprocessing: bool = True
    llm_provider: str = "openai"  # "openai" or "anthropic"
    llm_model: str = ""  # empty = use default for provider
    llm_system_prompt: str = ""  # empty = use default prompt

    # Speaker labels (maps SPEAKER_00 -> display name)
    speaker_labels: dict[str, str] = field(
        default_factory=lambda: {
            "SPEAKER_00": "Speaker 1",
            "SPEAKER_01": "Speaker 2",
        }
    )

    def get_llm_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-sonnet-4-20250514",
            "openai": "gpt-5.4-mini",
        }[self.llm_provider]
