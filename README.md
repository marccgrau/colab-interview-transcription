# Interview Transcription & Speaker Diarization

Transcribe audio interviews with automatic speaker identification. Designed to run on Google Colab with GPU acceleration.

## Pipeline

1. **WhisperX** — transcription (faster-whisper) + forced alignment (wav2vec2) + speaker diarization (pyannote.audio)
2. **LLM post-processing** — punctuation, filler word removal, ASR error correction (OpenAI or Anthropic)
3. **Output** — speaker-labeled transcript (`.txt`), timed subtitles (`.srt`), raw segments (`.json`)

## Setup

### Prerequisites

1. **Hugging Face token** — required for pyannote diarization models
   - Create a token at [hf.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Accept terms for [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
2. **LLM API key** (optional) — for transcript cleanup
   - [OpenAI API key](https://platform.openai.com) (default) or [Anthropic API key](https://console.anthropic.com)
3. **Audio file** uploaded to Google Drive

### Secrets

**On Colab:** Add secrets via the Secrets panel (key icon in sidebar):
- `HF_TOKEN`
- `OPENAI_API_KEY` (or `ANTHROPIC_API_KEY`)

**Locally:** Create a `.env` file in the project root:
```
HF_TOKEN=hf_...
OPENAI_API_KEY=sk-...
```

## Usage on Google Colab

Open the notebook at `notebooks/Interview_Transcription_Diarization.ipynb` and run the cells:

```python
# 1. Install
!git clone https://github.com/marccgrau/colab-interview-transcription.git /content/interview-transcription
%pip install whisperx
%pip install -e /content/interview-transcription

# 2. Mount Google Drive
from google.colab import drive
drive.mount("/content/drive")

# 3. Configure
from interview_transcription import Config, TranscriptionPipeline

cfg = Config(
    audio_path="/content/drive/MyDrive/your-interview.mp4",
    language="de",              # or None for auto-detect
    num_speakers=2,             # or None for auto-detect
    speaker_labels={
        "SPEAKER_00": "Interviewer",
        "SPEAKER_01": "Subject",
    },
)

# 4. Run
result = TranscriptionPipeline(cfg).run()

# 5. Save & view
result.save()       # writes .txt, .srt, .json next to the audio file
result.display()    # color-coded transcript in the notebook
result.download()   # triggers browser download in Colab
```

## Configuration

All options are set via the `Config` dataclass:

| Parameter | Default | Description |
|---|---|---|
| `audio_path` | `""` | Path to audio/video file |
| `whisper_model` | `"large-v3-turbo"` | Whisper model (`"large-v3"` for max quality) |
| `language` | `None` | Language code (e.g. `"de"`) or `None` for auto-detect |
| `batch_size` | `8` | Reduce to `4` if OOM on T4 |
| `num_speakers` | `2` | Number of speakers, or `None` for auto-detect |
| `enable_stemming` | `False` | Vocal isolation via Demucs |
| `enable_llm_postprocessing` | `True` | Clean transcript with LLM |
| `llm_provider` | `"openai"` | `"openai"` or `"anthropic"` |
| `llm_model` | `""` | Empty = provider default (gpt-5.4 / claude-sonnet-4) |
| `llm_system_prompt` | `""` | Empty = built-in Swiss German prompt |
| `speaker_labels` | `{"SPEAKER_00": "Speaker 1", ...}` | Display names for speakers |

## Project Structure

```
├── src/interview_transcription/
│   ├── config.py        # Config dataclass
│   ├── auth.py          # Secret loading (.env / Colab secrets)
│   ├── transcribe.py    # WhisperX: transcribe, align, diarize
│   ├── postprocess.py   # LLM transcript cleanup
│   ├── format.py        # Output formatting (TXT, SRT, JSON, display)
│   └── pipeline.py      # Pipeline orchestration
├── notebooks/
│   └── Interview_Transcription_Diarization.ipynb
├── pyproject.toml
└── .env                 # secrets (not committed)
```
