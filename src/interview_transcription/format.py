import json
import os
from pathlib import Path

SPEAKER_COLORS = ["#e74c3c", "#27ae60", "#2980b9", "#8e44ad", "#e67e22", "#1abc9c"]


def format_raw_transcript(segments, speaker_labels):
    """Merge consecutive same-speaker segments into paragraphs."""
    paragraphs = []
    current_speaker = None
    current_text = []

    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue

        if speaker != current_speaker:
            if current_text:
                label = speaker_labels.get(current_speaker, current_speaker)
                paragraphs.append(f"{label}: {' '.join(current_text)}")
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_text:
        label = speaker_labels.get(current_speaker, current_speaker)
        paragraphs.append(f"{label}: {' '.join(current_text)}")

    return paragraphs


def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format (HH:MM:SS,mmm)."""
    assert seconds >= 0, "non-negative timestamp expected"
    ms = int(round(seconds * 1000))
    hours = ms // 3_600_000
    ms -= hours * 3_600_000
    minutes = ms // 60_000
    ms -= minutes * 60_000
    secs = ms // 1_000
    ms -= secs * 1_000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def write_srt(segments, speaker_labels, filepath):
    """Write segments to SRT subtitle file with speaker labels."""
    with open(filepath, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            speaker = seg.get("speaker", "UNKNOWN")
            label = speaker_labels.get(speaker, speaker)
            text = seg.get("text", "").strip().replace("-->", "->")
            f.write(f"{i}\n{start} --> {end}\n{label}: {text}\n\n")


def save_outputs(segments, transcript, cfg, output_dir=None):
    """Write TXT, SRT, and JSON output files. Returns dict of paths."""
    stem = Path(cfg.audio_path).stem
    if output_dir is None:
        output_dir = str(Path(cfg.audio_path).parent)
    os.makedirs(output_dir, exist_ok=True)

    paths = {
        "txt": os.path.join(output_dir, f"{stem}_transcript.txt"),
        "srt": os.path.join(output_dir, f"{stem}_transcript.srt"),
        "json": os.path.join(output_dir, f"{stem}_segments.json"),
    }

    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write(transcript)

    write_srt(segments, cfg.speaker_labels, paths["srt"])

    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    for fmt, path in paths.items():
        print(f"{fmt.upper()}: {path}")

    return paths


def display_transcript(transcript, speaker_labels):
    """Display color-coded transcript in a Jupyter notebook."""
    from IPython.display import display, HTML

    css = (
        "<style>"
        ".transcript-line { font-size: 110%; line-height: 1.6; margin-bottom: 12px; }"
        ".speaker-label { font-weight: bold; }"
        "</style>"
    )

    color_map = {}
    for i, label in enumerate(speaker_labels.values()):
        color_map[label] = SPEAKER_COLORS[i % len(SPEAKER_COLORS)]

    html_parts = [css]
    for line in transcript.split("\n\n"):
        if not line.strip():
            continue
        colored_line = line
        for label, color in color_map.items():
            if line.startswith(f"{label}:"):
                colored_line = line.replace(
                    f"{label}:",
                    f"<span class='speaker-label' style='color:{color}'>{label}:</span>",
                    1,
                )
                break
        html_parts.append(f"<div class='transcript-line'>{colored_line}</div>")

    display(HTML("\n".join(html_parts)))


def download_outputs(paths):
    """Trigger file downloads in Colab, or print paths locally."""
    try:
        from google.colab import files

        for path in paths.values():
            files.download(path)
    except ImportError:
        print("Files saved to:")
        for path in paths.values():
            print(f"  {path}")
