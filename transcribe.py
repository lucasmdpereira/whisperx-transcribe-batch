#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import env_config

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_VOICE_FILTER = "highpass=f=80,lowpass=f=7600,afftdn=nr=12:nf=-40:tn=1"
DEFAULT_NORMALIZE_FILTER = "loudnorm=I=-18:TP=-1.5:LRA=11"
DEFAULT_SILENCE_THRESHOLD = "-30dB"
DEFAULT_SILENCE_DURATION = 2.0
DEFAULT_KEEP_SILENCE_SECONDS = 0.25


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def resolve_hf_token(token_file_arg: str | None) -> str:
    direct = env_config.get_str("HF_TOKEN")
    if direct:
        return direct.strip()
    path_str = token_file_arg or env_config.get_str("HF_TOKEN_FILE")
    if not path_str:
        raise SystemExit(
            "error: Hugging Face token not configured. Set HF_TOKEN or HF_TOKEN_FILE in .env,"
            " or pass --token-file. Diarization requires a HF token."
        )
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Hugging Face token file not found: {path}")
    return path.read_text(encoding="utf-8").strip()


def resolve_whisperx_binary() -> Path:
    configured = env_config.get_path("WHISPERX_BIN")
    if configured and configured.exists():
        return configured
    venv_bin = SCRIPT_DIR / ".venv" / "bin" / "whisperx"
    if venv_bin.exists():
        return venv_bin
    on_path = shutil.which("whisperx")
    if on_path:
        return Path(on_path)
    raise SystemExit(
        "error: whisperx binary not found. Set WHISPERX_BIN in .env, install whisperx"
        " in .venv/, or make sure it is on PATH."
    )


def normalize_speaker(label: str, mapping: dict[str, str]) -> str:
    if label not in mapping:
        mapping[label] = f"speaker_{len(mapping):03d}"
    return mapping[label]


def write_normalized_outputs(
    json_path: Path,
    normalized_json_path: Path,
    normalized_txt_path: Path | None = None,
) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    speaker_map: dict[str, str] = {}
    lines = []

    for segment in data.get("segments", []):
        original_speaker = segment.get("speaker") or "SPEAKER_UNKNOWN"
        speaker = normalize_speaker(original_speaker, speaker_map)
        text = (segment.get("text") or "").strip()
        if text:
            lines.append(f"{speaker}: {text}")
        segment["source_speaker"] = original_speaker
        segment["speaker"] = speaker

    for word in data.get("word_segments", []):
        original_speaker = word.get("speaker")
        if original_speaker:
            word["source_speaker"] = original_speaker
            word["speaker"] = normalize_speaker(original_speaker, speaker_map)

    data["speaker_map"] = speaker_map
    normalized_json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if normalized_txt_path:
        normalized_txt_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def remove_extra_outputs(output_dir: Path, stem: str, keep: set[Path]) -> None:
    for path in output_dir.glob(f"{stem}*"):
        if path.is_file() and path not in keep:
            path.unlink()


def stem_without_audio_suffix(path: Path) -> str:
    return re.sub(r"\.(wav|mp3|m4a|mka|webm|ogg|opus|flac|aac)$", "", path.name, flags=re.IGNORECASE)


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not generate unique name for: {path}")


def treated_audio_path_for(input_path: Path, backup_dir: Path) -> Path:
    return backup_dir / f"{stem_without_audio_suffix(input_path)}.wav"


def build_audio_filters(
    trim_silence: bool,
    silence_threshold: str,
    silence_duration: float,
    keep_silence_seconds: float,
) -> str:
    filters = [DEFAULT_VOICE_FILTER]
    if trim_silence:
        filters.append(
            "silenceremove="
            "start_periods=1:"
            f"start_duration={silence_duration}:"
            f"start_threshold={silence_threshold}:"
            "stop_periods=-1:"
            f"stop_duration={silence_duration}:"
            f"stop_threshold={silence_threshold}:"
            f"stop_silence={keep_silence_seconds}"
        )
    filters.append(DEFAULT_NORMALIZE_FILTER)
    return ",".join(filters)


def treat_audio(
    input_path: Path,
    output_path: Path,
    trim_silence: bool,
    silence_threshold: str,
    silence_duration: float,
    keep_silence_seconds: float,
    limit_seconds: float | None,
) -> None:
    filters = build_audio_filters(trim_silence, silence_threshold, silence_duration, keep_silence_seconds)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-af",
        filters,
        "-sample_fmt",
        "s16",
    ]
    if limit_seconds:
        cmd += ["-t", f"{limit_seconds:.3f}"]
    cmd.append(str(output_path))
    run(cmd)


def mask_token(cmd: list[str], token: str | None) -> str:
    return " ".join("***" if token and part == token else part for part in cmd)


def main() -> int:
    env_config.load_env(SCRIPT_DIR / ".env")

    default_output_dir = env_config.get_str("OUTPUT_DIR", required=True)
    default_audio_backup_dir = env_config.get_str("AUDIO_BACKUP_DIR", required=True)
    default_gpu = env_config.get_str("GPU_INDEX", default="0")
    default_token_file = env_config.get_str("HF_TOKEN_FILE")

    parser = argparse.ArgumentParser(description="Transcribe audio with WhisperX + GPU diarization.")
    parser.add_argument("audio", help="Path to the input audio")
    parser.add_argument("--output-dir", default=default_output_dir, help="Output directory for transcripts")
    parser.add_argument(
        "--audio-backup-dir",
        default=default_audio_backup_dir,
        help="Directory to store treated audio (16 kHz mono WAV)",
    )
    parser.add_argument("--model", default="large-v3", help="Whisper model. Default: large-v3")
    parser.add_argument("--language", default="pt", help="Language code. Default: pt")
    parser.add_argument("--gpu", default=default_gpu, help="Physical GPU index via CUDA_VISIBLE_DEVICES")
    parser.add_argument("--batch-size", default="8", help="Batch size. Default: 8")
    parser.add_argument("--compute-type", default="float16", help="Compute type. Default: float16")
    parser.add_argument("--beam-size", default="5", help="Whisper beam size. Use 1 for speed")
    parser.add_argument("--no-align", action="store_true", help="Skip phoneme/word alignment (faster)")
    parser.add_argument("--vad-method", choices=["pyannote", "silero"], default=None, help="VAD method")
    parser.add_argument("--vad-onset", default=None, help="VAD onset threshold")
    parser.add_argument("--vad-offset", default=None, help="VAD offset threshold")
    parser.add_argument("--chunk-size", default=None, help="VAD chunk size")
    parser.add_argument("--min-speakers", default=None, help="Minimum speakers for diarization")
    parser.add_argument("--max-speakers", default=None, help="Maximum speakers for diarization")
    parser.add_argument("--keep-silence", action="store_true", help="Do not strip silence in the treated audio")
    parser.add_argument("--silence-threshold", default=DEFAULT_SILENCE_THRESHOLD, help=f"Silence threshold. Default: {DEFAULT_SILENCE_THRESHOLD}")
    parser.add_argument("--silence-duration", type=float, default=DEFAULT_SILENCE_DURATION, help=f"Minimum silence duration to cut. Default: {DEFAULT_SILENCE_DURATION}")
    parser.add_argument("--keep-silence-seconds", type=float, default=DEFAULT_KEEP_SILENCE_SECONDS, help=f"Silence kept around cuts. Default: {DEFAULT_KEEP_SILENCE_SECONDS}")
    parser.add_argument("--limit-seconds", type=float, default=None, help="Limit treated audio length (for testing)")
    parser.add_argument("--force-audio-treatment", action="store_true", help="Recreate the treated WAV even if it already exists")
    parser.add_argument("--keep-all-outputs", action="store_true", help="Keep TXT/SRT/VTT/TSV/raw JSON alongside speakers.json")
    parser.add_argument("--no-diarize", action="store_true", help="Disable diarization")
    parser.add_argument("--token-file", default=default_token_file, help="File containing the Hugging Face token")
    args = parser.parse_args()

    audio_path = Path(args.audio).expanduser().resolve()
    if not audio_path.exists():
        print(f"error: audio not found: {audio_path}", file=sys.stderr)
        return 2

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_backup_dir = Path(args.audio_backup_dir).expanduser().resolve()
    audio_backup_dir.mkdir(parents=True, exist_ok=True)

    treated_audio_path = treated_audio_path_for(audio_path, audio_backup_dir)
    if args.force_audio_treatment and treated_audio_path.exists():
        treated_audio_path = unique_path(treated_audio_path)

    if treated_audio_path.exists() and not args.force_audio_treatment and args.limit_seconds is None:
        print(f"Reusing treated audio from backup: {treated_audio_path}", flush=True)
    else:
        print(f"Treating audio with ffmpeg: {audio_path}", flush=True)
        treat_audio(
            audio_path,
            treated_audio_path,
            trim_silence=not args.keep_silence,
            silence_threshold=args.silence_threshold,
            silence_duration=args.silence_duration,
            keep_silence_seconds=args.keep_silence_seconds,
            limit_seconds=args.limit_seconds,
        )
        print(f"Treated audio saved to backup: {treated_audio_path}", flush=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu

    whisperx_bin = resolve_whisperx_binary()

    cmd = [
        str(whisperx_bin),
        str(treated_audio_path),
        "--model",
        args.model,
        "--language",
        args.language,
        "--device",
        "cuda",
        "--device_index",
        "0",
        "--compute_type",
        args.compute_type,
        "--batch_size",
        args.batch_size,
        "--beam_size",
        args.beam_size,
        "--output_dir",
        str(output_dir),
        "--output_format",
        "all" if args.keep_all_outputs else "json",
        "--print_progress",
        "True",
    ]
    if args.no_align:
        cmd.append("--no_align")
    if args.vad_method:
        cmd += ["--vad_method", args.vad_method]
    if args.vad_onset:
        cmd += ["--vad_onset", args.vad_onset]
    if args.vad_offset:
        cmd += ["--vad_offset", args.vad_offset]
    if args.chunk_size:
        cmd += ["--chunk_size", args.chunk_size]

    token = None
    if not args.no_diarize:
        token = resolve_hf_token(args.token_file)
        cmd += ["--diarize", "--hf_token", token]
        if args.min_speakers:
            cmd += ["--min_speakers", args.min_speakers]
        if args.max_speakers:
            cmd += ["--max_speakers", args.max_speakers]

    print("Running WhisperX:", mask_token(cmd, token), flush=True)
    run(cmd, env)

    stem = stem_without_audio_suffix(treated_audio_path)
    json_path = output_dir / f"{stem}.json"
    if not args.no_diarize and json_path.exists():
        normalized_json = output_dir / f"{stem}.speakers.json"
        normalized_txt = output_dir / f"{stem}.speakers.txt" if args.keep_all_outputs else None
        write_normalized_outputs(json_path, normalized_json, normalized_txt)
        if not args.keep_all_outputs:
            remove_extra_outputs(output_dir, stem, keep={normalized_json})
        elif normalized_txt:
            print(f"Normalized transcript saved to: {normalized_txt}")
        print(f"Normalized JSON saved to: {normalized_json}")
    else:
        print(f"JSON saved to: {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
