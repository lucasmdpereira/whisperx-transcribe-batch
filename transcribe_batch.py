#!/usr/bin/env python3
"""On-demand batch orchestrator: pulls audio from a remote SFTP, transcribes, evaluates, writes metadata."""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import socket
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import paramiko

import env_config
from transcribe import stem_without_audio_suffix

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = SCRIPT_DIR / ".env"
DEFAULT_MIN_AGE_SECONDS = 60
DEFAULT_GPU_WAIT_TIMEOUT = 1800
DEFAULT_GPU_POLL_INTERVAL = 30

LOCK_GLOBAL = Path("/tmp/transcribe_batch.lock")
LOCK_PER_FILE_FMT = "/tmp/transcribe_batch_{key}.lock"

MANIFEST_FILE = "_manifest.json"
ORCHESTRATOR_LOG = "_orchestrator.log"
FAILURE_LOG = "_quality_failures.log"

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mka", ".webm", ".ogg", ".opus", ".flac", ".aac"}

DEFAULT_THRESHOLDS = {
    "min_avg_logprob": -0.7,
    "max_low_conf_fraction": 0.2,
    "min_coverage": 0.05,
    "min_speech_seconds": 60.0,
    "min_segments": 5,
    "min_speakers": 1,
    "max_empty_text_fraction": 0.2,
}

DEFAULT_MAX_ATTEMPTS = 3


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


@contextmanager
def flock_nonblocking(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            yield False
            return
        try:
            yield True
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
    except Exception:
        os.close(fd)
        raise


def identity_key(filename: str, size: int, mtime: float | int) -> str:
    raw = f"{filename}|{int(size)}|{int(mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# GPU gate
# ---------------------------------------------------------------------------

def _read_pgid(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def gpu_foreign_pids(gpu_index: int, own_pgid: int) -> list[dict]:
    """Return processes on the given GPU whose PGID is not our own."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        log_event(action="gpu_query_failed", error=str(exc))
        return []
    if result.returncode != 0:
        log_event(action="gpu_query_failed", rc=result.returncode, stderr=result.stderr.strip())
        return []
    foreign: list[dict] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 1 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        pgid = _read_pgid(pid)
        if pgid == own_pgid:
            continue
        foreign.append({
            "pid": pid,
            "pgid": pgid,
            "name": parts[1] if len(parts) > 1 else "",
            "mem_mib": parts[2] if len(parts) > 2 else "",
        })
    return foreign


def wait_for_gpu(gpu_index: int, own_pgid: int, timeout_s: int, poll_s: int = DEFAULT_GPU_POLL_INTERVAL) -> bool:
    deadline = time.monotonic() + max(0, timeout_s)
    notified = False
    while True:
        foreign = gpu_foreign_pids(gpu_index, own_pgid)
        if not foreign:
            return True
        if not notified:
            log_event(action="gpu_busy_waiting", gpu=gpu_index, foreign=foreign, timeout_s=timeout_s)
            notified = True
        if time.monotonic() >= deadline:
            log_event(action="gpu_busy_timeout", gpu=gpu_index, foreign=foreign)
            return False
        time.sleep(poll_s)


# ---------------------------------------------------------------------------
# SFTP
# ---------------------------------------------------------------------------

@dataclass
class RemoteFile:
    filename: str
    remote_path: str
    size: int
    mtime: int

    @property
    def stem(self) -> str:
        return stem_without_audio_suffix(Path(self.filename))


class SftpClient:
    def __init__(self, host: str, user: str, password: str, port: int = 22):
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self._transport: paramiko.Transport | None = None
        self._sftp: paramiko.SFTPClient | None = None

    def __enter__(self) -> "SftpClient":
        addr = (self.host, self.port)
        self._transport = paramiko.Transport(addr)
        # Disable host key checking — server uses password auth and the operator
        # accepted the host key out-of-band when issuing credentials.
        self._transport.connect(username=self.user, password=self.password)
        self._sftp = paramiko.SFTPClient.from_transport(self._transport)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sftp is not None:
                self._sftp.close()
        finally:
            if self._transport is not None:
                self._transport.close()

    @property
    def sftp(self) -> paramiko.SFTPClient:
        if self._sftp is None:
            raise RuntimeError("SftpClient not opened")
        return self._sftp

    def list_remote_audios(self, remote_dir: str, min_age_seconds: int) -> list[RemoteFile]:
        threshold = time.time() - min_age_seconds
        entries = self.sftp.listdir_attr(remote_dir)
        out: list[RemoteFile] = []
        for attr in entries:
            name = attr.filename
            if not name or name.startswith("."):
                continue
            if stat.S_ISDIR(attr.st_mode or 0):
                continue
            ext = Path(name).suffix.lower()
            if ext not in AUDIO_EXTS:
                continue
            if attr.st_mtime is None or attr.st_size is None:
                continue
            if attr.st_mtime > threshold:
                continue
            out.append(
                RemoteFile(
                    filename=name,
                    remote_path=f"{remote_dir.rstrip('/')}/{name}",
                    size=int(attr.st_size),
                    mtime=int(attr.st_mtime),
                )
            )
        out.sort(key=lambda r: r.mtime)
        return out

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        self.sftp.get(remote_path, str(tmp))
        os.replace(tmp, local_path)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def load_manifest(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        log_event(action="manifest_corrupt", path=str(path), error=str(exc))
        return {}
    # Migracao: entradas antigas nao tem 'attempts'. Conta tentativa previa (passou ou falhou)
    # para respeitar a politica de "3 strikes totais".
    for entry in manifest.values():
        if "attempts" not in entry:
            entry["attempts"] = 1 if entry.get("last_processed_iso") else 0
        entry.setdefault("abandoned", False)
    return manifest


def save_manifest(path: Path, manifest: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_manifest_from_existing(
    manifest: dict[str, dict],
    remote_files: list[RemoteFile],
    transcripts_dir: Path,
    thresholds: dict,
) -> int:
    """For each remote file whose stem already has a local speakers.json, seed a passing entry.

    This avoids reprocessing audios that were transcribed before this orchestrator existed.
    """
    seeded = 0
    by_stem = {rf.stem: rf for rf in remote_files}
    for stem, rf in by_stem.items():
        key = identity_key(rf.filename, rf.size, rf.mtime)
        if key in manifest:
            continue
        speakers_json = transcripts_dir / f"{stem}.speakers.json"
        if not speakers_json.exists():
            continue
        metrics = evaluate_quality(speakers_json, thresholds)
        manifest[key] = {
            "identity_key": key,
            "remote_path": rf.remote_path,
            "filename": rf.filename,
            "size": rf.size,
            "remote_mtime": rf.mtime,
            "stem": stem,
            "speakers_json": str(speakers_json),
            "metadata_json": None,
            "quality_passed": bool(metrics["quality_passed"]),
            "score": metrics["score"],
            "attempts": 1,
            "abandoned": False,
            "first_seen_iso": now_iso(),
            "last_processed_iso": None,
            "seeded": True,
        }
        seeded += 1
    return seeded


# ---------------------------------------------------------------------------
# Quality evaluation
# ---------------------------------------------------------------------------

def _norm(x: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return max(0.0, min(1.0, (x - lo) / (hi - lo)))


def evaluate_quality(speakers_json_path: Path, thresholds: dict) -> dict:
    """Read the normalized speakers.json and compute aggregate quality metrics.

    The WhisperX output preserves `avg_logprob` per segment but not `no_speech_prob` /
    `compression_ratio` / `temperature`, so the score formula uses only fields that
    actually survive in /mnt/backup/transcricoes/<stem>.speakers.json.
    """
    if not speakers_json_path.exists():
        return {
            "quality_passed": False,
            "score": 0.0,
            "failure_reason": f"speakers.json ausente: {speakers_json_path}",
            "metrics": {},
        }
    try:
        data = json.loads(speakers_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return {
            "quality_passed": False,
            "score": 0.0,
            "failure_reason": f"speakers.json invalido: {exc}",
            "metrics": {},
        }

    segments = data.get("segments", []) or []
    n_segments = len(segments)
    if n_segments == 0:
        return {
            "quality_passed": False,
            "score": 0.0,
            "failure_reason": "nenhum segmento na transcricao",
            "metrics": {"n_segments": 0},
        }

    audio_duration_s = 0.0
    speech_seconds = 0.0
    logprobs: list[float] = []
    low_conf_count = 0
    empty_count = 0
    speakers: set[str] = set()
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        audio_duration_s = max(audio_duration_s, end)
        speech_seconds += max(0.0, end - start)
        lp = seg.get("avg_logprob")
        if isinstance(lp, (int, float)):
            logprobs.append(float(lp))
            if lp < -1.0:
                low_conf_count += 1
        text = (seg.get("text") or "").strip()
        if len(text) < 2:
            empty_count += 1
        spk = seg.get("speaker")
        if spk:
            speakers.add(str(spk))

    coverage_ratio = (speech_seconds / audio_duration_s) if audio_duration_s > 0 else 0.0
    coverage_ratio = max(0.0, min(1.0, coverage_ratio))
    mean_logprob = sum(logprobs) / len(logprobs) if logprobs else float("-inf")
    low_conf_fraction = (low_conf_count / n_segments) if n_segments else 1.0
    empty_text_fraction = (empty_count / n_segments) if n_segments else 1.0
    unique_speakers = len(speakers)

    metrics = {
        "n_segments": n_segments,
        "audio_duration_s": round(audio_duration_s, 3),
        "speech_seconds": round(speech_seconds, 3),
        "coverage_ratio": round(coverage_ratio, 4),
        "mean_avg_logprob": round(mean_logprob, 4) if logprobs else None,
        "low_conf_fraction": round(low_conf_fraction, 4),
        "empty_text_fraction": round(empty_text_fraction, 4),
        "unique_speakers": unique_speakers,
        "logprob_segments_counted": len(logprobs),
    }

    reasons: list[str] = []
    if not logprobs:
        reasons.append("avg_logprob ausente em todos os segmentos")
    if logprobs and mean_logprob < thresholds["min_avg_logprob"]:
        reasons.append(f"mean_avg_logprob {mean_logprob:.3f} < {thresholds['min_avg_logprob']}")
    if low_conf_fraction > thresholds["max_low_conf_fraction"]:
        reasons.append(f"low_conf_fraction {low_conf_fraction:.3f} > {thresholds['max_low_conf_fraction']}")
    if coverage_ratio < thresholds["min_coverage"]:
        reasons.append(f"coverage_ratio {coverage_ratio:.3f} < {thresholds['min_coverage']}")
    if speech_seconds < thresholds["min_speech_seconds"]:
        reasons.append(f"speech_seconds {speech_seconds:.1f} < {thresholds['min_speech_seconds']}")
    if n_segments < thresholds["min_segments"]:
        reasons.append(f"n_segments {n_segments} < {thresholds['min_segments']}")
    if unique_speakers < thresholds["min_speakers"]:
        reasons.append(f"unique_speakers {unique_speakers} < {thresholds['min_speakers']}")
    if empty_text_fraction > thresholds["max_empty_text_fraction"]:
        reasons.append(f"empty_text_fraction {empty_text_fraction:.3f} > {thresholds['max_empty_text_fraction']}")

    quality_passed = not reasons

    if logprobs:
        score = (
            0.35 * _norm(mean_logprob, -1.5, -0.3)
            + 0.25 * (1 - _norm(low_conf_fraction, 0.0, 0.5))
            + 0.20 * _norm(coverage_ratio, 0.2, 0.8)
            + 0.10 * (1 - _norm(empty_text_fraction, 0.0, 0.4))
            + 0.05 * _norm(n_segments, 1, 20)
            + 0.05 * _norm(unique_speakers, 1, 3)
        )
    else:
        score = 0.0

    return {
        "quality_passed": quality_passed,
        "score": round(score, 4),
        "failure_reason": "; ".join(reasons) if reasons else None,
        "metrics": metrics,
    }


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_path: Path | None = None


def configure_log(path: Path) -> None:
    global _log_path
    _log_path = path
    path.parent.mkdir(parents=True, exist_ok=True)


def log_event(**kv) -> None:
    parts = [now_iso()]
    for k, v in kv.items():
        if isinstance(v, (dict, list)):
            v_str = json.dumps(v, ensure_ascii=False, separators=(",", ":"))
        else:
            v_str = str(v)
        if any(c.isspace() for c in v_str) or "=" in v_str:
            v_str = shlex.quote(v_str)
        parts.append(f"{k}={v_str}")
    line = " ".join(parts)
    print(line, flush=True)
    if _log_path is not None:
        try:
            with _log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def append_failure_log(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(
        {
            "ts": now_iso(),
            "identity_key": payload.get("identity_key"),
            "filename": payload.get("source_remote_path"),
            "score": payload.get("score"),
            "failure_reason": payload.get("failure_reason"),
        },
        ensure_ascii=False,
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_whisperx_subprocess(local_audio: Path, gpu: int, passthrough: list[str]) -> int:
    wrapper = SCRIPT_DIR / "transcribe"
    cmd = [str(wrapper), str(local_audio), "--gpu", str(gpu), *passthrough]
    log_event(action="whisperx_start", file=local_audio.name, cmd=" ".join(cmd))
    env = os.environ.copy()
    # Ensure the wrapper sees the requested GPU; the wrapper also sets this itself
    # but we set it here so the GPU gate seen via os.setpgrp() inheritance is consistent.
    env.setdefault("CUDA_VISIBLE_DEVICES", str(gpu))
    proc = subprocess.run(cmd, env=env, check=False)
    log_event(action="whisperx_done", file=local_audio.name, rc=proc.returncode)
    return proc.returncode


def build_metadata(
    rf: RemoteFile,
    key: str,
    stem: str,
    speakers_json: Path,
    treated_audio: Path,
    wall_seconds: float,
    start_iso: str,
    end_iso: str,
    args,
    thresholds: dict,
    quality: dict,
) -> dict:
    return {
        "schema_version": 1,
        "source_remote_host": args.remote_host,
        "source_remote_path": rf.remote_path,
        "source_filename": rf.filename,
        "source_size_bytes": rf.size,
        "source_remote_mtime_iso": _dt.datetime.fromtimestamp(rf.mtime, _dt.timezone.utc).isoformat(timespec="seconds"),
        "identity_key": key,
        "treated_audio": str(treated_audio),
        "transcription": str(speakers_json),
        "processing": {
            "start_iso": start_iso,
            "end_iso": end_iso,
            "wall_seconds": round(wall_seconds, 3),
            "gpu_index": args.gpu,
        },
        "thresholds": thresholds,
        "metrics": quality["metrics"],
        "score": quality["score"],
        "quality_passed": quality["quality_passed"],
        "failure_reason": quality["failure_reason"],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str]) -> argparse.Namespace:
    env_file_default = str(DEFAULT_ENV_FILE)
    # Resolve env-driven defaults early so --help shows current values.
    env_config.load_env(DEFAULT_ENV_FILE)
    default_remote_host = env_config.get_str("SFTP_HOST", default="")
    default_remote_port = env_config.get_int("SFTP_PORT", default=22)
    default_remote_dir = env_config.get_str("SFTP_REMOTE_DIR", default="/")
    default_cache_dir = env_config.get_str("CACHE_DIR", required=True)
    default_transcripts_dir = env_config.get_str("OUTPUT_DIR", required=True)
    default_audios_dir = env_config.get_str("AUDIO_BACKUP_DIR", required=True)
    default_gpu = env_config.get_int("GPU_INDEX", default=0)

    parser = argparse.ArgumentParser(
        description="On-demand orchestrator for remote audio transcription with WhisperX.",
        epilog="Arguments after '--' are forwarded to the transcribe wrapper.",
    )
    parser.add_argument("--env-file", default=env_file_default, help=f".env file. Default: {DEFAULT_ENV_FILE}")
    parser.add_argument("--remote-host", default=default_remote_host)
    parser.add_argument("--remote-port", type=int, default=default_remote_port)
    parser.add_argument("--remote-dir", default=default_remote_dir)
    parser.add_argument("--cache-dir", default=default_cache_dir)
    parser.add_argument("--transcripts-dir", "--transcricoes-dir", dest="transcripts_dir", default=default_transcripts_dir)
    parser.add_argument("--audios-dir", default=default_audios_dir)
    parser.add_argument("--gpu", type=int, default=default_gpu)
    parser.add_argument("--max-files-per-run", type=int, default=0, help="0 = no limit")
    parser.add_argument("--min-age-seconds", type=int, default=DEFAULT_MIN_AGE_SECONDS)
    parser.add_argument("--gpu-wait-timeout", type=int, default=DEFAULT_GPU_WAIT_TIMEOUT)
    parser.add_argument("--no-wait", action="store_true", help="Abort if the GPU is busy instead of waiting")
    parser.add_argument("--skip-failed", action="store_true", help="Skip every previously failed file (regardless of attempts).")
    parser.add_argument("--max-attempts", type=int, default=DEFAULT_MAX_ATTEMPTS, help=f"Maximum attempts per file before abandoning. Default: {DEFAULT_MAX_ATTEMPTS}.")
    parser.add_argument("--retry-abandoned", action="store_true", help="Retry files already marked as abandoned (attempts >= max-attempts).")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--purge-cache-after-success", action="store_true")
    parser.add_argument("--min-avg-logprob", type=float, default=DEFAULT_THRESHOLDS["min_avg_logprob"])
    parser.add_argument("--max-low-conf-fraction", type=float, default=DEFAULT_THRESHOLDS["max_low_conf_fraction"])
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_THRESHOLDS["min_coverage"])
    parser.add_argument("--min-speech-seconds", type=float, default=DEFAULT_THRESHOLDS["min_speech_seconds"])
    parser.add_argument("--min-segments", type=int, default=DEFAULT_THRESHOLDS["min_segments"])
    parser.add_argument("--min-speakers", type=int, default=DEFAULT_THRESHOLDS["min_speakers"])
    parser.add_argument("--max-empty-text-fraction", type=float, default=DEFAULT_THRESHOLDS["max_empty_text_fraction"])
    parser.add_argument(
        "--reevaluate-only",
        action="store_true",
        help="Only re-evaluate existing .speakers.json and update manifest/metadata. Does not download or run WhisperX.",
    )
    parser.add_argument("passthrough", nargs=argparse.REMAINDER, help="Arguments forwarded to the transcribe wrapper (after '--')")
    return parser.parse_args(argv)


def thresholds_from(args) -> dict:
    return {
        "min_avg_logprob": args.min_avg_logprob,
        "max_low_conf_fraction": args.max_low_conf_fraction,
        "min_coverage": args.min_coverage,
        "min_speech_seconds": args.min_speech_seconds,
        "min_segments": args.min_segments,
        "min_speakers": args.min_speakers,
        "max_empty_text_fraction": args.max_empty_text_fraction,
    }


def reevaluate_existing(
    transcripts_dir: Path,
    audios_dir: Path,
    thresholds: dict,
    args,
) -> tuple[int, int]:
    """Re-roda evaluate_quality em todos os .speakers.json e atualiza manifest + metadata.

    Nao baixa nem transcreve nada. Reusa o manifest se existir; caso contrario indexa por stem.
    """
    manifest_path = transcripts_dir / MANIFEST_FILE
    manifest = load_manifest(manifest_path)
    # Indice reverso: stem -> identity_key (para encontrar entradas no manifest)
    by_stem: dict[str, str] = {k: e.get("stem") for k, e in manifest.items() if e.get("stem")}
    stem_to_key: dict[str, str] = {v: k for k, v in by_stem.items()}

    passed = failed = 0
    for sj in sorted(transcripts_dir.glob("*.speakers.json")):
        stem = sj.stem  # "<stem>.speakers" -> queremos remover o ".speakers"
        if stem.endswith(".speakers"):
            stem = stem[: -len(".speakers")]
        quality = evaluate_quality(sj, thresholds)
        meta_path = transcripts_dir / f"{stem}.metadata.json"

        # Reusa metadata existente se houver, sobrescrevendo so os campos de qualidade.
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        payload.setdefault("schema_version", 1)
        payload.setdefault("transcription", str(sj))
        payload.setdefault("treated_audio", str(audios_dir / f"{stem}.wav"))
        payload["thresholds"] = thresholds
        payload["metrics"] = quality["metrics"]
        payload["score"] = quality["score"]
        payload["quality_passed"] = quality["quality_passed"]
        payload["failure_reason"] = quality["failure_reason"]
        payload["reevaluated_at"] = now_iso()

        tmp = meta_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, meta_path)

        # Atualiza manifest se houver chave conhecida; senao deixa o seed da rodada normal cuidar.
        key = stem_to_key.get(stem)
        if key and key in manifest:
            manifest[key]["quality_passed"] = bool(quality["quality_passed"])
            manifest[key]["score"] = quality["score"]
            manifest[key]["last_processed_iso"] = now_iso()
            manifest[key]["metadata_json"] = str(meta_path)

        if quality["quality_passed"]:
            passed += 1
        else:
            failed += 1
            append_failure_log(transcripts_dir / FAILURE_LOG, {
                "identity_key": key,
                "source_remote_path": payload.get("source_remote_path") or stem,
                "score": quality["score"],
                "failure_reason": quality["failure_reason"],
            })
        log_event(
            action="reevaluated",
            stem=stem,
            passed=quality["quality_passed"],
            score=quality["score"],
            reason=quality["failure_reason"] or "",
        )

    save_manifest(manifest_path, manifest)
    return passed, failed


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])

    # Strip leading "--" from passthrough (argparse REMAINDER keeps it)
    passthrough = list(args.passthrough or [])
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    transcripts_dir = Path(args.transcripts_dir).expanduser().resolve()
    audios_dir = Path(args.audios_dir).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    transcripts_dir.mkdir(parents=True, exist_ok=True)
    audios_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    configure_log(transcripts_dir / ORCHESTRATOR_LOG)

    # Make this process a process-group leader so the GPU gate can identify our subprocesses.
    try:
        os.setpgrp()
    except OSError as exc:
        log_event(action="setpgrp_failed", error=str(exc))

    own_pgid = os.getpid()

    env_config.load_env(Path(args.env_file))
    sftp_user = env_config.get_str("SFTP_USER")
    sftp_pass = env_config.get_str("SFTP_PASS")
    if not sftp_user or not sftp_pass:
        log_event(action="env_missing", env_file=args.env_file)
        print(f"error: SFTP_USER/SFTP_PASS not found in {args.env_file}", file=sys.stderr)
        return 2
    if not args.remote_host:
        log_event(action="env_missing", key="SFTP_HOST")
        print("error: SFTP_HOST not configured (set in .env or pass --remote-host)", file=sys.stderr)
        return 2

    thresholds = thresholds_from(args)

    with flock_nonblocking(LOCK_GLOBAL) as got_global:
        if not got_global:
            log_event(action="global_lock_busy")
            print("Another transcribe_batch run is already in progress.", file=sys.stderr)
            return 1

        if args.reevaluate_only:
            log_event(action="reevaluate_only_start", thresholds=thresholds)
            passed, failed = reevaluate_existing(transcripts_dir, audios_dir, thresholds, args)
            log_event(action="reevaluate_only_done", passed=passed, failed=failed)
            return 0

        # Initial GPU gate
        if gpu_foreign_pids(args.gpu, own_pgid):
            if args.no_wait:
                log_event(action="gpu_busy_abort", gpu=args.gpu)
                return 0
            if not wait_for_gpu(args.gpu, own_pgid, args.gpu_wait_timeout):
                return 0

        # SFTP listing
        try:
            log_event(action="sftp_connect", host=args.remote_host, user=sftp_user)
            with SftpClient(args.remote_host, sftp_user, sftp_pass, port=args.remote_port) as client:
                remote_files = client.list_remote_audios(args.remote_dir, args.min_age_seconds)
                log_event(action="sftp_listed", count=len(remote_files), dir=args.remote_dir)

                manifest_path = transcripts_dir / MANIFEST_FILE
                manifest = load_manifest(manifest_path)
                seeded = seed_manifest_from_existing(manifest, remote_files, transcripts_dir, thresholds)
                if seeded:
                    save_manifest(manifest_path, manifest)
                    log_event(action="manifest_seeded", count=seeded)

                pending: list[RemoteFile] = []
                abandoned_count = 0
                for rf in remote_files:
                    key = identity_key(rf.filename, rf.size, rf.mtime)
                    entry = manifest.get(key)
                    if entry and entry.get("quality_passed"):
                        continue
                    if entry and args.skip_failed:
                        continue
                    if entry and not args.retry_abandoned:
                        attempts = int(entry.get("attempts", 0))
                        if attempts >= args.max_attempts:
                            abandoned_count += 1
                            if not entry.get("abandoned"):
                                entry["abandoned"] = True
                            continue
                    pending.append(rf)

                if abandoned_count:
                    save_manifest(manifest_path, manifest)
                    log_event(action="abandoned_skipped", count=abandoned_count, max_attempts=args.max_attempts)

                limit = args.max_files_per_run if args.max_files_per_run and args.max_files_per_run > 0 else None
                if limit is not None:
                    pending = pending[:limit]

                log_event(action="run_plan", pending=len(pending), total_remote=len(remote_files), dry_run=args.dry_run)

                if args.dry_run:
                    for rf in pending:
                        log_event(action="would_process", file=rf.filename, size=rf.size, mtime=rf.mtime)
                    log_event(action="run_complete_dry_run", processed=0, pending=len(pending))
                    return 0

                processed = 0
                failed = 0
                for rf in pending:
                    key = identity_key(rf.filename, rf.size, rf.mtime)

                    # Re-check GPU between files
                    if gpu_foreign_pids(args.gpu, own_pgid):
                        if args.no_wait:
                            log_event(action="gpu_busy_abort_mid_run", gpu=args.gpu, remaining=len(pending) - processed - failed)
                            break
                        if not wait_for_gpu(args.gpu, own_pgid, args.gpu_wait_timeout):
                            log_event(action="gpu_busy_timeout_mid_run", remaining=len(pending) - processed - failed)
                            break

                    file_lock_path = Path(LOCK_PER_FILE_FMT.format(key=key))
                    with flock_nonblocking(file_lock_path) as got_file_lock:
                        if not got_file_lock:
                            log_event(action="file_lock_busy", file=rf.filename, key=key)
                            continue

                        local_audio = cache_dir / rf.filename
                        if not local_audio.exists():
                            log_event(action="download_start", file=rf.filename, size=rf.size)
                            try:
                                client.download(rf.remote_path, local_audio)
                            except Exception as exc:
                                log_event(action="download_failed", file=rf.filename, error=str(exc))
                                failed += 1
                                continue
                            log_event(action="download_done", file=rf.filename, local=str(local_audio))
                        else:
                            log_event(action="download_skipped_existing_cache", file=rf.filename, local=str(local_audio))

                        stem = stem_without_audio_suffix(local_audio)
                        speakers_json = transcripts_dir / f"{stem}.speakers.json"
                        treated_audio = audios_dir / f"{stem}.wav"

                        start_iso = now_iso()
                        t0 = time.monotonic()
                        rc = run_whisperx_subprocess(local_audio, args.gpu, passthrough)
                        wall = time.monotonic() - t0
                        end_iso = now_iso()

                        if rc != 0 or not speakers_json.exists():
                            quality = {
                                "quality_passed": False,
                                "score": 0.0,
                                "failure_reason": f"whisperx falhou (rc={rc})" if rc != 0 else "speakers.json nao foi gerado",
                                "metrics": {},
                            }
                        else:
                            quality = evaluate_quality(speakers_json, thresholds)

                        metadata = build_metadata(
                            rf, key, stem, speakers_json, treated_audio,
                            wall, start_iso, end_iso, args, thresholds, quality,
                        )
                        meta_path = transcripts_dir / f"{stem}.metadata.json"
                        meta_tmp = meta_path.with_suffix(".json.tmp")
                        meta_tmp.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        os.replace(meta_tmp, meta_path)

                        prev_attempts = int(manifest.get(key, {}).get("attempts", 0))
                        new_attempts = prev_attempts + 1
                        abandoned = (not quality["quality_passed"]) and (new_attempts >= args.max_attempts)
                        manifest[key] = {
                            "identity_key": key,
                            "remote_path": rf.remote_path,
                            "filename": rf.filename,
                            "size": rf.size,
                            "remote_mtime": rf.mtime,
                            "stem": stem,
                            "speakers_json": str(speakers_json) if speakers_json.exists() else None,
                            "metadata_json": str(meta_path),
                            "quality_passed": bool(quality["quality_passed"]),
                            "score": quality["score"],
                            "attempts": new_attempts,
                            "abandoned": abandoned,
                            "first_seen_iso": (manifest.get(key, {}).get("first_seen_iso") or now_iso()),
                            "last_processed_iso": now_iso(),
                            "seeded": False,
                        }
                        save_manifest(manifest_path, manifest)

                        if quality["quality_passed"]:
                            processed += 1
                            log_event(action="file_done", file=rf.filename, score=quality["score"], wall_s=round(wall, 1))
                            if args.purge_cache_after_success:
                                try:
                                    local_audio.unlink()
                                    log_event(action="cache_purged", file=rf.filename)
                                except OSError as exc:
                                    log_event(action="cache_purge_failed", file=rf.filename, error=str(exc))
                        else:
                            failed += 1
                            metadata["attempts"] = new_attempts
                            metadata["abandoned"] = abandoned
                            append_failure_log(transcripts_dir / FAILURE_LOG, metadata)
                            log_event(
                                action=("file_abandoned" if abandoned else "file_failed_quality"),
                                file=rf.filename,
                                attempts=new_attempts,
                                max_attempts=args.max_attempts,
                                reason=quality["failure_reason"],
                                score=quality["score"],
                            )

                log_event(action="run_complete", processed=processed, failed=failed, total_pending=len(pending))
                return 0
        except (paramiko.SSHException, socket.error) as exc:
            log_event(action="sftp_error", error=str(exc))
            print(f"SFTP error: {exc}", file=sys.stderr)
            return 3


if __name__ == "__main__":
    raise SystemExit(main())
