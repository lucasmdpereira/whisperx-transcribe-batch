# whisperx-batch

Two cooperating CLIs for transcribing audio with WhisperX + pyannote diarization:

- **`transcribe`** — process a single audio file: ffmpeg preprocessing (mono 16 kHz, voice filters, silence removal), WhisperX ASR, diarization, normalized speaker-tagged JSON output.
- **`transcribe_batch`** — pull new audios from an SFTP source on demand, run `transcribe` on each, score quality, keep a manifest, and retry/abandon according to a configurable policy. Nothing is deleted on the remote side; the orchestrator just tracks locally what has already been transcribed.

All paths, GPU index, SFTP credentials and the Hugging Face token live in a local `.env` file (gitignored). The code ships with no machine-specific defaults — you copy `.env.example`, fill it in, and run.

## Requirements

- Python 3.10+
- A CUDA-capable GPU. For the default `large-v3` model with diarization, allocate at least **12 GB of VRAM** (Whisper ASR uses ~2 GB; pyannote diarization peaks around 10 GB sequentially).
- `ffmpeg` on the system PATH.
- A Hugging Face token with access to the pyannote diarization models.
- For `transcribe_batch`: SFTP credentials to a server hosting your audio files.

## Setup

```bash
# 1. Clone and enter the repo
git clone <your-fork-url> whisperx-batch && cd whisperx-batch

# 2. Create a venv. Using uv is recommended:
uv venv
uv pip install -r requirements.txt

# 3. Copy the env template and fill in your values
cp .env.example .env
$EDITOR .env

# 4. (Optional) Put your HF token in a separate file referenced by HF_TOKEN_FILE,
#    or set HF_TOKEN directly in .env.
```

## Usage — single file

```bash
./transcribe path/to/audio.mka
```

With a known speaker count:

```bash
./transcribe path/to/audio.mka --min-speakers 2 --max-speakers 2
```

Without diarization (faster, no HF token needed):

```bash
./transcribe path/to/audio.mka --no-diarize
```

Speed-oriented profile for long recordings:

```bash
./transcribe path/to/audio.mka --model small --beam-size 1 --no-align --silence-threshold=-25dB --force-audio-treatment
```

### Outputs

The treated WAV is cached in `$AUDIO_BACKUP_DIR/<stem>.wav` and reused across runs (pass `--force-audio-treatment` to recreate it).

`$OUTPUT_DIR/<stem>.speakers.json` is the canonical output: WhisperX JSON enriched with a stable `speaker_map` (speakers labelled `speaker_000`, `speaker_001`, …). Use `--keep-all-outputs` to additionally keep the raw TXT/SRT/VTT/TSV/JSON files.

## Usage — batch mode

The batch orchestrator is manual (no cron) — you invoke it whenever you want.

```bash
./transcribe_batch                          # process everything pending
./transcribe_batch --max-files-per-run 1    # only the next pending file
./transcribe_batch --dry-run                # list what it would do
./transcribe_batch --reevaluate-only        # re-score existing transcripts
./transcribe_batch -- --beam-size 1 --no-align   # forward flags to transcribe
```

### Pipeline

1. **Global lock** on `/tmp/transcribe_batch.lock`; concurrent invocations exit immediately.
2. **GPU gate** — `nvidia-smi -i $GPU_INDEX`: if a foreign process is using the GPU, the orchestrator waits (poll every 30 s, timeout `--gpu-wait-timeout`, default 1800 s). `--no-wait` aborts instead.
3. **Remote listing** via SFTP using the credentials in `.env`. Files newer than `--min-age-seconds` (default 60 s) are skipped to avoid catching uploads in progress.
4. **Per file**: download to `$CACHE_DIR`, invoke `./transcribe`, evaluate quality, write `<stem>.metadata.json`, update the manifest.
5. **Nothing is deleted upstream.** The cache is also kept by default (`--purge-cache-after-success` to opt in).

### Quality evaluation

Reads `<stem>.speakers.json` and aggregates these metrics:

- `mean_avg_logprob` — average of per-segment Whisper log-probabilities.
- `low_conf_fraction` — fraction of segments with `avg_logprob < -1.0`.
- `coverage_ratio` — speech seconds / total audio duration.
- `speech_seconds` — absolute speech detected.
- `n_segments`, `unique_speakers`, `empty_text_fraction`.

A composite score in `[0, 1]`:

```
score = 0.35 * norm(mean_avg_logprob, -1.5, -0.3)
      + 0.25 * (1 - norm(low_conf_fraction, 0.0, 0.5))
      + 0.20 * norm(coverage_ratio, 0.2, 0.8)
      + 0.10 * (1 - norm(empty_text_fraction, 0.0, 0.4))
      + 0.05 * norm(n_segments, 1, 20)
      + 0.05 * norm(unique_speakers, 1, 3)
```

A file passes only if it meets **all** thresholds simultaneously (defaults, all CLI-overridable):

| Threshold | Default | Meaning |
|---|---|---|
| `--min-avg-logprob` | `-0.7` | minimum mean log-probability |
| `--max-low-conf-fraction` | `0.2` | max fraction of low-confidence segments |
| `--min-coverage` | `0.05` | min coverage (real recordings are sparse) |
| `--min-speech-seconds` | `60` | absolute speech floor (key gate for silent audios) |
| `--min-segments` | `5` | minimum number of segments |
| `--min-speakers` | `1` | minimum distinct speakers |
| `--max-empty-text-fraction` | `0.2` | max fraction of empty-text segments |

### Retry policy

Failures are reprocessed up to `--max-attempts` (default **3**). After the 3rd consecutive failure a file is marked `abandoned=true` in the manifest and is skipped on subsequent runs. The `.speakers.json` is preserved for inspection.

- `--max-attempts N` — adjust the cap (`1` disables retries).
- `--retry-abandoned` — ignore the mark and try once more (e.g. after changing the model).
- `--skip-failed` — skip every previously failed file regardless of attempt count.

### Files written by batch mode

| Path | Contents |
|---|---|
| `$OUTPUT_DIR/_manifest.json` | Per-file state keyed by `sha256(filename\|size\|mtime)[:16]`; tracks `attempts`, `abandoned`, `quality_passed`, `score`, timestamps. |
| `$OUTPUT_DIR/<stem>.metadata.json` | Per-file metrics + thresholds echo + processing timings. |
| `$OUTPUT_DIR/_orchestrator.log` | Structured run log. |
| `$OUTPUT_DIR/_quality_failures.log` | One JSON line per failed audio. |
| `$CACHE_DIR/<filename>` | Local copy of the downloaded raw audio. |

## Configuration reference

Resolution order: CLI flag > `os.environ` > `.env` file in the script directory > built-in default.

| Env var | Used by | Required | Description |
|---|---|---|---|
| `OUTPUT_DIR` | both | yes | Directory for transcripts (`.speakers.json`, manifest, metadata, logs). |
| `AUDIO_BACKUP_DIR` | both | yes | Directory for ffmpeg-treated WAVs (cached across runs). |
| `CACHE_DIR` | batch | yes | Local cache for files downloaded from SFTP. |
| `GPU_INDEX` | both | no | Physical GPU index. Default `0`. |
| `HF_TOKEN` | transcribe | one of | Hugging Face token (literal). |
| `HF_TOKEN_FILE` | transcribe | one of | Path to a file containing the HF token. |
| `WHISPERX_BIN` | transcribe | no | Override the whisperx binary path. Otherwise tries `.venv/bin/whisperx` then `$PATH`. |
| `SFTP_HOST` | batch | yes | SFTP server hostname. |
| `SFTP_PORT` | batch | no | Default `22`. |
| `SFTP_USER` | batch | yes | SFTP username. |
| `SFTP_PASS` | batch | yes | SFTP password. |
| `SFTP_REMOTE_DIR` | batch | yes | Absolute path on the SFTP server to list. |

The `.env` parser accepts keys case-insensitively, so legacy lowercase keys (`sftp_user`, `sftp_pass`, `hf_token`) still resolve.

## Single-file CLI options

```
--output-dir, --audio-backup-dir          override env-driven storage paths
--model, --language, --beam-size          Whisper config (default large-v3 / pt / 5)
--batch-size, --compute-type              GPU config (default 8 / float16)
--gpu                                     override $GPU_INDEX
--no-align, --no-diarize                  toggle alignment / diarization
--vad-method, --vad-onset, --vad-offset   VAD knobs
--chunk-size                              VAD chunk size
--min-speakers, --max-speakers            diarization speaker hints
--keep-silence, --silence-threshold,
  --silence-duration, --keep-silence-seconds   ffmpeg silence trimming
--limit-seconds                           cap audio length for tests
--force-audio-treatment                   recreate the treated WAV
--keep-all-outputs                        keep TXT/SRT/VTT/TSV/raw JSON too
--token-file                              override $HF_TOKEN_FILE
```

## Batch CLI options

```
Source & paths:
  --remote-host (def $SFTP_HOST), --remote-port (def $SFTP_PORT), --remote-dir (def $SFTP_REMOTE_DIR)
  --env-file, --cache-dir, --transcripts-dir, --audios-dir

Execution:
  --gpu (def $GPU_INDEX)
  --max-files-per-run 0   (0 = no limit)
  --min-age-seconds 60    (skip files mid-upload)
  --gpu-wait-timeout 1800, --no-wait
  --dry-run, --reevaluate-only, --purge-cache-after-success

Retry policy:
  --max-attempts 3, --retry-abandoned, --skip-failed

Quality thresholds:
  --min-avg-logprob, --max-low-conf-fraction, --min-coverage,
  --min-speech-seconds, --min-segments, --min-speakers,
  --max-empty-text-fraction
```

Everything after `--` is forwarded to the `transcribe` wrapper, so you can do
`./transcribe_batch -- --beam-size 1 --no-align` to override Whisper settings for the whole run.

## Files

```
transcribe              bash wrapper that exports CUDA env and execs the .py
transcribe.py           single-file transcription pipeline
transcribe_batch        bash wrapper for the orchestrator
transcribe_batch.py     SFTP/quality/manifest orchestrator
env_config.py           tiny .env loader + typed getters (no extra deps)
.env.example            template configuration
requirements.txt        paramiko + whisperx
```

## License

MIT — see `LICENSE`.
