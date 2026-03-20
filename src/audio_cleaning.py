"""
DSP-based guitar stem cleaning.

Applies a chain of signal-processing stages to a Demucs-extracted guitar WAV
to suppress bleed from other instruments, percussive transients, and
clipping/distortion artifacts before the CQT feature extraction step.

All functions operate on channel-first float32 numpy arrays: (C, N).
"""

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.signal
import librosa
import soundfile as sf


# ---------------------------------------------------------------------------
# Default configuration — each stage can be toggled and overridden per-call
# ---------------------------------------------------------------------------

DEFAULT_CLEANING_CONFIG = {
    "rms_normalize": {
        "enabled": True,
        "target_rms_db": -20.0,   # -20 dBFS broadcast reference; corrects Demucs amplitude boost
        "max_gain_db": 20.0,       # clamp: don't amplify near-silence by more than this
    },
    "bandpass": {
        "enabled": True,
        "low_hz": 70.0,       # E2 fundamental ~82 Hz; 70 gives a safety margin
        "high_hz": 6000.0,    # ~5th harmonic of high-e; raise to 8000 if too dull
        "order": 4,           # 4th-order Butterworth (~24 dB/oct rolloff)
    },
    "hpss": {
        "enabled": False,          # Off by default — diagnostic showed -5 to -7 dB guitar loss
        "kernel_size": 61,    # Median filter length in STFT frames; 31 strips guitar attack transients
        "power": 2.0,         # 2.0 = Wiener-style soft mask; inf = hard mask
        "margin": 1.0,        # >1 introduces a "no-man's-land" gap
    },
    "sat_repair": {
        "enabled": False,         # Off by default — only needed for genuinely clipped output
        "tanh_drive": 2.5,        # Soft-clip knee; raise for harder limiting
        "smooth_window_ms": 5.0,  # Envelope smoothing (0 to disable)
    },
    "spec_gate": {
        "enabled": True,
        "noise_percentile": 10.0,  # Bottom N% of frames used for noise floor
        "threshold_db": 12.0,      # was 6 — only suppress very quiet frames
        "reduction_db": 8.0,       # was 20 — max 8 dB attenuation instead of 20
    },
    "wiener": {
        "enabled": True,
        "noise_percentile": 10.0,  # Same convention as spec_gate
        "over_subtraction": 0.3,   # was 1.0 — barely subtracts noise floor
        "spectral_floor": 0.5,     # was 0.1 — minimum 50% gain retained everywhere
    },
}

_N_FFT = 2048
_HOP_LENGTH = 512


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _load_audio(path: Path):
    """Load a WAV as (C, N) float32 array plus sample rate."""
    audio, sr = sf.read(str(path), always_2d=True)  # (N, C)
    return audio.T.astype(np.float32), sr            # → (C, N)


def _save_audio(audio: np.ndarray, sr: int, path: Path) -> None:
    """Save (C, N) float32 array as PCM-16 WAV, normalising peak to ≤ 1.0."""
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak
    sf.write(str(path), audio.T, sr, subtype="PCM_16")  # sf expects (N, C)


# ---------------------------------------------------------------------------
# Stage 0: RMS normalisation
# ---------------------------------------------------------------------------

def apply_rms_normalize(
    audio: np.ndarray,
    sr: int,
    target_rms_db: float = -20.0,
    max_gain_db: float = 20.0,
) -> np.ndarray:
    """
    Normalize the Demucs stem to a consistent RMS level.
    Pure linear gain — zero spectral distortion. Directly targets the Demucs
    amplitude boost artifact without altering the frequency content seen by the model.
    """
    mono = audio.mean(axis=0)
    current_rms = np.sqrt(np.mean(mono ** 2))
    if current_rms < 1e-8:
        return audio
    target_rms = 10 ** (target_rms_db / 20.0)
    gain = target_rms / current_rms
    gain = np.clip(gain, 10 ** (-max_gain_db / 20.0), 10 ** (max_gain_db / 20.0))
    return audio * gain


# ---------------------------------------------------------------------------
# Stage 1: Guitar bandpass
# ---------------------------------------------------------------------------

def apply_guitar_bandpass(
    audio: np.ndarray,
    sr: int,
    low_hz: float = 70.0,
    high_hz: float = 6000.0,
    order: int = 4,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass to remove sub-bass rumble and ultrasonic
    content that cannot belong to a guitar.
    """
    nyq = sr / 2.0
    sos = scipy.signal.butter(
        order,
        [low_hz / nyq, high_hz / nyq],
        btype="bandpass",
        output="sos",
    )
    return np.stack([scipy.signal.sosfiltfilt(sos, ch) for ch in audio])


# ---------------------------------------------------------------------------
# Stage 2: Harmonic-Percussive Source Separation
# ---------------------------------------------------------------------------

def apply_hpss(
    audio: np.ndarray,
    sr: int,
    kernel_size: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> np.ndarray:
    """
    Keep only the harmonic component of the signal (discard percussive).
    Targets: claps, taps, drum transients bleeding through Demucs.
    """
    cleaned = []
    for ch in audio:
        harmonic, _ = librosa.effects.hpss(
            ch, kernel_size=kernel_size, power=power, margin=margin
        )
        cleaned.append(harmonic)
    return np.stack(cleaned)


# ---------------------------------------------------------------------------
# Stage 3: Soft saturation / clipping repair
# ---------------------------------------------------------------------------

def apply_soft_saturation_repair(
    audio: np.ndarray,
    sr: int,
    tanh_drive: float = 2.5,
    smooth_window_ms: float = 5.0,
) -> np.ndarray:
    """
    Apply a tanh soft-clip curve to repair hard-clipping artifacts from
    Demucs normalisation. Optionally smooth abrupt gain transitions.
    """
    peak = np.max(np.abs(audio))
    if peak > 1.0:
        audio = audio / peak

    # tanh(drive * x) / tanh(drive) maps [-1,1] → [-1,1] with a soft knee
    repaired = np.tanh(tanh_drive * audio) / np.tanh(tanh_drive)

    if smooth_window_ms > 0:
        win = max(1, int(smooth_window_ms * sr / 1000))
        kernel = np.ones(win) / win
        smoothed = np.empty_like(repaired)
        for i, (orig_ch, rep_ch) in enumerate(zip(audio, repaired)):
            orig_env = np.convolve(np.abs(orig_ch), kernel, mode="same") + 1e-8
            rep_env  = np.convolve(np.abs(rep_ch),  kernel, mode="same") + 1e-8
            ratio = np.minimum(orig_env / rep_env, 2.0)  # clamp: max 6 dB gain to prevent spikes
            smoothed[i] = rep_ch * ratio
        repaired = smoothed

    return repaired


# ---------------------------------------------------------------------------
# Stage 4: Spectral gate
# ---------------------------------------------------------------------------

def apply_spectral_gate(
    audio: np.ndarray,
    sr: int,
    noise_percentile: float = 10.0,
    threshold_db: float = 6.0,
    reduction_db: float = 20.0,
) -> np.ndarray:
    """
    Sigmoid soft-mask that suppresses frequency bins close to the noise floor.
    Noise floor is estimated from the quietest `noise_percentile`% of frames.
    Targets: quiet sustained bleed (voice, drum room, background instruments).
    """
    cleaned = []
    for ch in audio:
        D = librosa.stft(ch, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        mag, phase = np.abs(D), np.angle(D)

        n_frames = mag.shape[1]
        if n_frames < 20:
            warnings.warn(
                "apply_spectral_gate: too few STFT frames for reliable noise "
                "estimation — skipping this stage for this channel."
            )
            cleaned.append(ch)
            continue

        frame_power = mag.mean(axis=0)
        noise_mask  = frame_power <= np.percentile(frame_power, noise_percentile)
        noise_cols  = mag[:, noise_mask]
        if noise_cols.shape[1] == 0:
            cleaned.append(ch)
            continue
        noise_profile = np.median(noise_cols, axis=1, keepdims=True) + 1e-8

        ratio_db  = librosa.amplitude_to_db(mag / noise_profile)
        # Sigmoid centred on threshold_db
        sigmoid   = 1.0 / (1.0 + np.exp(-(ratio_db - threshold_db)))
        gate_floor = librosa.db_to_amplitude(-reduction_db)
        soft_mask  = gate_floor + (1.0 - gate_floor) * sigmoid

        D_out = soft_mask * mag * np.exp(1j * phase)
        cleaned.append(librosa.istft(D_out, hop_length=_HOP_LENGTH, length=len(ch)))

    return np.stack(cleaned)


# ---------------------------------------------------------------------------
# Stage 5: Wiener-style spectral subtraction
# ---------------------------------------------------------------------------

def apply_wiener_subtraction(
    audio: np.ndarray,
    sr: int,
    noise_percentile: float = 10.0,
    over_subtraction: float = 1.0,
    spectral_floor: float = 0.1,
) -> np.ndarray:
    """
    Per-bin SNR-weighted gain applied to the complex STFT.
    H(f,t) = max(1 - alpha * N(f) / S(f,t), floor)
    Targets: broadband stationary noise and quiet bleed surviving earlier stages.
    """
    cleaned = []
    for ch in audio:
        D     = librosa.stft(ch, n_fft=_N_FFT, hop_length=_HOP_LENGTH)
        power = np.abs(D) ** 2

        n_frames = power.shape[1]
        if n_frames < 20:
            warnings.warn(
                "apply_wiener_subtraction: too few STFT frames — skipping."
            )
            cleaned.append(ch)
            continue

        frame_power = power.mean(axis=0)
        noise_mask  = frame_power <= np.percentile(frame_power, noise_percentile)
        noise_cols  = power[:, noise_mask]
        if noise_cols.shape[1] == 0:
            cleaned.append(ch)
            continue
        noise_psd = np.mean(noise_cols, axis=1, keepdims=True) + 1e-8

        wiener_gain = np.maximum(
            1.0 - over_subtraction * noise_psd / (power + 1e-8),
            spectral_floor,
        )

        D_out = wiener_gain * D
        cleaned.append(librosa.istft(D_out, hop_length=_HOP_LENGTH, length=len(ch)))

    return np.stack(cleaned)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def clean_guitar_stem(
    input_path: Path,
    output_path: Path,
    config: dict | None = None,
) -> Path:
    """
    Run the full DSP cleaning chain on a guitar stem WAV.

    Args:
        input_path:  Path to the raw Demucs guitar stem (.wav)
        output_path: Where to write the cleaned stem
        config:      Per-stage overrides (merged over DEFAULT_CLEANING_CONFIG).
                     Example: {"hpss": {"enabled": False}, "spec_gate": {"threshold_db": 10}}

    Returns:
        output_path
    """
    # Merge user overrides onto defaults (shallow per-stage)
    cfg = {k: dict(v) for k, v in DEFAULT_CLEANING_CONFIG.items()}
    if config:
        for key, overrides in config.items():
            if key in cfg:
                cfg[key].update(overrides)

    audio, sr = _load_audio(Path(input_path))

    if cfg["rms_normalize"]["enabled"]:
        kw = {k: v for k, v in cfg["rms_normalize"].items() if k != "enabled"}
        audio = apply_rms_normalize(audio, sr, **kw)

    if cfg["bandpass"]["enabled"]:
        kw = {k: v for k, v in cfg["bandpass"].items() if k != "enabled"}
        audio = apply_guitar_bandpass(audio, sr, **kw)

    if cfg["hpss"]["enabled"]:
        kw = {k: v for k, v in cfg["hpss"].items() if k != "enabled"}
        audio = apply_hpss(audio, sr, **kw)

    if cfg["sat_repair"]["enabled"]:
        kw = {k: v for k, v in cfg["sat_repair"].items() if k != "enabled"}
        audio = apply_soft_saturation_repair(audio, sr, **kw)

    if cfg["spec_gate"]["enabled"]:
        kw = {k: v for k, v in cfg["spec_gate"].items() if k != "enabled"}
        audio = apply_spectral_gate(audio, sr, **kw)

    if cfg["wiener"]["enabled"]:
        kw = {k: v for k, v in cfg["wiener"].items() if k != "enabled"}
        audio = apply_wiener_subtraction(audio, sr, **kw)

    _save_audio(audio, sr, Path(output_path))
    return Path(output_path)
