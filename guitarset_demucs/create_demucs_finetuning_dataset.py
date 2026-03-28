"""
Create mixed GuitarSet dataset for fine-tuning and evaluation
Saves: Mixed audio + Original JAMS (no pre-separated stems)

Augmentations applied during mixing:
  - Unbalanced instrument levels (wide random volume ranges per stem)
  - Dynamic range compression (soft-knee, per-stem or mix)
  - Room acoustics (convolution reverb with synthetic RIR, per-stem)
"""

import mirdata
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import jams
from scipy.signal import fftconvolve


REQUIRED_STEMS = ("drums.wav", "bass.wav", "vocals.wav")


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def apply_compression(audio, threshold_db=None, ratio=None):
    """
    Vectorised soft-knee dynamic range compressor.

    Args:
        audio:        1-D float array
        threshold_db: compression threshold in dBFS (default: random -24 … -6)
        ratio:        compression ratio (default: random 2 … 8)
    """
    if threshold_db is None:
        threshold_db = random.uniform(-24.0, -6.0)
    if ratio is None:
        ratio = random.uniform(2.0, 8.0)

    threshold = 10 ** (threshold_db / 20.0)
    abs_audio = np.abs(audio)
    # Gain = 1 below threshold; compressed above
    gain = np.where(
        abs_audio > threshold,
        (threshold + (abs_audio - threshold) / ratio) / np.maximum(abs_audio, 1e-10),
        1.0,
    )
    return audio * gain


def _make_room_ir(sr, decay_time):
    """
    Synthesise a plausible room impulse response as exponentially-decaying
    white noise with a unit direct-sound spike at sample 0.
    """
    ir_len = int(decay_time * sr)
    t = np.arange(ir_len) / sr
    ir = np.random.randn(ir_len) * np.exp(-t / (decay_time / 6.0))
    ir[0] = 1.0  # direct sound
    ir /= np.max(np.abs(ir)) + 1e-10
    return ir


def apply_reverb(audio, sr, room_size=None, wet_dry=None):
    """
    Convolve `audio` with a synthetic room impulse response.

    Args:
        audio:     1-D float array
        sr:        sample rate
        room_size: 0-1 controlling decay length (default: random 0.1 … 0.9)
        wet_dry:   wet/dry blend 0-1 (default: random 0.1 … 0.5)
    """
    if room_size is None:
        room_size = random.uniform(0.1, 0.9)
    if wet_dry is None:
        wet_dry = random.uniform(0.1, 0.5)

    decay_time = 0.05 + room_size * 1.5   # 0.05 s … 1.55 s
    ir = _make_room_ir(sr, decay_time)
    wet = fftconvolve(audio, ir)[: len(audio)]
    wet /= np.max(np.abs(wet)) + 1e-10
    return audio * (1.0 - wet_dry) + wet * wet_dry


def discover_musdb_tracks(musdb_dir):
    """
    Discover MUSDB track directories by locating folders that contain the
    required stem files. Supports both flat and split layouts, e.g.:

    - musdb18hq/<track_name>/
    - musdb18hq/train/<track_name>/
    - musdb18hq/test/<track_name>/
    """
    musdb_dir = Path(musdb_dir)
    if not musdb_dir.exists():
        raise ValueError(f"MUSDB18 directory does not exist: {musdb_dir}")

    tracks_by_split = {"train": [], "test": [], "unknown": []}

    for candidate in musdb_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        if not all((candidate / stem_name).exists() for stem_name in REQUIRED_STEMS):
            continue

        relative_parts = candidate.relative_to(musdb_dir).parts
        split_name = relative_parts[0].lower() if relative_parts else "unknown"
        if split_name not in tracks_by_split:
            split_name = "unknown"

        tracks_by_split[split_name].append(candidate)

    all_tracks = tracks_by_split["train"] + tracks_by_split["test"] + tracks_by_split["unknown"]
    if not all_tracks:
        raise ValueError(
            f"No MUSDB18 tracks with required stems {REQUIRED_STEMS} found in {musdb_dir}"
        )

    return tracks_by_split


def choose_musdb_pool(tracks_by_split, split_name):
    """
    Prefer MUSDB's native split folders when available.
    - training GuitarSet examples draw from MUSDB `train`
    - validation GuitarSet examples draw from MUSDB `test`
    - fallback to any discovered track if those folders are absent
    """
    all_tracks = tracks_by_split["train"] + tracks_by_split["test"] + tracks_by_split["unknown"]

    if split_name == "train" and tracks_by_split["train"]:
        return tracks_by_split["train"]
    if split_name == "val" and tracks_by_split["test"]:
        return tracks_by_split["test"]
    return all_tracks

def create_mixed_guitarset(
    output_dir='data/guitarset_mixed',
    musdb_dir='musdb18hq',
    train_split=0.8,
    augment=True,
    seed=42,
):
    """
    Mix GuitarSet tracks with MUSDB18 background instrumentation

    Args:
        output_dir:  Where to save mixed audio + JAMS
        musdb_dir:   Path to MUSDB18-HQ dataset
        train_split: Fraction for training (rest is validation)
        augment:     Apply reverb, compression, and unbalanced levels (default True)
        seed:        Random seed for reproducible mixes (default 42)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Initialize datasets
    print("Loading GuitarSet...")
    guitarset = mirdata.initialize('guitarset')
    # guitarset.download()  # Uncomment if needed
    
    print("Loading MUSDB18 tracks...")
    tracks_by_split = discover_musdb_tracks(musdb_dir)
    musdb_tracks = tracks_by_split["train"] + tracks_by_split["test"] + tracks_by_split["unknown"]

    print(
        "Found "
        f"{len(musdb_tracks)} MUSDB18 tracks "
        f"(train={len(tracks_by_split['train'])}, "
        f"test={len(tracks_by_split['test'])}, "
        f"other={len(tracks_by_split['unknown'])})"
    )
    
    # Create output directories
    output_dir = Path(output_dir)
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    # Shuffle and split GuitarSet tracks
    track_ids = list(guitarset.track_ids)
    random.shuffle(track_ids)
    split_idx = int(len(track_ids) * train_split)
    train_ids = track_ids[:split_idx]
    val_ids = track_ids[split_idx:]
    
    print(f"\nCreating dataset:")
    print(f"  Training: {len(train_ids)} tracks")
    print(f"  Validation: {len(val_ids)} tracks")
    
    # Process training set
    print("\nProcessing training set...")
    process_split(guitarset, train_ids, choose_musdb_pool(tracks_by_split, 'train'), train_dir, augment=augment)

    # Process validation set
    print("\nProcessing validation set...")
    process_split(guitarset, val_ids, choose_musdb_pool(tracks_by_split, 'val'), val_dir, augment=augment)
    
    print(f"\nDataset creation complete!")
    print(f"   Location: {output_dir}")
    print(f"   Training tracks: {len(list(train_dir.glob('*.wav')))}")
    print(f"   Validation tracks: {len(list(val_dir.glob('*.wav')))}")
    
    # Print statistics
    total_size_gb = sum(f.stat().st_size for f in output_dir.rglob('*.wav')) / 1e9
    print(f"   Total size: {total_size_gb:.1f} GB")


def process_split(guitarset, track_ids, musdb_tracks, output_dir, augment=True):
    """Process one split (train or val)"""
    
    for i, track_id in enumerate(tqdm(track_ids, desc="Mixing tracks")):
        try:
            # 1. Load GuitarSet track
            gt_track = guitarset.track(track_id)
            guitar_audio, sr = sf.read(gt_track.audio_mix_path)
            ground_truth_jams = jams.load(gt_track.jams_path)
            
            # 2. Load random MUSDB18 background stems
            musdb_track = random.choice(musdb_tracks)
            
            # Load stems (drums, bass, vocals)
            drums_path = musdb_track / 'drums.wav'
            bass_path = musdb_track / 'bass.wav'
            vocals_path = musdb_track / 'vocals.wav'
            
            # Check if files exist
            if not all([drums_path.exists(), bass_path.exists(), vocals_path.exists()]):
                print(f"\nMissing stems for {musdb_track.name}, skipping...")
                continue
            
            drums, _ = sf.read(drums_path)
            bass, _ = sf.read(bass_path)
            vocals, _ = sf.read(vocals_path)
            
            # 3. Match lengths (use shortest)
            min_len = min(len(guitar_audio), len(drums), len(bass), len(vocals))
            guitar_audio = guitar_audio[:min_len]

            # Convert stereo to mono if needed
            drums  = drums [:min_len, 0] if drums .ndim > 1 else drums [:min_len]
            bass   = bass  [:min_len, 0] if bass  .ndim > 1 else bass  [:min_len]
            vocals = vocals[:min_len, 0] if vocals.ndim > 1 else vocals[:min_len]

            # 4. Volume levels
            if augment:
                # Unbalanced — wide, asymmetric ranges
                guitar_vol = random.uniform(0.3, 1.3)
                drum_vol   = random.uniform(0.05, 1.1)
                bass_vol   = random.uniform(0.05, 1.1)
                vocal_vol  = random.uniform(0.02, 0.9)
            else:
                guitar_vol = 1.0
                drum_vol   = 0.7
                bass_vol   = 0.7
                vocal_vol  = 0.6

            if augment:
                # Draw per-track augmentation params (all explicit so every mix is unique)
                # Room acoustics — independent room per stem; None = skip reverb entirely
                guitar_reverb = (random.uniform(0.05, 1.0), random.uniform(0.05, 0.6)) if random.random() < 0.7 else None
                drums_reverb  = (random.uniform(0.05, 0.6), random.uniform(0.05, 0.4)) if random.random() < 0.5 else None
                bass_reverb   = (random.uniform(0.05, 0.5), random.uniform(0.03, 0.3)) if random.random() < 0.4 else None
                vocals_reverb = (random.uniform(0.1,  0.8), random.uniform(0.1,  0.5)) if random.random() < 0.5 else None

                # Compression — (threshold_db, ratio) or None = skip
                guitar_comp = (random.uniform(-30, -6),  random.uniform(1.5, 10.0)) if random.random() < 0.6 else None
                drums_comp  = (random.uniform(-24, -6),  random.uniform(2.0, 8.0))  if random.random() < 0.5 else None
                bass_comp   = (random.uniform(-24, -8),  random.uniform(2.0, 6.0))  if random.random() < 0.5 else None
                vocals_comp = (random.uniform(-20, -6),  random.uniform(1.5, 5.0))  if random.random() < 0.4 else None

                # Apply per-stem room acoustics
                if guitar_reverb:
                    guitar_audio = apply_reverb(guitar_audio, sr, room_size=guitar_reverb[0], wet_dry=guitar_reverb[1])
                if drums_reverb:
                    drums  = apply_reverb(drums,  sr, room_size=drums_reverb[0],  wet_dry=drums_reverb[1])
                if bass_reverb:
                    bass   = apply_reverb(bass,   sr, room_size=bass_reverb[0],   wet_dry=bass_reverb[1])
                if vocals_reverb:
                    vocals = apply_reverb(vocals, sr, room_size=vocals_reverb[0], wet_dry=vocals_reverb[1])

                # Apply per-stem dynamic range compression
                if guitar_comp:
                    guitar_audio = apply_compression(guitar_audio, threshold_db=guitar_comp[0], ratio=guitar_comp[1])
                if drums_comp:
                    drums  = apply_compression(drums,  threshold_db=drums_comp[0],  ratio=drums_comp[1])
                if bass_comp:
                    bass   = apply_compression(bass,   threshold_db=bass_comp[0],   ratio=bass_comp[1])
                if vocals_comp:
                    vocals = apply_compression(vocals, threshold_db=vocals_comp[0], ratio=vocals_comp[1])

            # 5. Mix
            full_mix = (
                guitar_audio * guitar_vol +
                drums  * drum_vol +
                bass   * bass_vol +
                vocals * vocal_vol
            )

            # 6. Optional mix-bus compression (augment only; ~40% of tracks)
            if augment and random.random() < 0.4:
                full_mix = apply_compression(
                    full_mix,
                    threshold_db=random.uniform(-18, -4),
                    ratio=random.uniform(1.2, 3.0),
                )

            # 8. Normalize to prevent clipping
            max_val = np.max(np.abs(full_mix))
            if max_val > 0:
                full_mix = full_mix / max_val * 0.9
            
            # 6. Save mixed audio
            output_audio_path = output_dir / f'{track_id}.wav'
            sf.write(output_audio_path, full_mix, sr)
            
            # 7. Save original JAMS annotation
            output_jams_path = output_dir / f'{track_id}.jams'
            ground_truth_jams.save(str(output_jams_path))
            
        except Exception as e:
            print(f"\nError processing {track_id}: {e}")
            continue


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create mixed GuitarSet dataset')
    parser.add_argument('--output', default='data/guitarset_mixed',
                        help='Output directory')
    parser.add_argument('--musdb', default='musdb18hq',
                        help='Path to MUSDB18-HQ dataset')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/val split (default: 0.8)')
    parser.add_argument('--augment', action=argparse.BooleanOptionalAction, default=True,
                        help='Apply reverb/compression/unbalanced levels (default: on). Use --no-augment to disable.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible mixes (default: 42)')

    args = parser.parse_args()

    create_mixed_guitarset(
        output_dir=args.output,
        musdb_dir=args.musdb,
        train_split=args.split,
        augment=args.augment,
        seed=args.seed,
    )
