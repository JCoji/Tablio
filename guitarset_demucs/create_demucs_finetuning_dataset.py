"""
Create mixed GuitarSet dataset for fine-tuning and evaluation
Saves: Mixed audio + Original JAMS (no pre-separated stems)
"""

import mirdata
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm
import random
import jams


REQUIRED_STEMS = ("drums.wav", "bass.wav", "vocals.wav")


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
    train_split=0.8
):
    """
    Mix GuitarSet tracks with MUSDB18 background instrumentation
    
    Args:
        output_dir: Where to save mixed audio + JAMS
        musdb_dir: Path to MUSDB18-HQ dataset
        train_split: Fraction for training (rest is validation)
    """
    
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
    print("\n📦 Processing training set...")
    process_split(guitarset, train_ids, choose_musdb_pool(tracks_by_split, 'train'), train_dir)
    
    # Process validation set
    print("\n📦 Processing validation set...")
    process_split(guitarset, val_ids, choose_musdb_pool(tracks_by_split, 'val'), val_dir)
    
    print(f"\n✅ Dataset creation complete!")
    print(f"   Location: {output_dir}")
    print(f"   Training tracks: {len(list(train_dir.glob('*.wav')))}")
    print(f"   Validation tracks: {len(list(val_dir.glob('*.wav')))}")
    
    # Print statistics
    total_size_gb = sum(f.stat().st_size for f in output_dir.rglob('*.wav')) / 1e9
    print(f"   Total size: {total_size_gb:.1f} GB")


def process_split(guitarset, track_ids, musdb_tracks, output_dir):
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
                print(f"\n⚠️  Missing stems for {musdb_track.name}, skipping...")
                continue
            
            drums, _ = sf.read(drums_path)
            bass, _ = sf.read(bass_path)
            vocals, _ = sf.read(vocals_path)
            
            # 3. Match lengths (use shortest)
            min_len = min(len(guitar_audio), len(drums), len(bass), len(vocals))
            guitar_audio = guitar_audio[:min_len]
            
            # Convert stereo to mono if needed
            drums = drums[:min_len, 0] if len(drums.shape) > 1 else drums[:min_len]
            bass = bass[:min_len, 0] if len(bass.shape) > 1 else bass[:min_len]
            vocals = vocals[:min_len, 0] if len(vocals.shape) > 1 else vocals[:min_len]
            
            # 4. Mix with random volume variations (for diversity)
            drum_vol = random.uniform(0.5, 0.8)
            bass_vol = random.uniform(0.5, 0.8)
            vocal_vol = random.uniform(0.4, 0.7)
            
            full_mix = (
                guitar_audio * 1.0 +      # Guitar always at full volume
                drums * drum_vol +
                bass * bass_vol +
                vocals * vocal_vol
            )
            
            # 5. Normalize to prevent clipping
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
            print(f"\n❌ Error processing {track_id}: {e}")
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
    
    args = parser.parse_args()
    
    create_mixed_guitarset(
        output_dir=args.output,
        musdb_dir=args.musdb,
        train_split=args.split
    )
