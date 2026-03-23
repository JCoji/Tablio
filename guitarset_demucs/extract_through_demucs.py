"""
Run the existing GuitarSet synthetic mixes through Demucs to extract guitar stems.

Produces a new dataset where every training example is a Demucs guitar stem extracted
from a realistic guitar+drums+bass+vocals mix.  Retraining the Tab CRNN on this dataset
teaches the model to transcribe despite Demucs bleed and amplitude artifacts, eliminating
the need for DSP post-processing.

Input layout (one .wav + one .jams per track):
    {input_dir}/{split}/*.wav
    {input_dir}/{split}/*.jams

Output layout (same format — drop-in for experiment notebooks):
    {output_dir}/{split}/*.wav   ← Demucs guitar stem
    {output_dir}/{split}/*.jams  ← copied from input unchanged

Usage:
    python guitarset_demucs/extract_through_demucs.py
    python guitarset_demucs/extract_through_demucs.py --input guitarset_demucs/guitarset_mixes_dataset_clean
    python guitarset_demucs/extract_through_demucs.py --device mps --force
"""

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

# Make src/ importable from anywhere
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from stem_extraction import StemSeparator  # noqa: E402


def _auto_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def extract_split(
    separator: StemSeparator,
    split_input_dir: Path,
    split_output_dir: Path,
    force: bool,
) -> dict:
    """Process all .wav files in one split directory. Returns counts."""
    split_output_dir.mkdir(parents=True, exist_ok=True)

    # Redirect StemSeparator output to this split's directory (no model reload needed)
    separator.cache_dir = split_output_dir

    wav_files = sorted(split_input_dir.glob("*.wav"))
    if not wav_files:
        print(f"  [WARN] No .wav files found in {split_input_dir}")
        return {"processed": 0, "skipped": 0, "errors": 0}

    counts = {"processed": 0, "skipped": 0, "errors": 0}

    pbar = tqdm(wav_files, desc=f"  {split_input_dir.name}", unit="track", leave=True)
    for wav_path in pbar:
        out_wav = split_output_dir / wav_path.name
        jams_src = wav_path.with_suffix(".jams")
        jams_dst = split_output_dir / jams_src.name

        # Skip if both output files already exist and --force not set
        if out_wav.exists() and jams_dst.exists() and not force:
            counts["skipped"] += 1
            pbar.set_postfix(skipped=counts["skipped"])
            continue

        try:
            separator.separate(wav_path, force=force, target_stem="guitar")
            counts["processed"] += 1
        except Exception as e:
            tqdm.write(f"  [ERROR] {wav_path.name}: {e}")
            counts["errors"] += 1
            continue

        if jams_src.exists():
            shutil.copy2(jams_src, jams_dst)
        else:
            tqdm.write(f"  [WARN] No .jams for {wav_path.name} — skipping annotation copy")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Extract guitar stems via Demucs from GuitarSet synthetic mixes."
    )
    parser.add_argument(
        "--input",
        default="guitarset_demucs/guitarset_mixed_dataset_augmented",
        help="Root of the source mixed dataset (default: guitarset_demucs/guitarset_mixed_dataset_augmented)",
    )
    parser.add_argument(
        "--output",
        default="GuitarSetIsolated",
        help="Root of the output dataset (default: GuitarSetIsolated)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device: cpu / cuda / mps (default: auto-detect)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if output already exists",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        metavar="SPLIT",
        help="Which splits to process (default: train val test)",
    )
    args = parser.parse_args()

    input_root = _REPO_ROOT / args.input
    output_root = _REPO_ROOT / args.output
    device = args.device or _auto_device()

    if not input_root.exists():
        print(f"[ERROR] Input directory not found: {input_root}")
        sys.exit(1)

    print(f"Input:  {input_root}")
    print(f"Output: {output_root}")
    print(f"Device: {device}")
    print(f"Splits: {args.splits}")
    print()

    # Load Demucs once — reused across all splits and files
    # cache_dir is overridden per split so stems land in the right place
    separator = StemSeparator(
        model="htdemucs_6s",
        cache_dir=str(output_root),  # overridden per split below
        device=device,
    )
    separator._load_model()
    print()

    total = {"processed": 0, "skipped": 0, "errors": 0}

    for split in args.splits:
        split_in = input_root / split
        split_out = output_root / split

        if not split_in.exists():
            print(f"[SKIP] Split directory not found: {split_in}")
            continue

        print(f"[{split}]")
        counts = extract_split(separator, split_in, split_out, force=args.force)
        for k, v in counts.items():
            total[k] += v
        print(
            f"  done — processed: {counts['processed']}, "
            f"skipped: {counts['skipped']}, errors: {counts['errors']}"
        )
        print()

    print("=" * 50)
    print(
        f"Total — processed: {total['processed']}, "
        f"skipped: {total['skipped']}, errors: {total['errors']}"
    )
    print(f"Output written to: {output_root}")


if __name__ == "__main__":
    main()
