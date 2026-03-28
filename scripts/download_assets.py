"""
Download model weights and datasets from Hugging Face Hub.

Usage:
    python scripts/download_assets.py              # download everything
    python scripts/download_assets.py --model-only
    python scripts/download_assets.py --data-only
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

HF_MODEL_REPO = "JCoji/tablio-model"
HF_DATASET_CLEAN_REPO = "JCoji/tablio-dataset-clean"
HF_DATASET_AUG_REPO   = "JCoji/tablio-dataset-augmented"

ROOT = Path(__file__).parent.parent

MODEL_DIR = ROOT / "hyperparam_set_v2" / "run_2_Test_Higher_Dropout_0.55_augEnabled"
CLEAN_DIR = ROOT / "guitarset_demucs" / "guitarset_mixed_dataset_clean"
AUG_DIR   = ROOT / "guitarset_demucs" / "guitarset_mixed_dataset_augmented"


def download_model():
    print("Downloading model weights...")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ["best_model.pth", "run_configuration.json"]:
        dest = MODEL_DIR / filename
        if dest.exists():
            print(f"  [skip] {filename} already exists")
            continue
        hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=filename,
            local_dir=MODEL_DIR,
        )
        print(f"  [done] {filename}")


def download_datasets():
    for repo_id, local_dir, split_name in [
        (HF_DATASET_CLEAN_REPO, CLEAN_DIR, "clean"),
        (HF_DATASET_AUG_REPO,   AUG_DIR,   "augmented"),
    ]:
        print(f"Downloading {split_name} dataset...")
        if local_dir.exists() and any(local_dir.rglob("*.wav")):
            print(f"  [skip] {local_dir.name} already exists")
            continue
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=local_dir,
        )
        print(f"  [done] {split_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()

    if args.data_only:
        download_datasets()
    elif args.model_only:
        download_model()
    else:
        download_model()
        download_datasets()

    print("\nAll done.")
