import sys
from pathlib import Path

# Ensure project root is on the path for guitarset_demucs and other top-level modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stem_extraction import StemSeparator
from predict_on_custom import run_custom_prediction
from guitarset_demucs.create_demucs_finetuning_dataset import create_mixed_guitarset


def stem_extraction(state):
    separator = StemSeparator()
    state.guitar_stem_path = separator.separate(state.input_path, target_stem="guitar")
    return state


def create_dataset(state):
    create_mixed_guitarset(
        output_dir=state.mixed_dataset_dir,
        musdb_dir=state.extras.get("musdb_dir", "musdb18hq"),
    )
    return state


def predict(state):
    audio = state.guitar_stem_path or state.input_path
    run_custom_prediction(audio, state.artifacts_dir, device=state.extras.get("device", "cpu"))
    return state
