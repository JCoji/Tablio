import sys
from pathlib import Path

# Ensure project root is on the path for guitarset_demucs and other top-level modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from stem_extraction import StemSeparator
from predict_on_custom import run_custom_prediction, predict_notes
from guitarset_demucs.create_demucs_finetuning_dataset import create_mixed_guitarset
from model.architecture import GuitarTabCRNN
from model.utils import load_best_model
from audio_cleaning import clean_guitar_stem


def stem_extraction(state):
    separator = StemSeparator()
    state.guitar_stem_path = separator.separate(state.input_path, target_stem="guitar")
    return state


def guitar_stem_cleaning(state, cleaning_config=None):
    if state.guitar_stem_path is None:
        raise ValueError("guitar_stem_cleaning: state.guitar_stem_path is None — run stem_extraction first")
    out_dir = state.output_dir / "cleaned_guitar_stems"
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"{state.guitar_stem_path.stem}_cleaned.wav"
    state.guitar_stem_path = clean_guitar_stem(
        state.guitar_stem_path, output_path, config=cleaning_config
    )
    return state


def create_dataset(state):
    create_mixed_guitarset(
        output_dir=state.mixed_dataset_dir,
        musdb_dir=state.extras.get("musdb_dir", "musdb18hq"),
    )
    return state


def load_model(state):
    """Load the model once and store it in state for reuse across predict calls."""
    if state.model is not None:
        return state  # already loaded
    device = state.extras.get("device", "cpu")
    run_name = state.extras.get("run_name")
    artifacts_dir = str(state.artifacts_dir)
    import os
    run_dir = os.path.join(artifacts_dir, run_name)
    state.model = load_best_model(
        model_class=GuitarTabCRNN,
        model_path=os.path.join(run_dir, "best_model.pth"),
        run_config_path=os.path.join(run_dir, "run_configuration.json"),
        device=device,
    )
    return state


def predict(state):
    audio = state.guitar_stem_path or state.input_path
    device = state.extras.get("device", "cpu")
    if state.model is not None:
        state.predicted_notes = predict_notes(audio, state.model, device)
    else:
        run_custom_prediction(audio, state.artifacts_dir, device=device, run_name=state.extras.get("run_name"))
    return state
