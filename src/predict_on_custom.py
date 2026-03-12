import os
import sys
import torch
import numpy as np
import librosa
import pretty_midi

# Ensure project root is on the path so model/, training/, evaluation/ are importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from model.architecture import GuitarTabCRNN
from model.utils import load_best_model
from training.note_conversion_utils import frames_to_notes_for_eval
from evaluation.tablature_export import save_notes_to_ascii_tab


def _compute_cqt_features(audio_path, device):
    audio, _ = librosa.load(audio_path, sr=config.SAMPLE_RATE, mono=True)
    cqt = librosa.cqt(
        y=audio,
        sr=config.SAMPLE_RATE,
        hop_length=config.HOP_LENGTH,
        fmin=config.FMIN_CQT,
        n_bins=config.N_BINS_CQT,
        bins_per_octave=config.BINS_PER_OCTAVE_CQT,
    )
    log_cqt = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    return torch.tensor(log_cqt, dtype=torch.float32).to(device)


def _run_inference(model, features_tensor):
    with torch.no_grad():
        onset_logits, fret_logits = model(features_tensor.unsqueeze(0))
        onset_probs = torch.sigmoid(onset_logits).squeeze(0)    # (n_frames, 6)
        fret_indices = torch.argmax(fret_logits, dim=-1).squeeze(0)  # (n_frames, 6)
    return onset_probs, fret_indices


def _save_midi(notes_list, output_path, track_id):
    midi = pretty_midi.PrettyMIDI(initial_tempo=config.DEFAULT_MIDI_INITIAL_TEMPO)
    instrument = pretty_midi.Instrument(
        program=config.ACOUSTIC_GUITAR_STEEL_PROGRAM,
        name=str(track_id),
    )
    for note in notes_list:
        instrument.notes.append(pretty_midi.Note(
            velocity=config.DEFAULT_MIDI_VELOCITY,
            pitch=note["pitch_midi"],
            start=note["start_time"],
            end=max(note["end_time"], note["start_time"] + 0.05),
        ))
    midi.instruments.append(instrument)
    midi.write(output_path)


def _select_run(artifacts_dir):
    available_runs = [
        d for d in sorted(os.listdir(artifacts_dir))
        if d.startswith("run_")
        and os.path.exists(os.path.join(artifacts_dir, d, "best_model.pth"))
    ]
    if not available_runs:
        print(f"No model folders found in '{artifacts_dir}'")
        return None

    print("\nAvailable models:")
    for idx, name in enumerate(available_runs):
        print(f"  [{idx + 1}] {name}")

    while True:
        try:
            choice = int(input(f"\nSelect run number (1-{len(available_runs)}) or 0 to cancel: "))
            if choice == 0:
                return None
            if 1 <= choice <= len(available_runs):
                return available_runs[choice - 1]
            print("Invalid choice.")
        except ValueError:
            print("Please enter a number.")


def run_custom_prediction(audio_path, artifacts_dir, device, output_dir=None):
    audio_path = os.path.abspath(audio_path)
    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return

    selected_run = _select_run(artifacts_dir)
    if selected_run is None:
        print("Cancelled.")
        return

    run_dir = os.path.join(artifacts_dir, selected_run)
    model = load_best_model(
        model_class=GuitarTabCRNN,
        model_path=os.path.join(run_dir, "best_model.pth"),
        run_config_path=os.path.join(run_dir, "run_configuration.json"),
        device=device,
    )
    if model is None:
        print("Failed to load model.")
        return

    track_id = os.path.splitext(os.path.basename(audio_path))[0]
    output_dir = output_dir or os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nProcessing: {os.path.basename(audio_path)}")
    print(f"Output -> {output_dir}\n")

    try:
        features = _compute_cqt_features(audio_path, device)
        onset_probs, fret_indices = _run_inference(model, features)

        onset_binary = (onset_probs > config.DEFAULT_TDR_THRESHOLD).float()
        notes = frames_to_notes_for_eval(
            onset_preds_binary_frames=onset_binary.cpu(),
            fret_pred_indices_frames=fret_indices.cpu(),
            frame_hop_length=config.HOP_LENGTH,
            audio_sample_rate=config.SAMPLE_RATE,
        )

        tab_path = os.path.join(output_dir, f"{track_id}_tab.txt")
        save_notes_to_ascii_tab(notes, tab_path, track_id, config)
        print(f"  Tablature -> {tab_path}")

        midi_path = os.path.join(output_dir, f"{track_id}.mid")
        _save_midi(notes, midi_path, track_id)
        print(f"  MIDI      -> {midi_path}")

    except Exception as e:
        print(f"  Error: {e}")

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", help="Path to input audio file (e.g. data/Yume_Utsutsu.wav)")
    args = parser.parse_args()

    _artifacts_dir = os.path.join(PROJECT_ROOT, "hyperparam_set_v1")
    _output_dir = os.path.join(PROJECT_ROOT, "output")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {_device}")
    run_custom_prediction(args.audio_file, _artifacts_dir, _device, output_dir=_output_dir)
