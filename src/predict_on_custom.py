import os
import torch
import numpy as np
import librosa
import pretty_midi

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


def run_custom_prediction(artifacts_dir, audio_dir, device):
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

    supported_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = [
        f for f in sorted(os.listdir(audio_dir))
        if os.path.splitext(f)[1].lower() in supported_exts
    ]
    if not audio_files:
        print(f"No audio files found in '{audio_dir}'")
        return

    output_dir = os.path.join(audio_dir, "predictions")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nFound {len(audio_files)} audio file(s). Saving outputs to: {output_dir}\n")

    for filename in audio_files:
        track_id = os.path.splitext(filename)[0]
        print(f"Processing: {filename}")
        try:
            features = _compute_cqt_features(os.path.join(audio_dir, filename), device)
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
