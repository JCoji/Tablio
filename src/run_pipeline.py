"""
Pipeline orchestration for Tablio.

Usage:
    python src/run_pipeline.py --input data/my_song.wav

To skip a node, set enabled=False on it in the `nodes` list below.
To swap a node, replace its `fn` with any callable that accepts and returns a PipelineState.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from nodes import stem_extraction, guitar_stem_cleaning, create_dataset, load_model, predict
from predict_on_custom import _save_midi
from evaluation.tablature_export import save_notes_to_ascii_tab
import config


@dataclass
class PipelineState:
    input_path: Path
    guitar_stem_path: Optional[Path] = None
    predicted_notes: Optional[list] = None
    model: Optional[object] = None
    mixed_dataset_dir: Path = Path("data/guitarset_mixed")
    output_dir: Path = Path("output")
    artifacts_dir: Path = Path("hyperparam_set_v1")
    extras: dict = field(default_factory=dict)


@dataclass
class Node:
    name: str
    fn: Callable
    enabled: bool = True


def run_pipeline(state: PipelineState, nodes: list[Node]) -> PipelineState:
    for node in nodes:
        if not node.enabled:
            print(f"[SKIP] {node.name}")
            continue
        print(f"[RUN]  {node.name}")
        state = node.fn(state)
    return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tablio pipeline")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--musdb", default="musdb18hq", help="Path to MUSDB18-HQ dataset")
    parser.add_argument("--artifacts", default="hyperparam_set_v1", help="Path to model artifacts dir")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--device", default="cpu", help="Device for inference (cpu, cuda, mps)")
    parser.add_argument("--run", default="run_7_Test_Higher_OnsetLossWeight_10_augEnabled", help="Model run folder name (skips interactive prompt)")
    parser.add_argument("--no-cleaning", action="store_true", help="Skip guitar_stem_cleaning entirely")
    parser.add_argument("--no-hpss",     action="store_true", help="Disable HPSS stage in cleaning")
    parser.add_argument("--sat-repair",  action="store_true", help="Enable soft saturation repair (off by default)")
    args = parser.parse_args()

    # Build per-run cleaning config from flags
    cleaning_cfg = {}
    if args.no_hpss:
        cleaning_cfg["hpss"] = {"enabled": False}
    if args.sat_repair:
        cleaning_cfg["sat_repair"] = {"enabled": True}

    state = PipelineState(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        artifacts_dir=Path(args.artifacts),
        extras={"musdb_dir": args.musdb, "device": args.device, "run_name": args.run},
    )

    nodes = [
        Node("stem_extraction",      stem_extraction),
        Node("guitar_stem_cleaning",
             lambda s: guitar_stem_cleaning(s, cleaning_config=cleaning_cfg or None),
             enabled=not args.no_cleaning),
        Node("create_dataset",       create_dataset,  enabled=False),
        Node("load_model",           load_model),
        Node("predict",              predict),
    ]

    state = run_pipeline(state, nodes)

    # Save outputs if predict node produced notes
    if state.predicted_notes:
        track_id = Path(args.input).stem
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)

        midi_path = str(out / f"{track_id}.mid")
        _save_midi(state.predicted_notes, midi_path, track_id)
        print(f"  MIDI      -> {midi_path}")

        tab_path = str(out / f"{track_id}_tab.txt")
        save_notes_to_ascii_tab(state.predicted_notes, tab_path, track_id, config)
        print(f"  Tablature -> {tab_path}")
    else:
        print("No notes predicted — check model output.")
