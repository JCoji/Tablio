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

from nodes import stem_extraction, create_dataset, predict


@dataclass
class PipelineState:
    input_path: Path
    guitar_stem_path: Optional[Path] = None
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
    args = parser.parse_args()

    state = PipelineState(
        input_path=Path(args.input),
        output_dir=Path(args.output),
        artifacts_dir=Path(args.artifacts),
        extras={"musdb_dir": args.musdb, "device": args.device},
    )

    nodes = [
        Node("stem_extraction", stem_extraction),
        Node("create_dataset",  create_dataset,  enabled=False),
        Node("predict",         predict),
    ]

    run_pipeline(state, nodes)
