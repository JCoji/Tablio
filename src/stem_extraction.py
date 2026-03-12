"""
Demucs stem separation wrapper using native Python library
"""

import shutil
import logging
from pathlib import Path
import torch
import torchaudio as ta
from demucs.pretrained import get_model
from demucs.apply import apply_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StemSeparator:
    """
    Wrapper for Demucs stem separation

    Args:
        model: Demucs model name ('htdemucs_6s' for 6-stem including guitar)
        cache_dir: Where to store separated stems (for caching)
        device: 'mps', 'cuda', 'cpu', or None to auto-detect
    """

    def __init__(self, model='htdemucs_6s', cache_dir='output/extracted_guitar_stems', device=None):
        self.model_name = model
        self.cache_dir = Path(cache_dir).resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")
        self._model = None  # lazy-loaded on first use

    def _load_model(self):
        if self._model is None:
            logger.info(f"Loading model: {self.model_name}")
            self._model = get_model(self.model_name)
            self._model.to(self.device)
        return self._model

    def _get_cached_stem_path(self, audio_path, stem):
        return self.cache_dir / f"{audio_path.stem}.wav"

    def separate(self, audio_path, force=False, target_stem='guitar'):
        """
        Separate audio file into stems using Demucs

        Args:
            audio_path: Path to input audio file (.mp3, .wav, .flac, etc.)
            force: If True, re-process even if cached stem exists
            target_stem: Which stem to return ('guitar', 'vocals', 'bass', 'drums', 'piano', 'other')

        Returns:
            Path to separated stem (WAV file)
        """
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        stem_path = self._get_cached_stem_path(audio_path, target_stem)
        if stem_path.exists() and not force:
            logger.info(f"Using cached stem: {stem_path}")
            return stem_path

        model = self._load_model()
        logger.info(f"Separating stems for: {audio_path.name}")

        wav, sr = ta.load(str(audio_path))

        if sr != model.samplerate:
            wav = ta.functional.resample(wav, sr, model.samplerate)

        if wav.shape[0] != model.audio_channels:
            if model.audio_channels == 2 and wav.shape[0] == 1:
                wav = wav.repeat(2, 1)
            else:
                wav = wav[:model.audio_channels]

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(model, wav[None].to(self.device), device=self.device)[0]

        target_idx = model.sources.index(target_stem)
        ta.save(str(stem_path), sources[target_idx].cpu(), model.samplerate)

        logger.info(f"Stem saved to: {stem_path}")
        return stem_path

    def get_all_stems(self, audio_path, force=False):
        """
        Separate all stems and return their paths.

        Returns:
            dict: Mapping of stem names to file paths
        """
        audio_path = Path(audio_path).resolve()
        model = self._load_model()

        stem_dir = self.cache_dir / audio_path.stem
        if not force and stem_dir.exists():
            return {f.stem: f for f in stem_dir.glob('*.wav')}

        wav, sr = ta.load(str(audio_path))
        if sr != model.samplerate:
            wav = ta.functional.resample(wav, sr, model.samplerate)
        if wav.shape[0] != model.audio_channels:
            wav = wav.repeat(2, 1) if model.audio_channels == 2 else wav[:model.audio_channels]

        ref = wav.mean(0)
        wav = (wav - ref.mean()) / ref.std()

        with torch.no_grad():
            sources = apply_model(model, wav[None].to(self.device), device=self.device)[0]

        stem_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate(model.sources):
            ta.save(str(stem_dir / f"{name}.wav"), sources[i].cpu(), model.samplerate)

        logger.info(f"All stems saved to: {stem_dir}")
        return {f.stem: f for f in stem_dir.glob('*.wav')}

    def clear_cache(self, song_name=None):
        """
        Clear cached stems

        Args:
            song_name: If provided, only clear this song. Otherwise clear all.
        """
        if song_name:
            song_cache = self.cache_dir / song_name
            if song_cache.exists():
                shutil.rmtree(song_cache)
                logger.info(f"Cleared cache for: {song_name}")
        else:
            for item in self.cache_dir.iterdir():
                if item.is_dir():
                    shutil.rmtree(item)
            logger.info("Cleared all cached stems")


def separate_guitar_stem(audio_path, cache_dir='data/demucs_stems', force=False):
    """Simple wrapper to extract just the guitar stem"""
    separator = StemSeparator(cache_dir=cache_dir)
    return separator.separate(audio_path, force=force, target_stem='guitar')


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stem_extraction.py <audio_file.mp3>")
        sys.exit(1)

    audio_file = sys.argv[1]
    _data_dir = str(Path(__file__).resolve().parent.parent / 'data')

    separator = StemSeparator(model='htdemucs_6s', cache_dir=_data_dir)

    guitar_stem = separator.separate(audio_file, target_stem='guitar')
    print(f"\nGuitar stem saved to: {guitar_stem}")

    all_stems = separator.get_all_stems(audio_file)
    print("\nAll stems:")
    for stem_name, stem_path in all_stems.items():
        print(f"  {stem_name}: {stem_path}")
