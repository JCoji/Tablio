"""
Demucs stem separation wrapper for Docker deployment
Handles volume mounting, path resolution, and caching
"""

import subprocess
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StemSeparator:
    """
    Wrapper for dockerized Demucs stem separation
    
    Args:
        model: Demucs model name (htdemucs_6s for 6-stem including guitar)
        docker_image: Docker image to use
        cache_dir: Where to store separated stems (for caching)
        use_gpu: Whether to pass GPU to Docker container
    """
    
    def __init__(
        self, 
        model='htdemucs_6s',
        docker_image='xserrat/demucs',
        cache_dir='output/extracted_guitar_stems',
        use_gpu=True
    ):
        self.model = model
        self.docker_image = docker_image
        self.cache_dir = Path(cache_dir).resolve()
        self.use_gpu = use_gpu
        
        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Verify Docker is available
        self._check_docker()
    
    def _check_docker(self):
        """Verify Docker is installed and running"""
        try:
            result = subprocess.run(
                ['docker', '--version'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Docker detected: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "Docker not found. Please install Docker: "
                "https://docs.docker.com/get-docker/"
            )
    
    def separate(self, audio_path, force=False, target_stem='guitar'):
        """
        Separate audio file into stems using Demucs
        
        Args:
            audio_path: Path to input audio file (.mp3, .wav, .flac, etc.)
            force: If True, re-process even if cached stem exists
            target_stem: Which stem to return ('guitar', 'vocals', 'bass', 'drums', 'piano', 'other')
        
        Returns:
            Path to separated guitar stem (WAV file)
        """
        audio_path = Path(audio_path).resolve()
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Check cache first
        stem_path = self._get_cached_stem_path(audio_path, target_stem)
        if stem_path.exists() and not force:
            logger.info(f"Using cached stem: {stem_path}")
            return stem_path
        
        # Run separation
        logger.info(f"Separating stems for: {audio_path.name}")
        self._run_demucs(audio_path)
        
        # Move output to cache
        self._cache_output(audio_path, target_stem)
        
        if not stem_path.exists():
            raise RuntimeError(f"Separation failed - stem not found: {stem_path}")
        
        logger.info(f"Stem saved to: {stem_path}")
        return stem_path
    
    def _get_cached_stem_path(self, audio_path, stem):
        """Get path where cached stem should be stored"""
        song_name = audio_path.stem
        return self.cache_dir / song_name / f"{stem}.wav"
    
    def _run_demucs(self, audio_path):
        """
        Execute Demucs Docker container
        
        Volume mapping strategy:
        - Mount parent dir of input file as /input (read-only)
        - Mount cache dir as /output (read-write)
        """
        audio_path = audio_path.resolve()
        input_dir = audio_path.parent
        
        # Build Docker command
        docker_cmd = ['docker', 'run', '--rm']
        
        # GPU support (if requested and available)
        if self.use_gpu:
            docker_cmd.extend(['--gpus', 'all'])
        
        # Volume mounts
        # Input: mount the directory containing the audio file
        docker_cmd.extend(['-v', f'{input_dir}:/input:ro'])
        
        # Output: mount cache directory
        docker_cmd.extend(['-v', f'{self.cache_dir}:/output'])
        
        # Image and Demucs arguments
        docker_cmd.extend([
            self.docker_image,
            '-n', self.model,           # Model name
            '-o', '/output',            # Output directory
            f'/input/{audio_path.name}' # Input file (relative to mounted /input)
        ])
        
        logger.info(f"Running: {' '.join(docker_cmd)}")
        
        try:
            # Run with real-time output
            process = subprocess.Popen(
                docker_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            # Stream output (Demucs shows progress bars)
            for line in process.stdout:
                logger.info(line.rstrip())
            
            process.wait()
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode, 
                    docker_cmd
                )
                
        except subprocess.CalledProcessError as e:
            logger.error(f"Demucs separation failed: {e}")
            raise RuntimeError(f"Stem separation failed for {audio_path.name}")
    
    def _cache_output(self, audio_path, target_stem):
        """
        Move Demucs output from temp location to organized cache
        
        Demucs outputs to: /output/{model}/{song_name}/{stem}.wav
        We want: cache_dir/{song_name}/{stem}.wav
        """
        song_name = audio_path.stem
        
        # Demucs creates: cache_dir/htdemucs_6s/song_name/*.wav
        demucs_output_dir = self.cache_dir / self.model / song_name
        
        # Our organized structure: cache_dir/song_name/*.wav
        target_dir = self.cache_dir / song_name
        
        if not demucs_output_dir.exists():
            logger.warning(f"Demucs output directory not found: {demucs_output_dir}")
            # Check if it went somewhere else
            possible_dirs = list(self.cache_dir.glob(f'*/{song_name}'))
            if possible_dirs:
                demucs_output_dir = possible_dirs[0]
                logger.info(f"Found output at: {demucs_output_dir}")
            else:
                raise RuntimeError(f"Could not find Demucs output for {song_name}")
        
        # Move to organized cache location
        if target_dir.exists():
            shutil.rmtree(target_dir)
        
        shutil.move(str(demucs_output_dir), str(target_dir))
        
        # Clean up empty model directory
        model_dir = self.cache_dir / self.model
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
        
        logger.info(f"Organized stems in: {target_dir}")
    
    def get_all_stems(self, audio_path, force=False):
        """
        Get paths to all separated stems
        
        Returns:
            dict: Mapping of stem names to file paths
        """
        # Run separation once (uses cache if available)
        self.separate(audio_path, force=force, target_stem='guitar')
        
        song_name = Path(audio_path).stem
        stem_dir = self.cache_dir / song_name
        
        stems = {}
        for stem_file in stem_dir.glob('*.wav'):
            stem_name = stem_file.stem
            stems[stem_name] = stem_file
        
        return stems
    
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


# Convenience function for simple use cases
def separate_guitar_stem(audio_path, cache_dir='data/demucs_stems', force=False):
    """
    Simple wrapper to extract just the guitar stem
    
    Usage:
        guitar_wav = separate_guitar_stem('song.mp3')
    """
    separator = StemSeparator(cache_dir=cache_dir)
    return separator.separate(audio_path, force=force, target_stem='guitar')


if __name__ == '__main__':
    # Test/demo usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python separation.py <audio_file.mp3>")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    
    # Initialize separator
    separator = StemSeparator(
        model='htdemucs_6s',
        cache_dir='data/demucs_stems',
        use_gpu=True  # Set to False if no GPU available
    )
    
    # Separate and get guitar stem
    guitar_stem = separator.separate(audio_file, target_stem='guitar')
    print(f"\nGuitar stem saved to: {guitar_stem}")
    
    # Show all available stems
    all_stems = separator.get_all_stems(audio_file)
    print("\nAll stems:")
    for stem_name, stem_path in all_stems.items():
        print(f"  {stem_name}: {stem_path}")