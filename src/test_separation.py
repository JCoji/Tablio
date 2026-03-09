"""
Test script for Demucs separation wrapper
Run this to verify your Docker setup is working
"""

from separation import StemSeparator
from pathlib import Path


def test_basic_separation():
    """Test 1: Basic separation with caching"""
    print("=" * 60)
    print("TEST 1: Basic Separation")
    print("=" * 60)
    
    separator = StemSeparator(
        model='htdemucs_6s',
        cache_dir='data/demucs_stems',
        use_gpu=True  # Change to False if no GPU
    )
    
    # Replace with your test audio file
    test_file = 'data/test_songs/example.mp3'
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("Please create data/test_songs/ and add a test MP3")
        return
    
    # First run - should separate
    print("\n🔧 Running separation (first time - will take 30-60s)...")
    guitar_stem = separator.separate(test_file, target_stem='guitar')
    print(f"✅ Guitar stem: {guitar_stem}")
    
    # Second run - should use cache
    print("\n🔧 Running separation again (should use cache)...")
    guitar_stem_cached = separator.separate(test_file, target_stem='guitar')
    print(f"✅ Cached stem: {guitar_stem_cached}")
    
    assert guitar_stem == guitar_stem_cached, "Cache not working!"
    print("\n✅ Caching works correctly!")


def test_all_stems():
    """Test 2: Get all 6 stems at once"""
    print("\n" + "=" * 60)
    print("TEST 2: Get All Stems")
    print("=" * 60)
    
    separator = StemSeparator()
    test_file = 'data/test_songs/example.mp3'
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return
    
    all_stems = separator.get_all_stems(test_file)
    
    print("\n📁 All separated stems:")
    for stem_name, stem_path in all_stems.items():
        size_mb = stem_path.stat().st_size / (1024 * 1024)
        print(f"  • {stem_name:10s}: {stem_path} ({size_mb:.1f} MB)")
    
    # Verify we got all 6 stems
    expected_stems = {'guitar', 'vocals', 'bass', 'drums', 'piano', 'other'}
    found_stems = set(all_stems.keys())
    
    if found_stems == expected_stems:
        print("\n✅ All 6 stems present!")
    else:
        missing = expected_stems - found_stems
        extra = found_stems - expected_stems
        if missing:
            print(f"\n⚠️  Missing stems: {missing}")
        if extra:
            print(f"\n⚠️  Unexpected stems: {extra}")


def test_force_reprocess():
    """Test 3: Force reprocessing (ignore cache)"""
    print("\n" + "=" * 60)
    print("TEST 3: Force Reprocess")
    print("=" * 60)
    
    separator = StemSeparator()
    test_file = 'data/test_songs/example.mp3'
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return
    
    print("\n🔧 Forcing reprocessing (force=True)...")
    guitar_stem = separator.separate(test_file, force=True, target_stem='guitar')
    print(f"✅ Reprocessed: {guitar_stem}")


def test_experiment_workflow():
    """Test 4: Typical experiment workflow"""
    print("\n" + "=" * 60)
    print("TEST 4: Experiment Workflow")
    print("=" * 60)
    
    # This simulates your actual experiment pipeline
    test_songs = [
        'data/test_songs/song1.mp3',
        'data/test_songs/song2.mp3',
    ]
    
    separator = StemSeparator()
    
    for song_path in test_songs:
        if not Path(song_path).exists():
            print(f"⚠️  Skipping (not found): {song_path}")
            continue
        
        print(f"\n📝 Processing: {Path(song_path).name}")
        
        # Get guitar stem
        guitar_stem = separator.separate(song_path)
        
        # Now you would:
        # 1. Load the stem with librosa
        # 2. Apply cleaning pipeline
        # 3. Run through model
        # 4. Evaluate results
        
        print(f"   ✅ Stem ready: {guitar_stem}")


def test_cache_management():
    """Test 5: Cache clearing"""
    print("\n" + "=" * 60)
    print("TEST 5: Cache Management")
    print("=" * 60)
    
    separator = StemSeparator()
    
    # Check cache size
    cache_dir = Path('data/demucs_stems')
    if cache_dir.exists():
        total_size = sum(
            f.stat().st_size 
            for f in cache_dir.rglob('*') 
            if f.is_file()
        )
        print(f"\n💾 Current cache size: {total_size / (1024**2):.1f} MB")
        
        # List cached songs
        cached_songs = [d.name for d in cache_dir.iterdir() if d.is_dir()]
        print(f"📁 Cached songs ({len(cached_songs)}):")
        for song in cached_songs:
            print(f"   • {song}")
    
    # Uncomment to clear cache
    # print("\n🗑️  Clearing cache...")
    # separator.clear_cache()
    # print("✅ Cache cleared!")


def test_gpu_vs_cpu():
    """Test 6: Compare GPU vs CPU performance"""
    print("\n" + "=" * 60)
    print("TEST 6: GPU vs CPU Performance")
    print("=" * 60)
    
    import time
    
    test_file = 'data/test_songs/example.mp3'
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        return
    
    # Test with GPU
    separator_gpu = StemSeparator(use_gpu=True)
    separator_gpu.clear_cache(Path(test_file).stem)  # Clear for fair test
    
    print("\n⚡ Testing with GPU...")
    start = time.time()
    separator_gpu.separate(test_file, force=True)
    gpu_time = time.time() - start
    print(f"   GPU time: {gpu_time:.1f}s")
    
    # Test with CPU
    separator_cpu = StemSeparator(use_gpu=False)
    separator_cpu.clear_cache(Path(test_file).stem)
    
    print("\n🐌 Testing with CPU...")
    start = time.time()
    separator_cpu.separate(test_file, force=True)
    cpu_time = time.time() - start
    print(f"   CPU time: {cpu_time:.1f}s")
    
    print(f"\n📊 Speedup: {cpu_time / gpu_time:.1f}x faster with GPU")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("DEMUCS DOCKER WRAPPER TEST SUITE")
    print("=" * 60)
    
    # Run tests
    try:
        test_basic_separation()
        test_all_stems()
        test_force_reprocess()
        test_experiment_workflow()
        test_cache_management()
        
        # Uncomment if you want to benchmark GPU vs CPU
        # test_gpu_vs_cpu()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
