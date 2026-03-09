import os
import librosa

# --- Ścieżki Podstawowe ---
BASE_PROJECT_DIR = os.getcwd()
DATA_HOME_DEFAULT = os.path.join(BASE_PROJECT_DIR, "_mir_datasets_storage")
OUTPUT_BASE_DIR_DEFAULT = os.path.join(BASE_PROJECT_DIR, "_processed_guitarset_data")
DEFAULT_HYPERPARAMETER_FILE = "hyperparam_set_v1.json"

# --- Parametry Przetwarzania Audio ---
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
MAX_FRETS = 20

# --- Parametry CQT (dostosowane do potencjalnie najlepszej konfiguracji) ---
FMIN_CQT = librosa.note_to_hz('E2') # Najniższy dźwięk gitary E2 (MIDI 40)
N_BINS_CQT = 168
BINS_PER_OCTAVE_CQT = 24

# --- Parametry Podziału Danych i Preprocessingu ---
OPEN_STRING_PITCHES_JAMS = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
FRET_PADDING_VALUE = -100
FRET_SILENCE_CLASS_OFFSET = 1
PROBLEMATIC_FILES = ["04_BN3-154-E_comp", "04_Jazz1-200-B_comp"]
VALIDATION_SPLIT_SIZE = 0.1
TEST_SPLIT_SIZE = 0.1
RANDOM_SEED = 42
CLEAR_CONSOLE_EVERY_N_RUNS = 10
NUM_SAMPLES_TO_VISUALIZE_FROM_TEST = 5

# --- Domyślne Parametry Datasetu (dla GuitarSetTabDataset) ---
DATASET_COMMON_PARAMS = {
    "audio_hop_length": HOP_LENGTH,
    "audio_sample_rate": SAMPLE_RATE,
    "max_fret_value": MAX_FRETS,
    "audio_n_cqt_bins": N_BINS_CQT,
    "audio_cqt_bins_per_octave": BINS_PER_OCTAVE_CQT,
    "audio_cqt_fmin": FMIN_CQT,
}

# --- Domyślne Parametry Augmentacji (dla GuitarSetTabDataset, split 'train') ---
DATASET_TRAIN_AUGMENTATION_PARAMS = {
    "enable_audio_augmentations": True,
    "aug_p_time_stretch": 0.6,
    "aug_time_stretch_limits": [0.8, 1.2],
    "aug_p_add_noise": 0.7,
    "aug_noise_level_limits": [0.001, 0.01],
    "aug_p_random_gain": 0.7,
    "aug_gain_limits": [0.6, 1.4],
    "enable_specaugment": True,
    "specaug_time_mask_param": 40,
    "specaug_freq_mask_param": 26
}

DATASET_EVAL_AUGMENTATION_PARAMS = {
    "enable_audio_augmentations": False,
    "enable_specaugment": False,
}

# --- Parametry Walidacji Danych (dla validation.py) ---
VALIDATION_SHAPE_PARAMS = {
    'N_BINS_CQT': N_BINS_CQT,
    'N_PITCH_BINS': 88,
    'NUM_STRINGS': 6
}

# --- Domyślne Parametry Modelu (szczególnie dla stałej części CNN) ---
CNN_INPUT_CHANNELS = 1
CNN_OUTPUT_CHANNELS_LIST_DEFAULT = [32, 64, 128, 128, 128]
CNN_KERNEL_SIZES_DEFAULT = [(3, 3), (3, 3), (3, 3), (3, 3), (3, 3)]
CNN_STRIDES_DEFAULT = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_PADDINGS_DEFAULT = [(1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
CNN_POOLING_KERNELS_DEFAULT = [(2,1), (2,1), (2,1), (2,1), (1,1)]
CNN_POOLING_STRIDES_DEFAULT = [(2,1), (2,1), (2,1), (2,1), (1,1)]
DEFAULT_NUM_STRINGS = 6

# --- Ustawienia MIDI ---
OPEN_STRING_PITCHES_MIDI = {0: 40, 1: 45, 2: 50, 3: 55, 4: 59, 5: 64}
DEFAULT_MIDI_VELOCITY = 100
ACOUSTIC_GUITAR_STEEL_PROGRAM = 25
DEFAULT_MIDI_INITIAL_TEMPO = 120.0
MIN_NOTE_DURATION_FRAMES = 2

# --- Ustawienia Wyszukiwania Progu Onsetów ---
TDR_THRESH_SEARCH_MIN = 0.05
TDR_THRESH_SEARCH_MAX = 0.95
TDR_THRESH_SEARCH_STEP = 0.01
DEFAULT_TDR_THRESHOLD = 0.5

# --- Nazewnictwo Plików i Ścieżki dla Artefaktów ---
DEFAULT_TRACK_ID_BASE = "unknown_track"
OUTPUT_MIDI_FILENAME_SUFFIX = "_prediction.mid"
ORIGINAL_WAV_FILENAME_SUFFIX = "_original.wav"
PLOT_FILENAME_PREFIX = "plot_prediction_"
PLOT_FILENAME_SUFFIX = ".png"
DEFAULT_PLOT_TRACK_ID_PREFIX = "test_sample_"

# --- Ustawienia Generowania Tabulatur Tekstowych ---
TAB_SLOT_CHAR_WIDTH = 4
TAB_SUSTAIN_CHAR = '-'
TAB_OVERFLOW_CHAR = '>'
TAB_FRAMES_PER_SLOT = 4
TAB_LINE_BREAK_AFTER_SLOTS = 30
TAB_STRING_NAMES_DISPLAY_6_STRING = ["e", "B", "G", "D", "A", "E"]
TAB_SEPARATOR_CHAR = '|'
TAB_GROUP_SEPARATOR_EVERY_N_SLOTS = 4
TAB_TRACK_ID_PREFIX = "sample"
TAB_GT_FILENAME_SUFFIX = "_tablature_ground_truth.txt"
TAB_PRED_FILENAME_SUFFIX_TEMPLATE = "_tablature_prediction_thresh{threshold:.2f}.txt"

# --- Domyślne Parametry Treningu ---
NUM_EPOCHS_DEFAULT = 300
BATCH_SIZE_DEFAULT = 2
FRET_LOSS_WEIGHT_DEFAULT = 1.0
EARLY_STOPPING_PATIENCE_DEFAULT = 25
CHECKPOINT_METRIC_DEFAULT = 'val_tdr_f1'

# --- Najlepsze Hiperparametry RNN (z run_72, TDR F1=0.8569) ---
RNN_TYPE_DEFAULT = "GRU"
RNN_HIDDEN_SIZE_DEFAULT = 768
RNN_LAYERS_DEFAULT = 2
RNN_DROPOUT_DEFAULT = 0.5
RNN_BIDIRECTIONAL_DEFAULT = True

# --- Najlepsze Hiperparametry Optymalizatora (z run_72) ---
LEARNING_RATE_INIT_DEFAULT = 0.0003
ONSET_LOSS_WEIGHT_DEFAULT = 9.0
ONSET_POS_WEIGHT_MANUAL_VALUE_DEFAULT = 6.0
WEIGHT_DECAY_DEFAULT = 0.0001
SCHEDULER_PATIENCE_DEFAULT = 10
SCHEDULER_FACTOR_DEFAULT = 0.2

# --- Augmentacja: Pogłos (Reverb) oparta na Scipy ---
DATASET_TRAIN_AUGMENTATION_REVERB_PARAMS = {
    "enabled": True,
    "probability": 0.4,
    "decay_seconds_range": [0.10, 0.45],  # Zakres losowania czasu pogłosu
    "wet_level_range": [0.1, 0.35],        # Zakres losowania "ilości" pogłosu
}

# --- Augmentacja: Korektor (EQ) oparta na Scipy ---
DATASET_TRAIN_AUGMENTATION_EQ_PARAMS = {
    "enabled": True,
    "probability": 0.5,
    "low_cutoff_hz_range": [250, 400],   # Zakres losowania dolnej częstotliwości
    "high_cutoff_hz_range": [3000, 4500], # Zakres losowania górnej częstotliwości
}

# --- Augmentacja: Clipping (Przesterowanie) ---
DATASET_TRAIN_AUGMENTATION_CLIPPING_PARAMS = {
    "enabled": True,
    "probability": 0.3,
    "threshold_range": [0.5, 0.9] # Zakres losowania progu obcięcia sygnału
}