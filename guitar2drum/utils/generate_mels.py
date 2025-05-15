# utils/generate_mels.py

import os
import librosa
import numpy as np
from tqdm import tqdm

# ========== CONFIG ==========
SR = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

INPUT_ROOT = "/path/to/songs"  # Each folder has guitar.wav and drums.wav
OUTPUT_ROOT = "/path/to/output"
MEL_DIRS = {
    "guitar": os.path.join(OUTPUT_ROOT, "guitar_mel"),
    "drums": os.path.join(OUTPUT_ROOT, "drums_mel")
}

for path in MEL_DIRS.values():
    os.makedirs(path, exist_ok=True)

def compute_mel(path):
    y, _ = librosa.load(path, sr=SR)
    y = librosa.util.normalize(y)
    mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=N_FFT,
                                         hop_length=HOP_LENGTH, n_mels=N_MELS)
    return librosa.power_to_db(mel, ref=np.max)

# ========== PROCESS ==========
folders = sorted(os.listdir(INPUT_ROOT))
for folder in tqdm(folders, desc="Generating Mel spectrograms"):
    song_path = os.path.join(INPUT_ROOT, folder)
    for inst in ["guitar", "drums"]:
        audio_path = os.path.join(song_path, f"{inst}.wav")
        if not os.path.exists(audio_path):
            continue
        try:
            mel = compute_mel(audio_path)
            np.save(os.path.join(MEL_DIRS[inst], f"{folder}_{inst}_mel.npy"), mel)
        except Exception as e:
            print(f"Failed on {audio_path}: {e}")

print(" All Mel spectrograms saved.")
