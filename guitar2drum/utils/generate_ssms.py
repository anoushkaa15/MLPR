import os
import numpy as np
from tqdm import tqdm
import librosa

MEL_ROOT = "/path/to/output"
SSM_DIRS = {
    "guitar": os.path.join(MEL_ROOT, "guitar_ssm"),
    "drums": os.path.join(MEL_ROOT, "drums_ssm")
}
MEL_DIRS = {
    "guitar": os.path.join(MEL_ROOT, "guitar_mel"),
    "drums": os.path.join(MEL_ROOT, "drums_mel")
}

for path in SSM_DIRS.values():
    os.makedirs(path, exist_ok=True)

def compute_ssm(mel):
    mel = librosa.util.normalize(mel, axis=0)
    return np.dot(mel.T, mel)

# ========== PROCESS ==========
for inst in ["guitar", "drums"]:
    mel_files = [f for f in os.listdir(MEL_DIRS[inst]) if f.endswith('.npy')]
    for mel_file in tqdm(mel_files, desc=f"Generating SSMs ({inst})"):
        try:
            mel = np.load(os.path.join(MEL_DIRS[inst], mel_file))
            ssm = compute_ssm(mel)
            song_id = mel_file.replace(f"_{inst}_mel.npy", "")
            np.save(os.path.join(SSM_DIRS[inst], f"{song_id}_{inst}_ssm.npy"), ssm)
        except Exception as e:
            print(f"Error in {mel_file}: {e}")

print("All SSMs saved.")
