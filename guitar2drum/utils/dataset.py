# ========== FILE GATHERING ==========
class MultiFolderGuitarToDrumDataset:
    def __init__(self):
        guitar_ssm_dirs = ["mlpr-guitar-ssm", "mlpr-guitar-ssm"]
        drum_mel_dir = os.path.join(BASE_PATH, "mlpr-drums-mel", "drums_mel")
        guitar_mel_dir = os.path.join(BASE_PATH, "mlpr-guitar-mel", "guitar_mel")
        g_map, d_map = {}, {}

        for d in guitar_ssm_dirs:
            p = os.path.join(BASE_PATH, d)
            if not os.path.isdir(p): continue
            for f in os.listdir(p):
                if f.endswith("_guitar_ssm.npy"):
                    g_map[f.replace("_guitar_ssm.npy", "")] = os.path.join(p, f)

        for f in os.listdir(drum_mel_dir):
            if f.endswith("_drums_mel.npy"):
                d_map[f.replace("_drums_mel.npy", "")] = os.path.join(drum_mel_dir, f)

        self.samples = []
        for key in sorted(set(g_map) & set(d_map)):
            g_mel = os.path.join(guitar_mel_dir, key + "_guitar_mel.npy")
            if os.path.exists(g_mel):
                self.samples.append({
                    "g_ssm": g_map[key],
                    "d_mel": d_map[key],
                    "g_mel": g_mel
                })

# ========== LAZY DATASET ==========
class MelSSMDataset(Dataset):
    def __init__(self, song_list):
        self.song_list = song_list
        self.index = []
        for song_id, song in enumerate(song_list):
            g_mel = np.load(song["g_mel"], mmap_mode='r')
            T = g_mel.shape[1]
            for start in range(0, T - INTERVAL_FRAMES + 1, STRIDE_FRAMES):
                self.index.append({"song_id": song_id, "start": start})

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        item = self.index[idx]
        song = self.song_list[item["song_id"]]
        start = item["start"]
        end = start + INTERVAL_FRAMES

        g_mel = np.load(song["g_mel"])
        g_ssm = np.load(song["g_ssm"])
        d_mel = np.load(song["d_mel"])

        def norm(x): return np.clip((x - x.mean()) / (x.std() + 1e-6), -2, 2)

        gm = norm(g_mel[:, start:end])
        gs = norm(g_ssm[start:end, start:end])
        dm = norm(d_mel[:, start:end])

        g_ssm_tensor = torch.tensor(gs).unsqueeze(0).unsqueeze(0)
        g_ssm_downsampled = F.interpolate(g_ssm_tensor, size=SSM_TARGET_SIZE, mode="bilinear", align_corners=False).squeeze(0)
        return (
            torch.tensor(gm).unsqueeze(0),
            g_ssm_downsampled,
            torch.tensor(dm).unsqueeze(0)
        )

