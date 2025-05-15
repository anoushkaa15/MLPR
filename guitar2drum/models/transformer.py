# ========== MODEL ==========
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20480):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x): return x + self.pe[:, :x.size(1)].to(x.device)

class DrumTransformer(nn.Module):
    def __init__(self, embed_dim=128, nhead=8, num_layers=4):
        super().__init__()
        self.g_mel_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, embed_dim, 3, padding=1)
        )
        self.g_ssm_encoder = nn.Sequential(
            nn.Conv2d(1, embed_dim, 3, padding=1), nn.ReLU()
        )
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, nhead, batch_first=True),
            num_layers=num_layers
        )
        self.decoder = nn.Linear(embed_dim, 1)

    def forward(self, g_mel, g_ssm):
        B = g_mel.size(0)
        gm = self.g_mel_encoder(g_mel).flatten(2).permute(0, 2, 1)
        gs = self.g_ssm_encoder(g_ssm).flatten(2).permute(0, 2, 1)
        x_cat = torch.cat([gm, gs], dim=1)
        x = self.pos_encoding(x_cat)
        enc = self.encoder(x)
        mel_tok = self.decoder(enc).squeeze(-1)
        mel_out = mel_tok[:, :16384].view(B, 1, 128, 128)
        return mel_out


   
