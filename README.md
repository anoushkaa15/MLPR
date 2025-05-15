Evaluating Generative AI Drumbeats on Guitar Tracks

The challenge of generating expressive and coherent drum accompaniments remains a significant barrier for solo guitarists and independent musicians. Traditional solutions such as DAW-based programming or loop libraries are either time-consuming or musically generic, often failing to reflect the intricacies of a guitarist’s performance. Prior research in this space primarily utilizes symbolic MIDI data and sequence models like RNNs or Transformers to produce drum tracks, but these often suffer from repetitive patterns, weak structural awareness, and limited audio fidelity.

Our project aims to address these limitations by leveraging a hybrid architecture that combines Self-Similarity Matrices (SSMs), Transformer networks, and Diffusion Models to generate realistic and stylistically responsive drum accompaniments from guitar tracks. Unlike previous models, which separately predict rhythm and dynamics or rely on MIDI-only input, our multitask system jointly learns to predict both the drum SSM and the corresponding drum Mel-spectrogram, enhancing temporal precision and musical alignment.

## Dataset Preparation

We used 710 publicly available multitrack rock and metal songs. Each song was source-separated into `guitar.wav` and `drums.wav` using [Demucs](https://github.com/facebookresearch/demucs). These were then converted to:

- Mel spectrograms using Librosa with:
  - Sample rate = 22050
  - n_fft = 2048
  - hop_length = 512
  - n_mels = 128

- SSMs (Self-Similarity Matrices) by computing cosine similarity over time-normalized Mel spectrograms.

Scripts for preprocessing are provided:
- `utils/generate_mels.py`: Converts `.wav` to Mel spectrograms and saves them as `.npy`
- `utils/generate_ssms.py`: Computes SSMs from Mel spectrograms and saves them as `.npy`

## Model Overview

- **Input**: Guitar Mel-spectrogram and Guitar SSM
- **Output**: Drum Mel-spectrogram (and optionally Drum SSM)
- **Model**: Transformer with cross-attention layers trained in a multitask setup
- **Post-processing**: Diffusion model reconstructs drum audio waveform using predicted features

## Project Structure

```
guitar2drum/
├── train/
│   ├── config.py              # Configuration constants
│   ├── transformer_train.py   # Training loop
├── models/
│   ├── drum_transformer.py    # Model definition (Transformer)
├── utils/
│   ├── generate_mels.py       # Converts .wav to Mel spectrograms
│   ├── generate_ssms.py       # Converts Mel spectrograms to SSMs
│   ├── dataset.py             # Dataset utilities
├── requirements.txt           # Python dependencies

```
