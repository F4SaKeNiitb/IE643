# ------------------------------
# Import necessary libraries
# ------------------------------
import os
import glob
import random
import argparse
import warnings
from pathlib import Path
from tqdm import tqdm
import numpy as np
import soundfile as sf
import librosa
import scipy.signal as sps
from pydub import AudioSegment
import pandas as pd


# =============================================================================
#                               UTILITY FUNCTIONS
# =============================================================================

def load_audio(path, sr=None):
    """
    Load an audio file using soundfile, optionally resample to target rate.

    Parameters:
        path (str): Path to audio file.
        sr (int): Target sample rate. If None, original rate is retained.
    Returns:
        wav (np.ndarray): Loaded (and possibly resampled) waveform.
        sr (int): Sampling rate of output waveform.
    """
    wav, orig_sr = sf.read(path)
    if wav.ndim > 1:
        wav = np.mean(wav, axis=1)  # Convert to mono
    if sr is not None and orig_sr != sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr, sr)
        return wav, sr
    return wav.astype(np.float32), orig_sr


def save_wav(path, wav, sr):
    """Save waveform as 16-bit PCM WAV file."""
    sf.write(path, wav, samplerate=sr, subtype='PCM_16')


def rms(sig):
    """Compute Root Mean Square (RMS) of a signal."""
    return np.sqrt(np.mean(sig**2) + 1e-12)


def scale_to_snr(clean, noise, target_snr_db):
    """
    Scale noise so that the resulting Signal-to-Noise Ratio matches target SNR.

    SNR(clean:noise) = target_snr_db
    """
    clean_r = rms(clean)
    noise_r = rms(noise)
    if noise_r == 0:
        return noise
    target_ratio = 10.0 ** (target_snr_db / 20.0)
    noise_scale = clean_r / (target_ratio * noise_r + 1e-12)
    return noise * noise_scale


def add_awgn_at_snr(clean, snr_db):
    """Add white Gaussian noise at a defined SNR."""
    noise = np.random.normal(0, 1, size=clean.shape).astype(np.float32)
    noise = scale_to_snr(clean, noise, snr_db)
    return clean + noise, snr_db


def mix_with_noise_file(clean, noise_file, snr_db, sr_target):
    """Mix the input audio with a real noise file at the specified SNR."""
    noise, n_sr = load_audio(noise_file, sr=sr_target)
    # Adjust noise length to match clean audio
    if len(noise) < len(clean):
        noise = np.tile(noise, int(np.ceil(len(clean) / len(noise))))[:len(clean)]
    else:
        start = random.randint(0, max(0, len(noise) - len(clean)))
        noise = noise[start:start + len(clean)]
    noise_scaled = scale_to_snr(clean, noise, snr_db)
    return clean + noise_scaled, snr_db


def lowpass_filter(wav, sr, cutoff_hz):
    """Apply a Butterworth low-pass filter to simulate bandwidth limitation."""
    ny = sr / 2.0
    if cutoff_hz >= ny - 1:
        return wav
    b, a = sps.butter(6, cutoff_hz / ny, btype='low')
    return sps.filtfilt(b, a, wav)


def highpass_filter(wav, sr, cutoff_hz):
    """Apply a Butterworth high-pass filter to remove low-frequency hum."""
    ny = sr / 2.0
    if cutoff_hz <= 0:
        return wav
    b, a = sps.butter(4, cutoff_hz / ny, btype='high')
    return sps.filtfilt(b, a, wav)


def apply_simple_reverb(wav, sr, reverb_scale=0.3, rt60=0.3):
    """
    Apply a simple artificial reverberation using an exponential-decay IR.
    rt60: time (in seconds) for signal to decay by 60 dB.
    """
    ir_len = int(min(sr * (rt60 * 2.0), sr * 3))
    t = np.arange(ir_len) / sr
    decay = np.exp(-3.0 * t / max(rt60, 1e-6))
    noise = np.random.randn(ir_len) * 0.001
    ir = (decay + noise) / np.max(np.abs(decay + noise) + 1e-12)
    wet = sps.fftconvolve(wav, ir)[:len(wav)]
    return (1.0 - reverb_scale) * wav + reverb_scale * wet


def simulate_dropouts(wav, sr, dropout_prob=0.01, dropout_len_ms=100):
    """Simulate packet loss by zeroing out random short-duration windows."""
    out = wav.copy()
    dropout_len = int(sr * dropout_len_ms / 1000.0)
    for i in range(int(len(wav) / dropout_len)):
        if random.random() < dropout_prob:
            start = i * dropout_len
            end = min(len(wav), start + dropout_len)
            out[start:end] = 0.0
    return out


def apply_clipping(wav, clip_threshold=0.8):
    """Clip waveform amplitude beyond threshold to simulate mic distortion."""
    out = np.copy(wav)
    out[out > clip_threshold] = clip_threshold
    out[out < -clip_threshold] = -clip_threshold
    return out


def encode_decode_with_bitrate(wav, sr, bitrate_kbps):
    """Simulate codec compression artifacts using MP3 encode-decode cycle."""
    seg = AudioSegment(
        (wav * 32767.0).astype(np.int16).tobytes(),
        frame_rate=sr, sample_width=2, channels=1
    )
    import tempfile
    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_out.close()
    seg.export(tmp_out.name, format="mp3", bitrate=f"{bitrate_kbps}k")
    seg2 = AudioSegment.from_file(tmp_out.name, format="mp3")
    os.remove(tmp_out.name)
    return np.array(seg2.get_array_of_samples()).astype(np.float32) / 32767.0


def downsample(wav, orig_sr, target_sr):
    """Resample waveform to target sample rate with fallback for safety."""
    if target_sr is None or orig_sr == target_sr:
        return wav.astype(np.float32)
    try:
        return librosa.resample(wav.astype(np.float32), orig_sr, target_sr)
    except TypeError:
        import math
        g = math.gcd(int(orig_sr), int(target_sr))
        up, down = int(target_sr // g), int(orig_sr // g)
        return sps.resample_poly(wav.astype(np.float32), up, down).astype(np.float32)


# =============================================================================
#                      AUGMENTATION PRESET CONFIGURATIONS
# =============================================================================
F_PRESETS = {
    "F0": {"snr_db": 27, "lowpass": None, "reverb_scale": 0.03, "rt60": 0.1,
           "dropout_prob": 0.0, "codec_kbps": 64, "downsample_sr": 16000},
    "F1": {"snr_db": 18, "lowpass": 6000, "reverb_scale": 0.08, "rt60": 0.15,
           "dropout_prob": 0.01, "codec_kbps": 32, "downsample_sr": 16000},
    "F2": {"snr_db": 8, "lowpass": 4200, "reverb_scale": 0.18, "rt60": 0.25,
           "dropout_prob": 0.03, "codec_kbps": 16, "clip_thresh": 0.85, "downsample_sr": 12000},
    "F3": {"snr_db": 0, "lowpass": 3000, "reverb_scale": 0.30, "rt60": 0.4,
           "dropout_prob": 0.08, "codec_kbps": 8, "clip_thresh": 0.75, "downsample_sr": 8000}
}


# =============================================================================
#                          MAIN AUGMENTATION PIPELINE
# =============================================================================

def make_feeble_versions(file_list, out_dir, noise_files=None, rir_dir=None,
                         presets=F_PRESETS, seed=42):
    """
    Generate augmented 'feeble' versions for each audio file in the dataset.
    Saves both the processed audio and a CSV with applied parameters.

    Steps applied for each level:
        1. Downsample to simulate low-quality capture
        2. Add noise (real or synthetic)
        3. Apply filtering
        4. Add reverberation
        5. Simulate packet loss
        6. Apply clipping distortion
        7. Optionally mix overlapping speech
        8. Apply codec compression
        9. Normalize and save
    """
    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)
    metadata = []

    for infile in tqdm(file_list, desc="Processing files"):
        try:
            basename = Path(infile).stem
            wav, orig_sr = load_audio(infile)
        except Exception as e:
            warnings.warn(f"Failed to load {infile}: {e}")
            continue

        for level, params in presets.items():
            y = wav.copy()
            sr_use = orig_sr

            # Step 1: Downsample
            y = downsample(y, sr_use, params["downsample_sr"])
            sr_use = params["downsample_sr"]

            # Step 2: Add noise
            if noise_files and random.random() < 0.9:
                noise_file = random.choice(noise_files)
                y, used_snr = mix_with_noise_file(y, noise_file, params["snr_db"], sr_use)
                noise_source = os.path.basename(noise_file)
            else:
                y, used_snr = add_awgn_at_snr(y, params["snr_db"])
                noise_source = "gaussian"

            # Step 3: Filtering
            if params.get("lowpass"):
                y = lowpass_filter(y, sr_use, params["lowpass"])
            y = highpass_filter(y, sr_use, 80.0)

            # Step 4: Reverberation
            y = apply_simple_reverb(y, sr_use,
                                    reverb_scale=params["reverb_scale"],
                                    rt60=params["rt60"])

            # Step 5: Dropouts
            if params.get("dropout_prob", 0) > 0:
                y = simulate_dropouts(y, sr_use, params["dropout_prob"], 100)

            # Step 6: Clipping
            if params.get("clip_thresh"):
                y = apply_clipping(y, params["clip_thresh"])

            # Step 7: Codec compression
            if params.get("codec_kbps"):
                try:
                    y_codec = encode_decode_with_bitrate(y, sr_use, params["codec_kbps"])
                    y[:len(y_codec)] = y_codec[:len(y)]
                except Exception as e:
                    warnings.warn(f"Codec simulation failed: {e}")

            # Step 8: Normalize and save
            y /= max(1.0, np.max(np.abs(y)))
            out_path = Path(out_dir) / level / f"{basename}_{level}.wav"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            save_wav(str(out_path), y, sr_use)

            # Record metadata
            metadata.append({
                "input_file": infile,
                "output_file": str(out_path),
                "level": level,
                "sr": sr_use,
                "snr_db": used_snr,
                "lowpass": params.get("lowpass"),
                "reverb_scale": params.get("reverb_scale"),
                "rt60": params.get("rt60"),
                "dropout_prob": params.get("dropout_prob"),
                "codec_kbps": params.get("codec_kbps"),
                "clip_thresh": params.get("clip_thresh"),
                "noise_source": noise_source
            })

    # Save metadata CSV
    meta_df = pd.DataFrame(metadata)
    meta_df.to_csv(Path(out_dir) / "augmentation_metadata.csv", index=False)
    print(f"✓ Augmented files and metadata saved to {out_dir}")


# =============================================================================
#                                 HELPER FUNCTIONS
# =============================================================================
def gather_files(input_dir, exts=("wav", "flac", "mp3")):
    """Recursively collect all audio files from a directory."""
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(input_dir, f"**/*.{ext}"), recursive=True))
    return files


# =============================================================================
#                                    MAIN
# =============================================================================
if __name__ == "__main__":
    # User-configurable paths
    input_dir = r"D:\Common Voice\cv-valid-test\cv-valid-test"
    out_dir = r"D:\Common Voice\cv-valid-test\Feebles_test"
    noise_dir = r"D:\noise\free-sound"
    rir_dir = r"rir_dir"
    seed = 42

    # Gather input files
    file_list = gather_files(input_dir)
    noise_files = gather_files(noise_dir) if noise_dir else None

    # Run augmentation
    make_feeble_versions(file_list=file_list,
                         out_dir=out_dir,
                         noise_files=noise_files,
                         rir_dir=rir_dir,
                         presets=F_PRESETS,
                         seed=seed)

