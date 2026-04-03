from scipy.io import loadmat
import mne
import numpy as np
from matplotlib import pyplot as plt

from .config import CH_NAMES, TARGET_FREQS, FS, STIM_ONSET
from scipy.fft import rfft, rfftfreq


# loadding Raw data
def load_subject(filepath):
    mat = loadmat(filepath)
    return mat["eeg"]


# preprocessing data


def preprocess_trial(trial, fs=256):
    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(trial, info, verbose=False)
    raw.filter(8.0, 40.0, verbose=False)
    return raw.get_data()


# computing FFT


def fft_scores(trial, fs=FS):
    freqs = rfftfreq(trial.shape[1], d=1 / fs)
    fft_vals = abs(rfft(trial, axis=1))

    scores = []
    for f in TARGET_FREQS:
        idx = np.argmin(abs(freqs - f))
        score = fft_vals[:, idx].mean()
        scores.append(score)

    return np.array(scores)


def compute_fft(trial, fs=256):
    """
    trial: (channels, samples)

    returns:
    - freqs: frequency bins
    - fft_vals: (channels, freq_bins)
    """
    freqs = rfftfreq(trial.shape[1], d=1 / fs)
    fft_vals = np.abs(rfft(trial, axis=1))

    return freqs, fft_vals


# Predict the target frequency with the highest FFT score


def predict_fft(trial, fs=256):
    scores = fft_scores(trial, fs)
    return int(np.argmax(scores))


# Evaluate the model on a subject's data


def evaluate_subject(eeg):
    correct = 0
    total = 0

    for target in range(12):
        for trial_idx in range(15):
            trial = eeg[target, :, STIM_ONSET:, trial_idx]

            trial = preprocess_trial(trial)

            pred = predict_fft(trial)

            if pred == target:
                correct += 1

            total += 1

    return correct / total


# Plotting FFT spectra


def plot_fft_all_channels(
    trial,
    fs=256,
    ch_names=None,
    target_freq=None,
    target_freqs=None,
    normalize=False,
    xlim=(5, 20),
):
    freqs, fft_vals = compute_fft(trial, fs)

    plt.figure(figsize=(10, 5))

    for ch_idx in range(fft_vals.shape[0]):
        spectrum = fft_vals[ch_idx]

        if normalize:
            spectrum = spectrum / spectrum.max()

        label = ch_names[ch_idx] if ch_names else f"ch{ch_idx}"
        plt.plot(freqs, spectrum, label=label, alpha=0.7)

    if target_freqs is not None:
        for f in target_freqs:
            plt.axvline(f, linestyle="--", alpha=0.2)

    if target_freq is not None:
        plt.axvline(
            target_freq, color="red", linewidth=2, label=f"True: {target_freq} Hz"
        )

    plt.xlim(*xlim)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude" if not normalize else "Normalized")
    plt.title("FFT - All Channels")

    plt.legend()
    plt.tight_layout()
    plt.show()
