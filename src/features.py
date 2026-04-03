import numpy as np
from scipy.fft import rfft, rfftfreq

TARGET_FREQS = np.array(
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
)


def fft_scores(trial, fs=256):
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
