import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from scipy.signal import detrend
from matplotlib import pyplot as plt

from .config import CH_NAMES, TARGET_FREQS, FS, STIM_ONSET, WINDOW_SAMPLES, WINDOW_SEC


# ==============================
# LOAD DATA
# ==============================
def load_subject(filepath):
    """
    Input: filepath to .mat file containing EEG data
    Output: numpy array of shape (targets, channels, samples, trials)

    """
    mat = loadmat(filepath)
    return mat["eeg"]  # (targets, channels, samples, trials)


# ==============================
# PREPROCESSING
# ==============================
def bandpass_filter(data, fs, low=8, high=40, order=4):
    """
    Apply a Butterworth bandpass filter to the data.
    Input:
        data: numpy array of shape (channels, samples)
        fs: sampling frequency
        low: low cutoff frequency (Hz)
        high: high cutoff frequency (Hz)
        order: filter order
    Output:
        filtered_data: numpy array of shape (channels, samples)
    """
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data, axis=1)


def preprocess_trial(trial, fs=FS):
    """
    Preprocess a single trial of EEG data.
    Input:
        trial: numpy array of shape (channels, samples)
    fs: sampling frequency
    Output:
        processed_trial: numpy array of shape (channels, samples)
    """

    # Detrending can help remove slow drifts and DC offsets, which can improve FFT results
    trial = detrend(trial, axis=1)

    # Average reference
    trial = trial - np.mean(trial, axis=0, keepdims=True)

    # Bandpass filter
    trial = bandpass_filter(trial, fs)

    return trial


# ==============================
# PRECOMPUTE FFT BINS
# ==============================
def precompute_bins(n_samples, fs, window_hz=0.3):
    """
    Precompute the FFT bins corresponding to the target frequencies and their harmonics.
    Input:
        n_samples: number of samples in the trial (after stimulus onset)
        fs: sampling frequency
        window_hz: frequency window around each target frequency (Hz)
    Output:
        bins: list of tuples (band1, band2) where:
        band1: indices of FFT bins around the target frequency
        band2: indices of FFT bins around the second harmonic

    """
    freqs = rfftfreq(n_samples, d=1 / fs)
    bins = []

    for f in TARGET_FREQS:
        band1 = np.where((freqs >= f - window_hz) & (freqs <= f + window_hz))[0]
        band2 = np.where((freqs >= 2 * f - window_hz) & (freqs <= 2 * f + window_hz))[0]
        band3 = np.where((freqs >= 3 * f - window_hz) & (freqs <= 3 * f + window_hz))[0]
        bins.append((band1, band2, band3))

    return bins


# ==============================
# SNR-BASED SCORING
# ==============================


def snr_score(spectrum, idx, guard=2, noise_width=6):
    """
    Compute a simple SNR score for the target frequency bins.
    Input:
        spectrum: 1D numpy array of FFT magnitudes
        idx: indices of FFT bins corresponding to the target frequency
        noise_width: number of bins on either side to consider for noise estimation
        guard: number of bins to exclude around the target frequency
    Output:
        score: SNR score (signal / noise)
    """
    if len(idx) == 0:
        return 0

    signal = spectrum[idx].mean()

    # Exclude immediate neighbors (guard band)
    left = np.arange(idx[0] - guard - noise_width, idx[0] - guard)
    right = np.arange(idx[-1] + guard + 1, idx[-1] + guard + 1 + noise_width)

    noise_idx = np.concatenate([left, right])
    noise_idx = noise_idx[(noise_idx >= 0) & (noise_idx < len(spectrum))]

    if len(noise_idx) == 0:
        return signal

    noise = spectrum[noise_idx].mean()

    return signal - noise  # log-space → subtraction is better than division


def fft_scores(trial, bins):
    """
    Compute SNR-based scores for each target frequency based on the FFT spectrum.
    Input:
        trial: numpy array of shape (channels, samples) - preprocessed EEG trial
        bins: list of tuples (band1, band2) where:
        band1: indices of FFT bins around the target frequency
        band2: indices of FFT bins around the second harmonic
    Output:
        scores: numpy array of shape (n_targets,) containing the SNR scores for each target frequency

    """
    # FFT
    fft_vals = np.abs(rfft(trial, axis=1))

    # Channel weighting : compute variance across time for each channel and use it as a weight
    # why ? : channels with higher variance are likely to contain more information related to the stimulus, while channels with low variance may be dominated by noise. By weighting the FFT values based on channel variance, we can enhance the contribution of informative channels and suppress noisy ones, leading to more robust SNR scores for frequency detection.

    weights = np.var(trial, axis=1)
    weights = weights / (weights.sum() + 1e-8)

    spectrum = np.sum(fft_vals * weights[:, None], axis=0)

    # Log spectrum (robust)
    # why? : The FFT spectrum can have a very wide dynamic range, with some frequencies having much higher magnitudes than others. Taking the logarithm of the spectrum helps to compress this dynamic range, making it easier to compare the relative strengths of different frequency components. Additionally, the log transformation can help to stabilize variance and make the SNR scores more robust to outliers and noise in the data.

    spectrum = np.log(spectrum + 1e-8)

    scores = []

    # Score each frequency
    # reason : The fundamental frequency (band1) is typically the strongest response in SSVEP, so it is given the highest weight in the scoring. The second harmonic (band2) can also contain significant information, especially if the fundamental frequency response is weak or noisy, so it is weighted at 0.7. The third harmonic (band3) may contain some information but is generally weaker than the first two, so it is weighted at 0.4. This weighting scheme allows us to leverage information from multiple harmonics while still prioritizing the most informative frequency components for accurate target frequency detection.
    for band1, band2, band3 in bins:
        score = 0

        # Fundamental frequency
        score += 1.0 * snr_score(spectrum, band1)

        # 2nd harmonic
        if len(band2):
            score += 0.6 * snr_score(spectrum, band2)

        # 3rd harmonic
        if len(band3):
            score += 0.2 * snr_score(spectrum, band3)

        scores.append(score)

    return np.array(scores)


# ==============================
# PREDICTION
# ==============================
def predict_fft(trial, bins, return_scores=False):
    """
    Predict the target frequency for a single trial based on FFT scores.
    Input:
        trial: numpy array of shape (channels, samples) - preprocessed EEG trial
        bins: precomputed FFT bins for target frequencies
        return_scores: if True, also return the raw scores for each target frequency
    Output:
        pred: predicted target frequency index
        confidence: gap between best and second best score (higher is more confident)
        scores (optional): raw scores for each target frequency
    """
    scores = fft_scores(trial, bins)

    pred = int(np.argmax(scores))

    # Confidence = gap between best and second best
    sorted_scores = np.sort(scores)
    confidence = sorted_scores[-1] - sorted_scores[-2]

    if return_scores:
        return pred, confidence, scores

    return pred


# ==============================
# EVALUATION
# ==============================
def evaluate_subject(eeg, reject_threshold=0.1):
    """
    Evaluate the accuracy of FFT-based predictions on the given EEG data.

    Input:
        eeg: numpy array of shape (targets, channels, samples, trials)
        reject_threshold: minimum confidence required to accept a prediction

    Output:
        acc: overall accuracy of predictions
    """

    correct = 0
    total = 0
    rejected = 0

    n_classes = len(TARGET_FREQS)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    n_samples = eeg.shape[2] - STIM_ONSET
    bins = precompute_bins(n_samples, FS)

    for target in range(n_classes):
        trials = eeg[target]

        for trial_idx in range(trials.shape[-1]):
            start = STIM_ONSET
            end = start + WINDOW_SAMPLES

            if end > trials.shape[1]:
                continue

            trial = trials[:, start:end, trial_idx]

            trial = preprocess_trial(trial)

            pred = predict_fft(trial, bins, return_scores=False)

            # -----------------------------
            # Update metrics
            # -----------------------------
            confusion[target, pred] += 1

            if pred == target:
                correct += 1

            total += 1

    # -----------------------------
    # Accuracy
    # -----------------------------
    acc = correct / (total + 1e-8)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Rejected trials: {rejected}")

    # -----------------------------
    # Per-class accuracy
    # -----------------------------
    print("\nPer-class accuracy:")
    for i in range(n_classes):
        class_total = confusion[i].sum()
        class_acc = confusion[i, i] / (class_total + 1e-8)
        print(f"Class {i} ({TARGET_FREQS[i]} Hz): {class_acc:.3f}")

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    print("\nConfusion Matrix:")
    print(confusion)

    return acc


# ==============================
# FFT VISUALIZATION
# ==============================
def compute_fft(trial, fs=FS):
    freqs = rfftfreq(trial.shape[1], d=1 / fs)
    fft_vals = np.abs(rfft(trial, axis=1))
    return freqs, fft_vals


def plot_fft_all_channels(
    trial,
    fs=FS,
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
            spectrum = spectrum / (spectrum.max() + 1e-8)

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
