import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, detrend
from .config import FS, TARGET_FREQS, STIM_ONSET, WINDOW_SEC, WINDOW_SAMPLES, CH_NAMES


# ==============================
# LOAD DATA
# ==============================
def load_subject(filepath):
    mat = loadmat(filepath)
    return mat["eeg"]  # (targets, channels, samples, trials)


# ==============================
# PREPROCESSING
# ==============================
def bandpass_filter(data, fs, low=6, high=50, order=4):
    b, a = butter(order, [low / (fs / 2), high / (fs / 2)], btype="band")
    return filtfilt(b, a, data, axis=1)


def preprocess_trial(trial, fs=FS):
    # Detrend
    trial = detrend(trial, axis=1)

    # Average reference
    trial = trial - np.mean(trial, axis=0, keepdims=True)

    # Bandpass
    trial = bandpass_filter(trial, fs)

    return trial


# ==============================
# REFERENCE SIGNALS
# ==============================
def generate_reference_signals(freqs, n_samples, fs, n_harmonics=3):
    t = np.arange(n_samples) / fs
    refs = []

    for f in freqs:
        ref = []
        for h in range(1, n_harmonics + 1):
            ref.append(np.sin(2 * np.pi * h * f * t))
            ref.append(np.cos(2 * np.pi * h * f * t))
        refs.append(np.array(ref))  # (2*harmonics, samples)

    return refs


# ==============================
# CCA CORE
# ==============================
def cca_score(X, Y):
    """
    X: (channels, samples)
    Y: (refs, samples)
    """
    X = X.T
    Y = Y.T

    # Zero-mean
    X = X - X.mean(axis=0)
    Y = Y - Y.mean(axis=0)

    # Covariance matrices
    Sxx = X.T @ X
    Syy = Y.T @ Y
    Sxy = X.T @ Y

    # Regularization
    reg = 1e-6
    Sxx += reg * np.eye(Sxx.shape[0])
    Syy += reg * np.eye(Syy.shape[0])

    # Solve eigen problem
    invSxx = np.linalg.inv(Sxx)
    invSyy = np.linalg.inv(Syy)

    M = invSxx @ Sxy @ invSyy @ Sxy.T
    eigvals = np.linalg.eigvals(M)

    return np.sqrt(np.max(np.real(eigvals)))


# ==============================
# PREDICTION
# ==============================
def predict_cca(trial, refs):
    scores = []

    for ref in refs:
        score = cca_score(trial, ref)
        scores.append(score)

    scores = np.array(scores)

    pred = int(np.argmax(scores))

    # Confidence
    sorted_scores = np.sort(scores)
    confidence = sorted_scores[-1] - sorted_scores[-2]

    return pred, confidence, scores


# ==============================
# EVALUATION
# ==============================
def evaluate_subject_cca(eeg, use_channels=None):
    correct = 0
    total = 0

    n_classes = len(TARGET_FREQS)
    confusion = np.zeros((n_classes, n_classes), dtype=int)

    refs = generate_reference_signals(TARGET_FREQS, WINDOW_SAMPLES, FS)

    # Convert channel names to indices if needed
    if (
        use_channels is not None
        and len(use_channels) > 0
        and isinstance(use_channels[0], str)
    ):
        use_channels = np.array([CH_NAMES.index(ch) for ch in use_channels])

    for target in range(n_classes):
        trials = eeg[target]

        for trial_idx in range(trials.shape[-1]):
            start = STIM_ONSET
            end = start + WINDOW_SAMPLES

            if end > trials.shape[1]:
                continue

            trial = trials[:, start:end, trial_idx]

            if use_channels is not None:
                trial = trial[use_channels, :]

            trial = preprocess_trial(trial)

            pred, conf, scores = predict_cca(trial, refs)

            confusion[target, pred] += 1

            if pred == target:
                correct += 1

            total += 1

    acc = correct / (total + 1e-8)

    per_class_acc = np.zeros(n_classes)
    for i in range(n_classes):
        class_total = confusion[i].sum()
        per_class_acc[i] = confusion[i, i] / (class_total + 1e-8)

    return acc, per_class_acc, confusion
