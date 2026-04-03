import numpy as np
from .features import fft_scores


def predict_fft(trial, fs=256):
    scores = fft_scores(trial, fs)
    return int(np.argmax(scores))
