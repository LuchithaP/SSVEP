from .preprocess import preprocess_trial
from .classifier import predict_fft

STIM_ONSET = 39


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
