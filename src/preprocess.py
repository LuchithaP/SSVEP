import mne

CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]


def preprocess_trial(trial, fs=256):
    info = mne.create_info(ch_names=CH_NAMES, sfreq=fs, ch_types="eeg")
    raw = mne.io.RawArray(trial, info, verbose=False)
    raw.filter(8.0, 40.0, verbose=False)
    return raw.get_data()
