from scipy.io import loadmat


def load_subject(filepath):
    mat = loadmat(filepath)
    return mat["eeg"]
