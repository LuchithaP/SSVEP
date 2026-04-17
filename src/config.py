import numpy as np

FS = 256
STIM_ONSET = 39

WINDOW_SEC = 1
WINDOW_SAMPLES = int(WINDOW_SEC * FS)


CH_NAMES = ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]

TARGET_FREQS = np.array(
    [9.25, 11.25, 13.25, 9.75, 11.75, 13.75, 10.25, 12.25, 14.25, 10.75, 12.75, 14.75]
)
