from .features import compute_fft
import matplotlib.pyplot as plt


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
