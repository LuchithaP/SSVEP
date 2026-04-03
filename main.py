from src.fft_helpers_2 import load_subject, evaluate_subject

for i in range(1, 11):
    eeg = load_subject(f"data/s{i}.mat")
    acc = evaluate_subject(eeg)
    print(f"s{i}: {acc:.3f}")
