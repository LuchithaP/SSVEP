from src.load_data import load_subject
from src.evaluate import evaluate_subject

for i in range(1, 11):
    eeg = load_subject(f"data/s{i}.mat")
    acc = evaluate_subject(eeg)
    print(f"s{i}: {acc:.3f}")
