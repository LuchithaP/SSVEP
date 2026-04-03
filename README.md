# 🧠 SSVEP-Based EEG Classification (Beginner Project)

This project implements a **Steady-State Visual Evoked Potential (SSVEP)** classification pipeline using EEG data.
The goal is to identify which visual stimulus frequency a subject is focusing on by analyzing EEG signals.

---

## 📌 Project Overview

When a person looks at a flickering visual stimulus (e.g., 10 Hz), their brain produces electrical activity at the same frequency.

This project:

* Loads EEG data from a public SSVEP dataset
* Preprocesses the signals using MNE
* Extracts frequency features using FFT
* Classifies the attended stimulus frequency
* Evaluates classification accuracy

---

## 🧠 Key Idea

👉 EEG signals are recorded in the **time domain**
👉 SSVEP information exists in the **frequency domain**

We use **FFT (Fast Fourier Transform)** to convert:

```
Time signal → Frequency spectrum
```

Then we detect which frequency is strongest.

---


## ⚙️ Dataset

We use the **Nakanishi 12-class SSVEP dataset**.

### Data shape:

```
(12, 8, 1114, 15)
```

* 12 targets (frequencies)
* 8 EEG channels
* 1114 samples per trial
* 15 trials per target

### Target frequencies:

```
9.25 – 14.75 Hz (step = 0.5 Hz)
```

---

## 🏗️ Pipeline

### 1. Load Data

```python
eeg = load_subject("data/s1.mat")
```

### 2. Preprocessing

* Remove pre-stimulus samples
* Bandpass filter (8–40 Hz)
* Optional: average reference

### 3. Feature Extraction (FFT)

We compute:

```
FFT → frequency spectrum
```

This reveals peaks at stimulus frequencies.

### 4. Classification

For each trial:

* Measure power near each target frequency
* Select the frequency with highest score

### 5. Evaluation

Accuracy is computed as:

```
accuracy = correct_predictions / total_trials
```

---

## 📊 Example Output

* FFT plots showing peaks at stimulus frequencies
* Predicted vs true frequency
* Subject-wise accuracy
* Confusion matrix

---


## 🔍 What is Being Evaluated?

We are evaluating:

👉 **Can the system correctly identify the stimulus frequency from EEG?**

Each trial has a known label (target index), and the model predicts one of 12 classes.

---

## 📈 Baseline Method

We use a simple **FFT-based classifier**:

* compute spectral power
* match to known stimulus frequencies

---

## ⚠️ Limitations

* FFT-only approach is basic
* sensitive to noise
* ignores phase information

---

## 🔥 Future Improvements

* Add **harmonics (2f, 3f)**
* Implement **CCA (Canonical Correlation Analysis)**
* Use **Filter Bank CCA (FBCCA)**
* Try **deep learning models (EEGNet, CNNs)**

---

## 🎯 Learning Goals

This project helps you understand:

* EEG signal structure
* Time vs frequency domain
* SSVEP mechanism
* Basic BCI pipeline
* Signal processing with Python

---

## 🧠 Key Takeaway

👉 The brain responds at the same frequency as the visual stimulus
👉 FFT helps us detect that frequency
👉 Classification = identifying the strongest frequency

---

## 📚 References

* Nakanishi et al., 2015 – SSVEP detection using CCA
* MNE-Python documentation
* BCI literature on SSVEP

---

