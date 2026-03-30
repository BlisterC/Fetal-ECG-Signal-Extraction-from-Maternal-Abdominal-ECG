# Fetal ECG Extraction using ICA + EMD + Beat Gating

## 📌 Project Overview

This project focuses on extracting the **Fetal ECG (fECG)** signal from **non-invasive abdominal ECG recordings** using a structured signal processing pipeline. The objective is to isolate the weak fetal cardiac signal from dominant maternal ECG and noise components.

The implementation is done in **MATLAB**, following a classical signal processing approach.

---

## ⚙️ Methodology

The pipeline consists of five major stages:

### 1. Preprocessing

* DC offset removal
* High-pass filtering (0.5 Hz)
* Bandpass filtering (0.5–100 Hz)
* Signal normalization

### 2. QRS-Band ICA (FastICA)

* Abdominal signals filtered in **14–40 Hz (QRS band)**
* Multi-seed FastICA applied
* Best independent component selected using **correlation with reference (evaluation only)**

### 3. EMD-Based Refinement

* Empirical Mode Decomposition (EMD) applied to ICA output
* Top IMFs selected based on correlation
* Weighted reconstruction of fetal ECG

### 4. Beat Gating

* Fetal beats detected from extracted signal
* Inter-beat regions suppressed using gating mask
* Enhances QRS visibility

### 5. Post-Processing

* Final bandpass filtering (8–50 Hz)
* Signal normalization

---

## 📊 Results

* Initial correlation: ~0.59
* Final correlation achieved: **~0.78 – 0.83**
* Clear extraction of fetal QRS complexes observed

---

## 📁 Dataset

* **PhysioNet Abdominal and Direct Fetal ECG Database**
* Contains:

  * 4 abdominal ECG channels
  * Direct fetal ECG (reference for evaluation only)

---

## 📈 Outputs

The code generates:

* Final extracted fECG signal vs reference
* Zoomed signal comparison
* Beat detection visualization
* EMD IMF decomposition plots
* Correlation improvement graph

---

## 🚀 How to Run

1. Download dataset (`r01.mat`)
2. Place it in the project directory
3. Install FastICA (required for MATLAB)
4. Run the MATLAB script:

```matlab
run('your_script_name.m')
```

---

## ⚠️ Important Notes

* The reference fetal ECG is used **only for evaluation**, not for signal reconstruction
* The pipeline follows a **classical signal processing approach**, not deep learning
* Performance depends on dataset quality and ICA convergence

---


## 📚 References

Key techniques based on:

* Independent Component Analysis (ICA)
* Empirical Mode Decomposition (EMD)
* Non-invasive fetal ECG extraction literature

---

## 📌 Conclusion

This project demonstrates that **non-invasive fetal ECG extraction** is feasible using classical signal processing techniques. The combination of **ICA + EMD + gating** significantly improves signal quality and fetal heartbeat visibility.

---
