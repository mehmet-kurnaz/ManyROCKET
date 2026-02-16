# MANYROCKET  
## Adaptive Parallel MiniRocket for Cross-Representation Time Series Classification

MANYROCKET is an adaptive, parallelized extension of MiniRocket designed for scalable and reproducible time series classification.

It introduces:

- Representation-level parallelism  
- MiniRocket kernel-level parallelism  
- Multi-representation feature extraction  
- Full timing + performance logging  

Unlike vanilla MiniRocket, MANYROCKET applies MiniRocket across multiple signal representations and concatenates their features, enabling richer discriminative power while maintaining computational efficiency.

---

## 🚀 Key Features

- 🔁 Multi-representation MiniRocket (TIME, FFT, DWT, DCT, Hilbert, Derivatives)
- ⚡ Adaptive CPU allocation between representation workers and kernel workers
- ⏱️ Detailed timing analysis (transform, classifier, combo, dataset)
---

## 🧠 Core Idea
Different signal representations capture different invariances:
- Time domain → shape information  
- Frequency domain → periodic structure  
- Wavelet domain → multi-resolution localization  
- Hilbert envelope → instantaneous amplitude  
- Derivatives → local dynamics  
Instead of committing to a single representation, MANYROCKET allows MiniRocket to operate on multiple representations in parallel.

---


