# Multi-Modal Dysarthria Severity Classification from Speech

> **SAND Task 1 — Speech-based Dysarthria Severity Classification**

This project develops and evaluates multiple machine-learning and deep-learning pipelines for **five-class dysarthria severity classification from speech recordings**. The implementation explores handcrafted acoustic/clinical features, self-supervised speech representations from **Wav2Vec 2.0**, spectrogram-based **Vision Transformers (ViT)**, feature-level fusion, oversampling strategies, and an experimental transformer sequence model.

The notebooks contain both the main experimental pipeline and several exploratory/alternative approaches.

---

## Project Overview

The objective is to classify subjects into five dysarthria-related classes using speech recordings collected across multiple speech tasks.

The project follows a **subject-level learning setup**:

1. Load speech metadata and training recordings.
2. Standardize the audio to a common format.
3. Build a normalized manifest linking recordings, subjects, tasks, and labels.
4. Extract handcrafted acoustic, articulatory, phonatory, and temporal features.
5. Extract **1024-dimensional Wav2Vec 2.0 embeddings**.
6. Aggregate recording-level features to the subject level.
7. Train and compare several classifiers.
8. Explore multimodal fusion of handcrafted and deep speech features.
9. Convert speech into spectrograms and train a Vision Transformer.
10. Experiment with sequence modeling and three-class severity grouping.

---

## Dataset

The notebooks process the SAND Task 1 training data.

### Dataset scale

| Level | Count |
|---|---:|
| Speech recordings | **2,176** |
| Unique subjects | **272** |
| Classes | **5** |

The recording-level class distribution is:

| Class | Recordings |
|---|---:|
| 1 | 48 |
| 2 | 208 |
| 3 | 456 |
| 4 | 608 |
| 5 | 856 |

After subject-level aggregation, the class distribution becomes:

| Class | Subjects |
|---|---:|
| 1 | 6 |
| 2 | 26 |
| 3 | 57 |
| 4 | 76 |
| 5 | 107 |

The extreme imbalance at the subject level is an important challenge, particularly for Class 1.

### Class semantics

The model/evaluation notebooks use the following mapping:

| Label | Meaning |
|---|---|
| **1** | Severe |
| **2** | Moderate |
| **3** | Mild |
| **4** | No Dysarthria |
| **5** | Healthy |

---

## Audio Preprocessing

The preprocessing pipeline standardizes the speech recordings before feature extraction.

### Standardization

- Resampling to **8 kHz**
- Conversion to **mono**
- Encoding as **PCM-16**
- Duration calculation
- SHA-256 checksums for processed files
- Manifest generation

### Voice Activity Detection

An energy-based VAD pipeline is implemented using:

- RMS energy
- Median + MAD based thresholding
- Short-silence filling
- Short-speech-segment removal
- Speech interval extraction

The following timing statistics are generated:

- Speech duration
- Speech ratio
- Number of speech segments
- Mean speech-segment duration
- Mean silence between segments

### RMS normalization

Audio is additionally normalized toward a **−20 dBFS RMS target**, with a peak-clipping safeguard.

---

# Feature Extraction

## 1. Handcrafted Acoustic / Clinical Features

The handcrafted pipeline combines multiple groups of speech features.

### Articulatory features

The project includes features derived from MFCC/VSA/FCR-related processing.

### Phonatory features

The phonatory pipeline extracts:

- **Jitter**
- **Shimmer**
- **CPPS**
- **HNR**
- **L/H spectral ratio**

Jitter, Shimmer and CPPS are extracted using Praat/Praat-Parselmouth based processing, while HNR and L/H ratio are computed using Librosa.

### Temporal features

For DDK speech tasks, the project extracts:

- **DDK Rate**
- **DDK Coefficient of Variation**
- **Percentage Pause**
- **F0 Range**
- **Intensity Range**

DDK tasks are identified from the `rhythmKA`, `rhythmPA`, and `rhythmTA` task names.

### Subject-level aggregation

Recording-level handcrafted features are aggregated by subject using mean statistics.

The resulting master handcrafted feature dataset contains:

- **272 subjects**
- **88 features**

Saved as:

```text
Train/task1/Task1_MASTER_FEATURES.csv
```

---

# 2. Wav2Vec 2.0 Features

A second pipeline uses self-supervised speech representations.

### Model

```text
facebook/wav2vec2-xls-r-300m
```

The model is used as a feature extractor rather than as an end-to-end classifier.

### Extraction process

For each recording:

1. Load audio using Librosa.
2. Resample to the model's expected sampling rate.
3. Process the waveform using the Wav2Vec 2.0 feature extractor.
4. Pass the waveform through the pretrained model.
5. Obtain the last hidden-state representation.
6. Mean-pool the time dimension.
7. Store the resulting **1024-dimensional embedding**.

Recording-level embeddings are subsequently aggregated by subject.

Output:

```text
Train/task1/wav2vec_audio_features.csv
Train/task1/Task1_WAV2VEC_MASTER_FEATURES.csv
```

The subject-level Wav2Vec dataset contains:

```text
272 subjects × 1024 embedding features
```

---

# Modeling Approaches

## A. Wav2Vec 2.0 + XGBoost

A baseline classifier is trained on the 1024-dimensional subject-level Wav2Vec embeddings.

Configuration includes:

- RobustScaler
- XGBoost
- 5-fold Stratified Cross-Validation
- Macro F1 as the primary metric

Recorded result:

```text
Mean Macro F1: 0.3343 ± 0.0799
```

Fold scores:

```text
0.3592
0.2470
0.2490
0.3561
0.4604
```

---

## B. Wav2Vec 2.0 + PCA + Oversampling

To address the high-dimensional / low-sample setting, the project experiments with:

```text
RobustScaler
      ↓
PCA (95% variance)
      ↓
Random Oversampling
      ↓
Classifier
```

Both XGBoost and RBF-SVM are evaluated.

Recorded results:

| Model | Mean Macro F1 |
|---|---:|
| XGBoost + PCA + Oversampling | **0.3456 ± 0.1267** |
| SVM + PCA + Oversampling | **0.3523 ± 0.0708** |

---

## C. Handcrafted Features

The handcrafted feature pipeline evaluates conventional machine-learning models including:

- Random Forest
- XGBoost
- SVM
- Voting ensembles

Recorded cross-validation results include:

```text
Random Forest: 0.3800 ± 0.0836
XGBoost:       0.3310 ± 0.0702
```

---

# 3. SMOTE / Oversampling Experiments

Because the subject-level dataset contains only **6 samples in Class 1**, multiple imbalance-handling strategies were investigated.

Evaluated methods include:

- Standard SMOTE
- BorderlineSMOTE
- Random Oversampling

Recorded Wav2Vec + XGBoost results:

| Sampling Strategy | Macro F1 |
|---|---:|
| Standard SMOTE | **0.3638 ± 0.0892** |
| BorderlineSMOTE | **0.3855 ± 0.1123** |
| Random Oversampling | **0.3439 ± 0.0717** |

---

# 4. Multimodal Feature Fusion

The project combines handcrafted and Wav2Vec features.

### Feature dimensions

```text
Handcrafted features : 40 columns
Wav2Vec features     : 1026 columns
Fused dataset        : 1064 columns
Subjects             : 272
```

A soft-voting ensemble combines:

- RBF-SVM
- Random Forest
- XGBoost

Pipeline:

```text
Handcrafted Features
        +
Wav2Vec Embeddings
        ↓
Feature Fusion
        ↓
RobustScaler
        ↓
PCA
        ↓
Random Oversampling
        ↓
SVM + Random Forest + XGBoost
        ↓
Soft Voting
```

Recorded 5-fold result:

```text
Macro F1: 0.4031 ± 0.0671
```

Fold scores:

```text
0.4588
0.3050
0.3391
0.4615
0.4511
```

A separate experiment also compares:

```text
Clinical-only
Wav2Vec-only
Smart Fusion
```

with recorded mean F1 values of:

```text
Clinical-only : 0.3245 ± 0.0747
Wav2Vec-only  : 0.3523 ± 0.0663
Smart Fusion  : 0.3506 ± 0.0746
```

---

# 5. Spectrogram + Vision Transformer

The project also explores treating speech as an image classification problem.

### Pipeline

```text
Speech Audio
     ↓
Spectrogram
     ↓
Image preprocessing / resizing
     ↓
ViT Base Patch16-224
     ↓
5-class classification
```

Model:

```text
google/vit-base-patch16-224
```

The classifier head is adapted for five classes:

```text
Severe
Moderate
Mild
No Dys
Healthy
```

### Cross-validation

A subject-level 5-fold split is used so that recordings from the same subject do not appear across training and validation folds.

Recorded fold Macro F1:

```text
Fold 1 : 0.4329
Fold 2 : 0.2934
Fold 3 : 0.3249
Fold 4 : 0.3625
Fold 5 : 0.4438
```

Recorded average:

```text
Macro F1: 0.3715 ± 0.0589
```

The notebooks also contain an earlier fold-specific evaluation of a ViT checkpoint on the full recording set. That evaluation reports an accuracy of **0.8304**, but it is not directly comparable to the subject-level 5-fold evaluation and should not be used as the main cross-validation result.

---

# 6. ALST-Style Sequence Model

An experimental transformer architecture is also implemented to model multiple recordings belonging to the same subject.

### Input

Each subject is represented as a sequence:

```text
Number of recordings × 1024 Wav2Vec features
```

Because subjects can have different numbers of recordings, sequences are padded within each batch.

### Architecture

```text
1024-D Wav2Vec embeddings
          ↓
Linear Projection
          ↓
512-D hidden representation
          ↓
Positional Embedding
          ↓
2-layer Transformer Encoder
          ↓
Pooling
       ↙     ↘
Classification  Regression
   5 classes    Severity score
```

Configuration:

```text
Hidden dimension : 512
Transformer layers : 2
Attention heads : 4
Dropout : 0.1
Batch size : 8
Epochs : 20
Learning rate : 1e-4
```

The model is experimental and supports both classification and regression objectives.

---

# 7. Three-Class Severity Grouping

An additional experiment reduces the five-class problem to three clinically broader groups.

### Mapping

```text
Class 5       → Healthy
Classes 3, 4  → Mild / Early
Classes 1, 2  → Severe / Pathological
```

Resulting distribution:

```text
Healthy             : 107
Mild / Early        : 133
Severe / Pathological : 32
```

The experiment uses:

```text
BorderlineSMOTE
      ↓
XGBoost
```

Recorded test-set results:

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Healthy | 0.61 | 0.64 | 0.62 |
| Mild/Early | 0.59 | 0.48 | 0.53 |
| Severe/Pathological | 0.40 | 0.67 | 0.50 |

Overall:

```text
Accuracy  : 0.56
Macro F1  : 0.55
```

SHAP and feature-importance visualizations are also generated for interpretability.

---

# Evaluation Strategy

The project primarily uses:

- **Stratified K-Fold Cross-Validation**
- **5 folds**
- **Random seed = 42**
- **Macro F1-score**

Macro F1 is emphasized because of the severe class imbalance.

For image-based modeling, folds are created at the **subject ID level**, preventing recordings from the same subject from being distributed across training and validation sets.

---

# Project Structure

```text
.
├── .gitattributes
│
├── Data_Loading.ipynb
│   └── Data loading, cleaning, VAD and normalization
│
├── Manual.ipynb
│   └── Handcrafted acoustic / clinical feature extraction
│
├── Wav2Vec.ipynb
│   └── Wav2Vec 2.0 feature extraction and classification
│
├── Manual + Wav2vec.ipynb
│   └── Multimodal feature fusion experiments
│
├── five_to_three.ipynb
│   └── Three-class severity grouping + XGBoost + SHAP
│
└── all_combined.ipynb
    └── Consolidated experiments including
        Wav2Vec, handcrafted features,
        spectrogram + ViT, ALST and fusion
```

Generated data/model directories referenced by the notebooks include:

```text
Train/task1/
├── training/
├── cleaned_wavs/
├── normalized_wavs/
├── manifest.csv
├── manifest_normalized.csv
├── phonatory_features_parallel.csv
├── phonatory_features_completed.csv
├── ddk_features.csv
├── final_features_task1.csv
├── Task1_MASTER_FEATURES.csv
├── wav2vec_audio_features.csv
├── Task1_WAV2VEC_MASTER_FEATURES.csv
├── spectrograms_vit/
├── vit_results/
├── xgb_top_features.csv
└── xgb_tuned_model.json
```

These generated files are not included in the uploaded notebook set and need to be produced from the source dataset.

---

# Installation

Create a Python environment and install the required dependencies.

```bash
pip install pandas numpy scipy scikit-learn
pip install librosa soundfile matplotlib seaborn tqdm openpyxl
pip install torch torchaudio torchvision
pip install transformers datasets accelerate
pip install xgboost imbalanced-learn
pip install opensmile
pip install praat-parselmouth
pip install shap
```

Depending on the notebook and hardware environment, additional packages may be required.

---

# Running the Project

## 1. Prepare the data

Start with:

```text
Data_Loading.ipynb
```

This creates the cleaned and normalized audio manifests.

## 2. Extract handcrafted features

Run:

```text
Manual.ipynb
```

This produces the handcrafted feature datasets and the final subject-level master feature table.

## 3. Extract Wav2Vec features

Run:

```text
Wav2Vec.ipynb
```

This generates recording-level and subject-level Wav2Vec embeddings.

## 4. Run fusion experiments

Run:

```text
Manual + Wav2vec.ipynb
```

to evaluate multimodal feature fusion.

## 5. Run the three-class experiment

Run:

```text
five_to_three.ipynb
```

to reproduce the three-class severity grouping, XGBoost classification, feature importance and SHAP analysis.

## 6. Run consolidated experiments

Use:

```text
all_combined.ipynb
```

for the consolidated Wav2Vec, handcrafted, fusion, spectrogram/ViT and ALST experiments.

---

# Main Results

The following results are directly recorded in the notebooks:

| Approach | Evaluation | Macro F1 |
|---|---|---:|
| Wav2Vec + XGBoost | 5-fold CV | **0.3343 ± 0.0799** |
| Wav2Vec + XGBoost + PCA + Oversampling | 5-fold CV | **0.3456 ± 0.1267** |
| Wav2Vec + SVM + PCA + Oversampling | 5-fold CV | **0.3523 ± 0.0708** |
| Handcrafted + Random Forest | 5-fold CV | **0.3800 ± 0.0836** |
| Handcrafted + XGBoost | 5-fold CV | **0.3310 ± 0.0702** |
| Wav2Vec + BorderlineSMOTE + XGBoost | 5-fold CV | **0.3855 ± 0.1123** |
| Handcrafted + Wav2Vec Fusion Ensemble | 5-fold CV | **0.4031 ± 0.0671** |
| ViT on Spectrograms | 5-fold CV | **0.3715 ± 0.0589** |
| Three-class XGBoost | Test split | **0.55** |

> **Note:** These experiments use different feature sets, splits, preprocessing pipelines and evaluation procedures. Results should therefore be interpreted as experiment-specific rather than as a single directly comparable leaderboard.

---

# Key Challenges

### Severe class imbalance

Only **6 subjects** belong to Class 1 compared with **107 subjects** in Class 5.

This motivates experiments with:

- Random Oversampling
- SMOTE
- BorderlineSMOTE
- Class weighting
- PCA
- Macro-F1-based evaluation

### High-dimensional representations

Wav2Vec produces **1024-dimensional embeddings**, while multimodal fusion increases the feature space further.

PCA and dimensionality-reduction experiments are therefore included.

### Multiple recordings per subject

A subject can contribute multiple speech recordings across tasks. The project therefore performs subject-level aggregation and, for several experiments, subject-level cross-validation.

### Low sample size

Although there are 2,176 recordings, the main subject-level modeling dataset contains only **272 subjects**. This makes overfitting and unstable minority-class performance important considerations.

---

# Interpretability

The project includes several interpretability components:

- XGBoost feature importance
- SHAP summary plots
- Confusion matrices
- Per-class precision, recall and F1
- Feature-distribution visualizations

The three-class experiment specifically uses SHAP to inspect how individual acoustic features influence severity predictions.

---

# Reproducibility

Most experiments use:

```python
RANDOM_STATE = 42
```

and 5-fold stratified cross-validation.

For reproducible experiments, ensure that:

1. The original SAND Task 1 dataset is placed under the expected `Train/task1/` directory.
2. The metadata spreadsheet is available.
3. The notebooks are run in the intended order.
4. Generated manifests and feature CSVs are not mixed between different preprocessing runs.
5. The same library/model versions are used where possible.

---

# Future Work

Potential directions supported by the current experimental setup include:

- Better handling of the extremely small severe-class population.
- Patient-level data augmentation.
- End-to-end fine-tuning of speech foundation models.
- Better temporal modeling across a subject's multiple speech tasks.
- More principled multimodal fusion.
- Calibration and uncertainty estimation.
- External validation on held-out subjects.
- Systematic ablation of individual acoustic feature groups.
- Improved spectrogram representations and audio-vision fusion.
- Regression of continuous dysarthria severity in addition to classification.
- More rigorous clinical interpretability analysis.

---

# Notebooks

| Notebook | Purpose |
|---|---|
| `Data_Loading.ipynb` | Dataset preparation, cleaning, VAD and normalization |
| `Manual.ipynb` | Handcrafted acoustic/clinical feature engineering |
| `Wav2Vec.ipynb` | Wav2Vec 2.0 embedding extraction and ML experiments |
| `Manual + Wav2vec.ipynb` | Handcrafted + Wav2Vec feature fusion |
| `five_to_three.ipynb` | Three-class severity experiment with SHAP |
| `all_combined.ipynb` | Consolidated experimental pipeline |

---

## Disclaimer

This repository contains an academic/research implementation for speech-based dysarthria severity classification. Model outputs should be treated as experimental predictions and **not as clinical diagnoses or medical decisions**.
