# Face Recognition Using PCA and SVM

A machine learning project that performs face recognition using **Principal Component Analysis (PCA)** for dimensionality reduction and a **Support Vector Machine (SVM)** for classification.

## Project Overview

Face images contain a large number of pixel features. A 64 × 64 grayscale image contains:

**64 × 64 = 4096 features**

Working directly with all 4096 features can make the model more computationally expensive and can include redundant information.

This project investigates whether PCA can reduce the dimensionality of face images while maintaining useful recognition performance.

The project compares:

* Face recognition **without PCA**
* Face recognition **with PCA**
* Different numbers of PCA components
* Real-time webcam recognition
* Unknown-person rejection

---

## Machine Learning Pipeline

```text
Face Image
     ↓
Grayscale
     ↓
Resize to 64 × 64
     ↓
4096 Pixel Features
     ↓
StandardScaler
     ↓
PCA
     ↓
30 Principal Components
     ↓
RBF SVM
     ↓
Person Prediction
```

---

## Dataset

The project uses a dataset containing:

* 20 people
* 10 images per person
* 200 face images
* Image size: 64 × 64
* Grayscale images
* 4096 original features per image

The dataset contains both webcam-collected images and images derived from the Olivetti Faces dataset.

This difference in image source is an important limitation because the two sources can have different lighting, camera characteristics, backgrounds, and image conditions.

---

## PCA Analysis

Different numbers of PCA components were tested.

| PCA Components |   Accuracy | Variance Retained |
| -------------: | ---------: | ----------------: |
|              5 |     59.72% |            57.69% |
|             10 |     80.56% |            70.36% |
|             20 |     90.28% |            81.07% |
|         **30** | **93.06%** |        **86.41%** |
|             50 |     93.06% |            92.37% |
|            100 |     93.06% |            98.27% |

Among the tested configurations, **30 components reached the highest observed accuracy while using only 30 features instead of the original 4096**.

This represents approximately:

**99.27% feature reduction.**

The 30-component configuration was selected for the PCA-SVM model based on the tested results. This does not mean that 30 components are universally optimal for every dataset.

---

## Technologies Used

* Python
* NumPy
* Pandas
* OpenCV
* Scikit-learn
* Matplotlib
* Joblib
* Jupyter Notebook
* VS Code

### Machine Learning Methods

* StandardScaler
* Principal Component Analysis (PCA)
* Support Vector Machine (SVM)
* RBF Kernel
* Haar Cascade Face Detection

---

## Real-Time Recognition

The project includes a webcam recognition system.

The webcam pipeline is:

```text
Webcam
   ↓
Face Detection
   ↓
64 × 64 Grayscale
   ↓
StandardScaler
   ↓
PCA
   ↓
SVM
   ↓
Predicted Person
```

The application also displays the SVM decision score and recognition information in real time.

---

## Unknown Person Detection

An additional experiment was performed to determine whether the system could reject people who were not present in the training dataset.

The experiments showed that the SVM decision scores of known and unknown people can overlap.

Therefore, a simple SVM-score threshold cannot currently provide reliable open-set recognition.

This is an important limitation of the current PCA-SVM approach.

---

## Key Findings

1. Face images contain a large number of pixel features.
2. PCA can reduce 4096 features to a much smaller representation.
3. Recognition accuracy increased as the number of PCA components increased in the tested range.
4. 30 components achieved 93.06% accuracy in the tested experiment.
5. Increasing from 30 to 50 or 100 components did not improve the observed accuracy.
6. PCA provides substantial dimensionality reduction.
7. The SVM classifier can recognize people from the known classes.
8. Reliable unknown-person rejection is more difficult than ordinary classification.

---

## Limitations

### 1. Dataset mismatch

Person1 images were collected using a webcam, while many other identities originate from the Olivetti Faces dataset.

This creates a domain difference between classes.

### 2. Closed-set classification

The SVM is trained to classify images into known classes. It does not naturally know that a completely new person exists.

### 3. Unknown-person rejection

The current threshold-based rejection method is not sufficiently reliable because known and unknown live webcam scores can overlap.

### 4. Dataset size

Only 200 images were used in the main dataset, which is relatively small for a real-world face-recognition system.

---

## Future Work

Future versions could investigate:

* More webcam images for every person
* More varied lighting and poses
* Face alignment
* More advanced face detection
* Face embeddings
* Deep learning-based face recognition
* Open-set recognition methods
* Distance-based identity verification
* Threshold calibration using separate validation data
* Larger and more diverse datasets

A particularly interesting extension would be to compare the PCA-SVM baseline with a modern deep face-embedding approach.

---

## Project Goal

The main goal of this project is not simply to recognize faces.

It is to investigate the relationship between:

**Mathematics → Dimensionality Reduction → Machine Learning → Real-Time AI**

PCA provides the mathematical foundation for reducing the high-dimensional face representation, while SVM performs the final classification.

---

## Author

**Pheany Phal**

M.Sc. Data Science & Artificial Intelligence
