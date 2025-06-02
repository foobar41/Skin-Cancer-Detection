# 🩺 Skin Cancer Detection

This project aims to detect skin cancer by combining **image data** and **tabular metadata** using advanced machine learning techniques. It explores performance across standalone models (image/tabular) and their fusion, emphasizing class imbalance handling and robust evaluation metrics.

---

## 📊 Dataset Summary

- **Positive samples**: 314  
- **Negative samples**: 320,533  
- **Samples with NULL features**: 15,314  
- High feature correlation observed based on lesion location metadata.

---

## 🔎 Tabular Data Processing

### Preprocessing

- Dropped 5 least important features using Gini importance from a Random Forest Classifier.
- Addressed imbalance using:
  - **SMOTE-ENN** (resampling minority class up to a 30-70 ratio)
  - **Class weights**

### Models and Performance

| Model            | AUC Score |
|------------------|-----------|
| SVM (RBF kernel) | 0.87      |
| Logistic Regression | 0.89  |
| XGBoost          | **0.93**  |

- **Optimization**: Bayesian Optimization with Stratified 5-Fold Cross Validation.
- **Evaluation Metric**: ROC-AUC

---

## 🖼️ Image-Based Modeling

### VGG-19 (with 50k samples)

| Variant                  | ROC-AUC | F1 Score |
|--------------------------|---------|----------|
| Base (with augmentation) | 0.81    | 0.66     |
| + L2 Norm                | 0.77    | 0.63     |
| + Class Weights          | 0.79    | 0.67     |

### Inception-ResNet-v2 (with 100k samples)

| Variant                          | ROC-AUC | F1 Score |
|----------------------------------|---------|----------|
| Base (with augmentation)         | 0.64    | 0.47     |
| + L2 Norm + Class Weights        | 0.74    | 0.63     |

### EfficientNet-v2-b1 (Full Dataset)

- **ROC-AUC**: **0.96**  
- **F1 Score**: **0.79**

---

## 🔄 Fusion Model

- Applied **99% undersampling** of negative class.
- Used **image augmentations**: Rotation, Flipping, Brightness Contrast, Resizing, Normalization.
- Extracted OOF predictions from **EfficientNet-v2-b1**, merged with metadata.
- Created new features like:
  - Lesion visibility score
  - Lesion position
- Used a **Voting Classifier** combining:
  - XGBoost
  - LightGBM
  - CatBoost

> 📈 **Final AUC Score**: **0.96** (Stratified 5-Fold CV)

---

## 💡 Key Observations

### Image Models

- **EfficientNet-v2-b1** gave the best results.
- Adjusting decision boundaries in **VGG19** improved results (ROC-AUC: 0.91, F1: 0.86 at threshold = 0.05).
- **Balancing data** proved more effective than model tuning.

### Tabular Models

- **SVM** had high convergence time vs. LR/XGBoost.
- Removing more than 5 features hurt performance.
- Undersampling allowed faster tuning with similar results.

### General

- **Tabular data outperformed image models.**
- **Fusion model** delivered the best overall performance.

---

## 🧠 Tech Stack

- Python, PyTorch, TensorFlow, Scikit-learn, XGBoost, LightGBM, CatBoost
- Bayesian Optimization
- SMOTE-ENN for imbalance handling
