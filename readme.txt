# Skin Lesion Classification using MobileNetV2

A deep learning project that classifies dermatoscopic skin images into three classes (**Melanoma**, **Melanocytic Nevus**, **Benign Keratosis**) using transfer learning.

> **Disclaimer:** This is an academic project for learning purposes only. It is **not** a medical diagnosis tool.

## Project Info
- **College:** Jyothi Engineering College
- **Course:** Machine Learning Project, CSE Batch C (2024-2028)
- **Group:** [group number]
- **Team members:** [name 1], [name 2], [name 3], [name 4]

## Overview
Skin lesions such as melanoma can look very similar to harmless moles. Telling them apart depends on subtle patterns of colour, texture, and shape that cannot be written as fixed rules, so a model learns them from labelled images instead. This project uses a pretrained MobileNetV2 network and fine-tunes it on a small balanced subset of HAM10000.

## Results

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Benign Keratosis | 0.74 | 0.67 | 0.70 |
| Melanoma | 0.66 | 0.73 | 0.69 |
| Nevus | 0.74 | 0.73 | 0.73 |

**Overall test accuracy: 70.9%** on 450 unseen images (random guessing would give 33%).

### Training curves
![Training curves](training_curves.png)

### Confusion matrix
![Confusion matrix](confusion_matrix.png)

### Sample predictions
![Sample predictions](sample_predictions.png)

## Dataset
- **HAM10000** ("Human Against Machine with 10000 training images"), 10,015 dermatoscopic images
- Kaggle: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000
- Original source: https://doi.org/10.7910/DVN/DBW86T
- **Subset used:** 1,000 images per class (3,000 total), split 70% train / 15% validation / 15% test
- Licence: CC-BY-NC-SA 4.0 (non-commercial use)
- The dataset is **not included** in this repository because of its size

## Model
- **Base:** MobileNetV2 pretrained on ImageNet
- **Head:** Global Average Pooling, Dropout (0.3), Dense (3)
- **Augmentation:** random flips, rotation, zoom
- **Training:** phase 1 trains only the new head (Adam, 1e-3); phase 2 fine-tunes the last 30 layers (Adam, 1e-5). If fine-tuning lowers validation accuracy, the better phase-1 weights are restored automatically.

## How to Run (Google Colab, free GPU)
1. Open Google Colab, set **Runtime > Change runtime type > T4 GPU**
2. Upload `skin_disease_classifier.py` and your own `kaggle.json`
3. Run these cells:

```python
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kmader/skin-cancer-mnist-ham10000
!unzip -q skin-cancer-mnist-ham10000.zip -d ham10000
!python skin_disease_classifier.py
```

**No Kaggle account?** Skip the first three lines, run `!pip install -q datasets`, then `!python skin_disease_classifier.py`. The script downloads the data from Hugging Face automatically.

## Output Files
| File | Description |
|---|---|
| `training_curves.png` | Accuracy and loss per epoch |
| `confusion_matrix.png` | Test set confusion matrix |
| `sample_predictions.png` | Example predictions (green = correct, red = wrong) |
| `skin_model.keras` | Saved trained model |

## Limitations
- Only 3 of the 7 HAM10000 classes and 3,000 images were used
- Visually similar lesion types (melanoma vs nevus) are often confused
- Not clinically validated; 70.9% is far below what a medical tool needs
- Trained on dermatoscope images, so it may not work on ordinary phone photos

## Future Work
- Use all 7 classes and the full dataset
- Try EfficientNet or ResNet, with class weighting
- Build a Streamlit web app for uploading images

## References
1. Tschandl, P., Rosendahl, C., Kittler, H. (2018). *The HAM10000 dataset*. Scientific Data, 5, 180161.
2. Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks*. CVPR.