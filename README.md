# Dual-Stream Vision Transformer for Multi-Domain Skin Disease Classification Using Frozen DINOv2

## Overview

This project presents a deep learning framework for skin disease classification using a Dual-Stream Vision Transformer architecture combined with a frozen DINOv2 feature extractor.

The objective is to improve skin disease recognition across multiple domains by leveraging powerful self-supervised visual representations while reducing computational costs through frozen backbone training.

The proposed approach extracts rich image features using DINOv2 and processes them through a dual-stream transformer architecture to enhance classification performance across diverse dermatological datasets.

---

## Problem Statement

Accurate skin disease diagnosis is challenging due to:

* Variations in image quality
* Different acquisition devices
* Diverse skin tones
* Domain shifts between datasets
* Limited labeled medical data

This project addresses these challenges through transfer learning and transformer-based feature fusion.

---

## Technologies Used

* Python
* PyTorch
* Vision Transformers (ViT)
* DINOv2
* Deep Learning
* Computer Vision
* Transfer Learning
* Medical Image Analysis
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## Model Architecture

### Frozen DINOv2 Backbone

* Pretrained DINOv2 model used for feature extraction
* Backbone weights remain frozen during training
* Reduces computational requirements
* Improves feature generalization

### Dual-Stream Vision Transformer

The architecture consists of:

1. Global Feature Stream

   * Captures overall lesion characteristics
   * Learns global contextual information

2. Local Feature Stream

   * Focuses on fine-grained lesion patterns
   * Captures texture and structural details

The outputs of both streams are fused for final classification.

---

## Workflow

1. Image Acquisition
2. Data Preprocessing
3. Feature Extraction using Frozen DINOv2
4. Dual-Stream Transformer Processing
5. Feature Fusion
6. Classification Layer
7. Performance Evaluation

---

## Dataset

The model is trained and evaluated using skin disease image datasets containing multiple disease categories.

Example categories:

* Melanoma
* Basal Cell Carcinoma
* Benign Keratosis
* Dermatofibroma
* Vascular Lesions
* Other Skin Conditions

---

## Evaluation Metrics

The following metrics are used:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* ROC-AUC

---

## Results

The proposed architecture demonstrates strong performance in multi-domain skin disease classification by combining:

* Robust DINOv2 representations
* Transformer-based feature learning
* Efficient transfer learning strategy

---

## Project Structure

dual-stream-vit-skin-disease-classification/

├── README.md

├── notebooks/

├── src/

├── models/

├── dataset/

├── results/

├── screenshots/

├── requirements.txt

└── trained_models/

---

## Future Improvements

* Multi-modal clinical data integration
* Explainable AI techniques (Grad-CAM)
* Real-time deployment
* Mobile healthcare applications
* Federated learning for privacy-preserving training

---

## Skills Demonstrated

* Deep Learning
* Medical AI
* Computer Vision
* Vision Transformers
* DINOv2
* Transfer Learning
* Model Evaluation
* Research & Development

---

## Author

Mohammed Amer Ahmed

MSc Data Analytics

Dublin City University, Ireland
