# Dual-Stream Vision Transformer for Multi-Domain Skin Disease Classification Using Frozen DINOv2

## MSc Practicum Project — Dublin City University

**Authors:** Sai Sawanth Kanjarla, Amer Ahmed Mohammed  
**Supervisor:** Professor Luca Rossetto  
**Programme:** MSc Computing, School of Computing, DCU

---

## Project Overview

This project proposes a dual-stream classification architecture built on a frozen DINOv2-ViT-B/14 backbone for classifying 14 skin diseases across three medical domains (Oncological, Benign, Infectious). The architecture trains only 0.58% of total model parameters (~500K out of 86.6M) while achieving 81.65% accuracy and 0.7236 macro F1, outperforming the single-stream baseline by 18.45 percentage points.

---

## Results

| Metric | Baseline | Proposed | Improvement |
|---|---|---|---|
| Accuracy | 63.20% | **81.65%** | +18.45% |
| Macro F1 | 0.5944 | **0.7236** | +0.1291 |
| Weighted F1 | 0.6606 | **0.8098** | +0.1492 |
| Trainable params | 12,302 | ~500,000 | 0.58% of total |

---

## Files

**src/** contains all source code notebooks:

| File | Description |
|---|---|
| dataset_analysis.ipynb | Dataset exploration and statistics |
| data_pipeline.ipynb | Augmentation, sampling, domain labels |
| baseline.ipynb | Frozen DINOv2 + linear head baseline |
| proposed_modelling.ipynb | Dual-stream model training and evaluation |
| evaluation_ablation.ipynb | Ablation studies, McNemar test, attention maps |

**docs/** contains all documentation:

| File | Description |
|---|---|
| Final_paper.pdf | IEEE double-column report (8 pages) |
| Practicum_Literature_Review.pdf | Literature review (20 papers) |
| proposal/ | Research proposal |

---

## Dataset

- **Source:** [ahmed-ai/skin-lesions-classification-dataset](https://huggingface.co/datasets/ahmed-ai/skin-lesions-classification-dataset)
- **Composition:** HAM10000 (ISIC 2019) + MSLDv2.0
- **Total:** 36,656 images | 14 classes | 3 domains
- **Splits:** Train 29,322 | Validation 3,660 | Test 3,674
- **Imbalance:** 53.9:1
- **License:** MIT

| Domain | Classes | Images |
|---|---|---|
| Oncological | Melanoma, BCC, SCC, Actinic keratoses | 9,340 (25.5%) |
| Benign | Nevi, Benign keratosis, Dermatofibroma, Vascular lesions | 15,991 (43.6%) |
| Infectious | Monkeypox, Chickenpox, Cowpox, HFMD, Measles, Healthy | 11,325 (30.9%) |

---

## Architecture

Frozen DINOv2-ViT-B/14 backbone with four trainable components:

- **Domain Router** (~197K params): Soft routing weights via softmax
- **Stream 1 Onc/Benign** (~397K params): 8-class classification head
- **Stream 2 Infectious** (~396K params): 6-class classification head
- **Projection Head** (~98K params): Contrastive embeddings (training only)

**Loss:** 1.0 x Focal + 1.0 x Routing + 0.1 x SupCon

---

## How to Run

Run notebooks in this order on Google Colab (T4 GPU):

1. dataset_analysis.ipynb
2. data_pipeline.ipynb
3. baseline.ipynb
4. proposed_modelling.ipynb
5. evaluation_ablation.ipynb

All intermediate data saves to Google Drive automatically.

**Requirements:** datasets, torch, torchvision, albumentations, timm, scikit-learn, matplotlib, seaborn, pandas, numpy, scipy, tqdm

---

## Experiments

| # | Experiment | Result |
|---|---|---|
| 1 | Baseline: DINOv2 + linear head | Acc=63.20%, F1=0.5944 |
| 2 | Proposed vs baseline | +18.45% accuracy, +0.1291 F1 |
| 3 | Focal loss vs cross-entropy | Focal contributes +0.0044 F1 |
| 4 | Contrastive learning | Contributes +0.0027 F1 |
| 5 | Ablation (4 variants) | All components contribute |
| 6 | Pox virus differentiation | Cowpox 0.99, Monkeypox 0.98, Chickenpox 0.95 |

---

## Key References

1. Oquab et al. (2023) - DINOv2
2. Mohan et al. (2025) - DINOv2 for skin classification
3. Lin et al. (2017) - Focal loss
4. Khosla et al. (2020) - Supervised contrastive learning
5. Badr et al. (2024) - Hierarchical multi-model skin classification
