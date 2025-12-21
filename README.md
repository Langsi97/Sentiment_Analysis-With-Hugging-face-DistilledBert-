
# Sentiment Analysis with Hugging Face and Custom Neural Models

## Overview
This project presents a complete, end-to-end **sentiment analysis pipeline** for binary text classification (positive / negative) on the **IMDB movie reviews dataset**.

The work compares **classical baselines**, **neural models trained from scratch**, and **pretrained Transformer models (DistilBERT)** under identical evaluation conditions.  
The emphasis is on **model comparison, ablation studies, error analysis, and engineering robustness**, rather than accuracy alone.

---

## Dataset
- **IMDB Reviews Dataset**
- 50,000 movie reviews
  - 25,000 training
  - 25,000 test
- Balanced labels (positive / negative)

Loaded using:
- TensorFlow Datasets (for custom models)
- Hugging Face `datasets` (for Transformer models)

---

## Models Implemented

### 1. Baseline Model (Neural Bag-of-Words)
- Text vectorization + embedding layer
- Global average pooling
- Dense classifier
- Trained end-to-end from scratch

**Purpose:**  
Establish a strong, interpretable baseline and sanity check.

---

### 2. Model Trained from Scratch (BiLSTM)
- Trainable word embeddings
- Bidirectional LSTM
- Dense classification head
- No pretrained language representations

**Purpose:**  
Evaluate whether sequence modeling alone can outperform simpler baselines.

---

### 3. Pretrained Transformer Models (DistilBERT)

Three fine-tuning regimes were explored:

#### a) Frozen DistilBERT (Feature Extraction)
- Entire encoder frozen
- Only classification head trained

**Trainable parameters:** ~0.59M / 66.9M

---

#### b) Partial-Freeze DistilBERT
- Embeddings + lower transformer layers frozen
- Upper layers fine-tuned

**Trainable parameters:** ~21.9M / 66.9M

---

#### c) Full Fine-Tuned DistilBERT
- All layers fine-tuned end-to-end

**Trainable parameters:** ~66.9M / 66.9M

---

## Results (Test Set)

| Model | Trainable Params | Test Accuracy |
|-----|-----------------|---------------|
| Baseline (Avg Embedding) | ~1M | ~86% |
| BiLSTM (from scratch) | ~3–5M | ~84% |
| Frozen DistilBERT | ~0.59M | **85.36%** |
| Partial Freeze DistilBERT | ~21.86M | **83.94%** |
| Full Fine-Tuned DistilBERT | ~66.96M | **91.30%** |

---

## Key Observations
- A simple baseline performs competitively, confirming that IMDB sentiment is highly lexical.
- The BiLSTM trained from scratch did **not** outperform the baseline, highlighting the limits of learning language representations from limited data.
- Pretrained Transformer models provide a clear performance boost.
- Full fine-tuning consistently achieved the highest accuracy, especially on context-dependent reviews.

---

## Error Analysis
A fixed subset of the test set was used to perform **qualitative error analysis** across all models.

Findings:
- Baseline and BiLSTM models struggled with:
  - Long-range negation
  - Mixed or contradictory sentiment
  - Subtle contextual cues
- Frozen DistilBERT handled lexical sentiment well but failed on deeper context.
- Full fine-tuned DistilBERT consistently resolved complex cases.

---

## Inference & Qualitative Comparison
Hand-crafted review examples were passed through all models to compare prediction confidence.

Full fine-tuned DistilBERT:
- Produced the most stable predictions
- Assigned higher confidence on ambiguous reviews
- Demonstrated better contextual understanding

---

## Engineering Considerations
- Implemented under **limited RAM/GPU constraints**
- Sequential training with explicit garbage collection
- Safe model persistence across Colab and Kaggle
- Restart-safe, reproducible workflows

---

## Tech Stack
- Python
- PyTorch
- TensorFlow / Keras
- Hugging Face Transformers & Datasets
- Matplotlib
- Google Colab / Kaggle

---

## Key Takeaways
- Strong baselines are essential for meaningful comparison
- Training language models from scratch is rarely competitive on limited data
- Pretrained Transformers significantly improve performance
- Full fine-tuning offers the best results when resources allow

---

## Author
**Langsi Ambe Revelation**  
MSc Data Science  
Focus: NLP, Machine Learning, Model Evaluation & Interpretability
