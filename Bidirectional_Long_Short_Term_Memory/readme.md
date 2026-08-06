# Bidirectional Long Short-Term Memory (BiLSTM)

## About this Repository

This repository contains my implementations and practical projects on **Bidirectional Long Short-Term Memory (BiLSTM)** networks using **TensorFlow** and **Keras**.

The objective of this repository is to understand the complete workflow of sequence modeling and sentiment analysis, including text preprocessing, tokenization, sequence padding, word embeddings, bidirectional LSTM architectures, model training, evaluation, and model checkpointing.

---

# Project Covered

- IMDb Movie Review Sentiment Analysis using Stacked Bidirectional LSTM (BiLSTM)

---

# Topics Covered

## BiLSTM Fundamentals

- Introduction to Bidirectional LSTM (BiLSTM)
- Difference Between LSTM and BiLSTM
- Forward LSTM
- Backward LSTM
- Bidirectional Processing
- Long-Term Dependencies
- Context from Both Directions
- Stacked Bidirectional LSTM

---

## Text Preprocessing

- Reading Dataset using Pandas
- Train-Test Split
- Label Encoding
- Tokenization
- Vocabulary Creation
- Text to Sequence Conversion
- Sequence Padding (`pad_sequences`)
- Fixed-Length Input Sequences

---

## Embedding Layer

- Word Embeddings
- Vocabulary Size (`input_dim`)
- Embedding Dimension (`output_dim`)
- Dense Vector Representation

---

## BiLSTM Architecture

- Embedding Layer
- First Bidirectional LSTM Layer
- Second Bidirectional LSTM Layer (Stacked BiLSTM)
- Dense Hidden Layer
- Output Layer

---

## Model Training

- Binary Sentiment Classification
- Adam Optimizer
- Binary Crossentropy Loss
- Accuracy Metric
- EarlyStopping Callback
- ModelCheckpoint Callback

---

## Model Evaluation

- Prediction
- Threshold-based Classification
- Test Loss
- Test Accuracy

---

# Project Structure

```text
Bidirectional_Long_Short_Term_Memory/
│
├── projects/
│   └── IMDb_Sentiment_Analysis_BiLSTM.py
│
├── README.md
└── requirements.txt
```

---

# Dataset

This project uses the **IMDb Movie Review Sentiment Analysis Dataset**.

The dataset is **not included** in this repository due to GitHub file size limitations.

You can download the dataset from Kaggle:

https://www.kaggle.com/datasets/hqdataprofiler/imdb-dataset-of-50k-movie-reviews-cleaned

After downloading, update the dataset path in the Python script before running the project.

---

# Libraries Used

- TensorFlow
- Keras
- Pandas
- Scikit-learn

---

# Learning Outcome

After completing this repository, I gained practical experience with:

- Working with sequential text data
- Reading and preprocessing NLP datasets
- Label Encoding
- Tokenization and vocabulary creation
- Converting text into numerical sequences
- Sequence Padding
- Building Word Embeddings
- Developing Bidirectional Long Short-Term Memory (BiLSTM) networks
- Building Stacked Bidirectional LSTM architectures
- Understanding forward and backward sequence processing
- Binary Sentiment Classification
- Using EarlyStopping and ModelCheckpoint callbacks
- Training and evaluating BiLSTM models
- Predicting sentiment from movie reviews

---

# Requirements

Install the required libraries before running the project:

```bash
pip install -r requirements.txt
```

---


This repository is part of my **Deep Learning** learning journey, where I implement neural network architectures using **TensorFlow** and **Keras**, build practical projects, and upload them to GitHub as part of my AI Engineer roadmap.
