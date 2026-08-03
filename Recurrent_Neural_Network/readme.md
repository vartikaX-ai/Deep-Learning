# Recurrent Neural Network (RNN)

## About this Repository

This repository contains my implementations and practical projects on **Recurrent Neural Networks (RNNs)** using **TensorFlow** and **Keras**.

The objective of this repository is to understand the complete workflow of sequence modeling and text classification, including text preprocessing, tokenization, sequence padding, word embeddings, recurrent neural networks, model training, evaluation, and model checkpointing.

---

# Project Covered

- IMDb Movie Review Sentiment Analysis

---

# Topics Covered

## RNN Fundamentals

- Introduction to Recurrent Neural Networks (RNNs)
- Sequential Data
- Difference Between Feed Forward Neural Networks and RNNs
- Hidden State
- Weight Sharing
- Recurrent Connections
- Sequence Processing
- Long-Term Dependencies
- Vanishing Gradient Problem

---

## Text Preprocessing

- Reading Dataset using Pandas
- Handling Missing Values
- Train-Test Split
- Label Encoding
- Tokenization
- Vocabulary
- Word Index
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

## RNN Architecture

- Embedding Layer
- SimpleRNN Layer
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
Recurrent_Neural_Network/
│
├── datasets/
│   └── IMDB_Dataset_CLEANED.csv
│
├── projects/
│   └── IMDb_Sentiment_Analysis_RNN.py
│
├── README.md
└── requirements.txt
```

---

# Dataset

This repository includes the **IMDb Movie Review Sentiment Analysis Dataset**.

Dataset File:

- `datasets/IMDB_Dataset_CLEANED.csv`

The dataset contains movie reviews labeled as:

- Positive
- Negative

It is used to train and evaluate a Recurrent Neural Network (RNN) for binary sentiment classification.

---

# Libraries Used

- TensorFlow
- Keras
- Pandas
- NumPy
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
- Developing Recurrent Neural Networks (SimpleRNN)
- Binary Sentiment Classification
- Using EarlyStopping and ModelCheckpoint callbacks
- Training and evaluating RNN models
- Predicting sentiment from movie reviews

---

# Requirements

Install the required libraries before running the project:

```bash
pip install -r requirements.txt
```

---

This repository is part of my **Deep Learning** learning journey, where I implement neural network architectures from scratch using **TensorFlow** and **Keras**, build practical projects, and upload them to GitHub as part of my AI Engineer roadmap.
