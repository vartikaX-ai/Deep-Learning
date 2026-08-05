# Long Short-Term Memory (LSTM)

## About this Repository

This repository contains my implementations and practical projects on **Long Short-Term Memory (LSTM)** networks using **TensorFlow** and **Keras**.

The objective of this repository is to understand the complete workflow of sequence modeling and sentiment analysis, including text preprocessing, tokenization, sequence padding, word embeddings, LSTM architectures, model training, evaluation, and model checkpointing.

---

# Project Covered

- IMDb Movie Review Sentiment Analysis using Stacked LSTM

---

# Topics Covered

## LSTM Fundamentals

- Introduction to Long Short-Term Memory (LSTM)
- Limitations of Simple RNN
- Vanishing Gradient Problem
- Long-Term Dependencies
- Cell State
- Hidden State
- Forget Gate
- Input Gate
- Output Gate
- Stacked LSTM

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

## LSTM Architecture

- Embedding Layer
- First LSTM Layer
- Second LSTM Layer (Stacked LSTM)
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
Long_Short_Term_Memory/
│
├── projects/
│   └── IMDb_Sentiment_Analysis_LSTM.py
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
- Developing Long Short-Term Memory (LSTM) networks
- Building Stacked LSTM architectures
- Binary Sentiment Classification
- Using EarlyStopping and ModelCheckpoint callbacks
- Training and evaluating LSTM models
- Predicting sentiment from movie reviews

---

# Requirements

Install the required libraries before running the project:

```bash
pip install -r requirements.txt
```

---
This repository is part of my **Deep Learning** learning journey, where I implement neural network architectures using **TensorFlow** and **Keras**, build practical projects, and upload them to GitHub as part of my AI Engineer roadmap.
