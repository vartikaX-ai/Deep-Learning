# Sequence Model Comparison for Sentiment Analysis

## About this Repository

This repository compares the performance of four popular sequence models for **binary sentiment analysis** using the **IMDb Movie Review Dataset**.

The objective of this project is to evaluate how different recurrent neural network architectures perform under the same training conditions. All models were trained using identical preprocessing steps, optimizer, loss function, and number of epochs to ensure a fair comparison.

---

# Models Compared

- Simple Recurrent Neural Network (SimpleRNN)
- Long Short-Term Memory (LSTM)
- Bidirectional Long Short-Term Memory (BiLSTM)
- Gated Recurrent Unit (GRU)

---

# Dataset

This project uses the **IMDb Movie Review Sentiment Analysis Dataset**.

The dataset is **not included** in this repository due to GitHub file size limitations.

You can download the dataset from Kaggle:

https://www.kaggle.com/datasets/hqdataprofiler/imdb-dataset-of-50k-movie-reviews-cleaned

After downloading the dataset, update the dataset path in each Python script before running the project.

---

# Training Configuration

All models were trained using the same configuration.

| Parameter | Value |
|-----------|-------|
| Dataset | IMDb Movie Reviews |
| Task | Binary Sentiment Classification |
| Optimizer | Adam |
| Loss Function | Binary Crossentropy |
| Metric | Accuracy |
| Epochs | 5 |
| Batch Size | 32 |
| Validation Split | 0.2 |
| Sequence Length | 200 |

---

# Model Comparison Results

| Model | Loss | Accuracy |
|-------|------:|----------:|
| SimpleRNN | 0.4340 | 80.96% |
| LSTM | 0.3011 | 87.74% |
| BiLSTM | 0.3158 | 86.99% |
| **GRU** | **0.2858** | **88.61%** |

---

# Performance Analysis

### SimpleRNN

- Simplest recurrent architecture
- Lowest computational complexity
- Lowest accuracy among the four models
- Struggles with long-term dependencies

---

### LSTM

- Uses memory cells and multiple gates
- Better at capturing long-term dependencies
- Achieved a significant improvement over SimpleRNN

---

### BiLSTM

- Processes sequences in both forward and backward directions
- Captures contextual information from both sides
- Produced performance similar to LSTM under the chosen training configuration

---

### GRU

- Uses Update Gate and Reset Gate
- Simpler architecture than LSTM
- Fewer trainable parameters
- Achieved the highest accuracy and lowest loss in this comparison

---

# Conclusion

Under identical preprocessing and training conditions, the **GRU model** achieved the best performance on the IMDb Movie Review dataset with an **accuracy of 88.61%** and the **lowest loss (0.2858)**.

Although **LSTM** and **BiLSTM** also produced strong results, **GRU** demonstrated the best balance between model complexity and predictive performance in this experiment.

---

# Project Structure

```text
Sequence_Model_Comparison/
│
├── projects/
│   ├── SimpleRNN_IMDB.py
│   ├── LSTM_IMDB.py
│   ├── BiLSTM_IMDB.py
│   └── GRU_IMDB.py
│
├── README.md
```

---

# Libraries Used

- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn

---

# Note

All models were trained for **5 epochs** using identical preprocessing and training configurations to provide a fair comparison while keeping training time manageable. Increasing the number of epochs or tuning hyperparameters may further improve model performance.

---



This repository is part of my **Deep Learning** learning journey, where I compare different sequence models for Natural Language Processing (NLP) and sentiment analysis using **TensorFlow** and **Keras**.
