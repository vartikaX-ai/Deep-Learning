# Convolutional Neural Network (CNN)

## About this Repository

This repository contains my implementations and practical projects on **Convolutional Neural Networks (CNNs)** using **TensorFlow** and **Keras**.

The objective of this repository is to understand the complete workflow of image classification using CNNs, including image preprocessing, data augmentation, convolution operations, pooling, feature extraction, model training, evaluation, and model checkpointing.

---

# Project Covered

- Cats vs Dogs Image Classification

---

# Topics Covered

## CNN Fundamentals

- Introduction to Convolutional Neural Networks
- Difference Between CNN and Feed Forward Neural Networks
- Spatial Structure of Images
- Local Connectivity
- Weight Sharing
- Receptive Field

---

## CNN Layers

- Input Layer
- Convolutional Layer (Conv2D)
- Filters / Kernels
- Feature Maps
- Activation Functions
- ReLU
- Pooling Layer
- Max Pooling
- Flatten Layer
- Fully Connected (Dense) Layer
- Output Layer

---

## CNN Concepts

- Kernel Size
- Number of Filters
- Stride
- Padding
  - Valid Padding
  - Same Padding
- Downsampling
- Depth of Feature Maps

---

## Image Preprocessing

- Loading Image Dataset
- Image Resizing
- Image Normalization
- Batch Processing
- Dataset Prefetching

---

## Data Augmentation

- Random Flip
- Random Rotation
- Random Zoom

---

## Model Training

- Binary Image Classification
- Adam Optimizer
- Binary Crossentropy Loss
- Accuracy Metric
- EarlyStopping
- ModelCheckpoint

---

## Model Evaluation

- Prediction on Validation Images
- Validation Accuracy
- Validation Loss
- Loading Best Saved Model

---

# Project Structure

```text
Convolutional_Neural_Network/
│
├── projects/
│   └── CNN_CatsVsDogs.py
│
├── README.md
└── requirements.txt
```

---

# Dataset

This project uses the **Microsoft Cats vs Dogs** dataset.

The dataset is not included in this repository due to its large size.

You can download it from Kaggle:

https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

After downloading, extract the dataset and update the dataset path inside the project.

---

# Libraries Used

- TensorFlow
- Keras
- NumPy

---

# Learning Outcome

After completing this repository, I gained practical experience with:

- Building CNN architectures using Keras Sequential API
- Loading image datasets using TensorFlow
- Image preprocessing and normalization
- Data augmentation techniques
- Convolution and pooling operations
- Binary image classification
- Model training using TensorFlow
- EarlyStopping and ModelCheckpoint callbacks
- Saving and loading trained models
- Evaluating CNN performance on validation data
