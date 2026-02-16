# Multimodal Emotion Recognition Using Speech and Text

This project implements an emotion recognition system using three different approaches: speech-based modeling, text-based modeling, and multimodal fusion of speech and text features. The goal is to improve emotion classification accuracy by combining complementary information from multiple modalities.

## 📌 Project Overview

Emotion recognition plays an important role in human-computer interaction, virtual assistants, and affective computing. Traditional systems rely on a single modality, which may not capture complete emotional information. This project demonstrates how combining speech and textual data can enhance classification performance.

The system consists of three pipelines:

1. Speech Model – Extracts acoustic features from audio signals and predicts emotions.
2. Text Model – Uses text sequences with a Bidirectional LSTM (BiLSTM) network to classify emotions.
3. Fusion Model – Combines speech and text representations to improve prediction accuracy.

## 📂 Dataset

The Toronto Emotional Speech Set (TESS) dataset was used for training and evaluation. The dataset contains speech recordings expressing multiple emotions such as:

- Angry
- Happy
- Sad
- Fear
- Disgust
- Surprise
- Neutral

Text data was generated using emotion labels derived from file names.

## ⚙️ Technologies Used

- Python
- TensorFlow / Keras
- Librosa
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## 📊 Results

The performance of the models is summarized below:

- Speech Model Accuracy: 96%
- Text Model Accuracy: 100%
- Fusion Model Accuracy: 98%

The fusion model achieved better performance than the speech-only model, demonstrating the advantage of multimodal learning.

## 📁 Project Structure
models/
speech_pipeline/
text_pipeline/
fusion_pipeline/

Results/
README.md
requirements.txt


## 🚀 Key Features

- Audio preprocessing and feature extraction using MFCC
- Text tokenization and sequence modeling using BiLSTM
- Multimodal feature fusion
- Performance evaluation using accuracy and confusion matrix
- Visualization of results

## 🔮 Future Improvements

- Use real speech transcripts instead of generated text
- Apply attention-based deep learning models
- Improve robustness with larger datasets
- Deploy as a real-time emotion recognition system






