# Emotion Recognition Project
This project is a facial emotion recognition system that combines a deep learning model with a user interface using **FastAPI**, **Streamlit**, and optionally **Anvil**. The model is trained on the KDEF dataset 
and predicts emotions like **Happy, Sad, Angry, Surprise, Neutral**, etc.

## 📁 Project Structure
Emotion-Recognition-Project/
│
├── README.md <- Project description (this file)
├── requirements.txt <- List of dependencies
├── notebooks/ <- Google Colab training notebook (.ipynb)
├── app/ <- FastAPI + Streamlit and Anvil interface
├── images/ <- Screenshots (UI, accuracy, loss, confusion matrix)
├── src/ <- Python source code (utils, preprocessing)

## 🚀 Features
- Deep learning model (CNN) for emotion classification
- Preprocessing pipeline (grayscale, resize, normalization)
- FastAPI REST API
- Streamlit or Anvil-based interactive interface
- Evaluation graphs: accuracy, loss, confusion matrix

