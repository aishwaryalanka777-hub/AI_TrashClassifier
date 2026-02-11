# 🗑️ AI Trash Classifier
![Python](https://img.shields.io/badge/python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13-orange)
![License](https://img.shields.io/badge/license-MIT-green)
A machine learning project that classifies trash into categories: **plastic, glass, metal, cardboard, paper, trash**.  
Built using **Python 3.10**, **TensorFlow**, **Keras**, and trained on a custom dataset.

---

## 🌟 Features

- Train a Convolutional Neural Network (CNN) on your trash dataset
- Classify images from dataset or webcam
- Save trained model as `trash_model.h5`
- Easy to set up and run on Windows PC
- Lightweight and beginner-friendly project


## 📂 Dataset Structure

## Dataset
- Custom dataset with images of trash categorized into six classes.
- Images preprocessed and augmented for better model performance.
- Suitable for beginners to practice computer vision projects.


## How it Works
1. **Data Preprocessing:** Images are resized, normalized, and augmented.
2. **Model Architecture:** Uses Convolutional Neural Networks (CNN) with TensorFlow/Keras.
3. **Training:** The model is trained on the dataset with multiple epochs until it reaches good accuracy.
4. **Prediction:** The webcam captures live input, and the model predicts the class of trash in real-time.

Installation

git clone https://github.com/aishwaryalanka777-hub/AI_TrashClassifier.git


cd AI_TrashClassifier

pip install -r requirements.txt

python main.py




