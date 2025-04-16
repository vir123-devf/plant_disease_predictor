# 🌿 Plant Disease Prediction

This project uses a **Convolutional Neural Network (CNN)** to classify **38 different plant diseases** from leaf images. The model is trained on the **PlantVillage** dataset and deployed using a clean and interactive **Streamlit** web app developed in **PyCharm**.

---

## 🧠 Project Overview

- **Model Type**: CNN (Convolutional Neural Network)
- **Deployment**: Streamlit web app
- **Frameworks**: TensorFlow, Keras, OpenCV, NumPy, Streamlit
- **Training Accuracy**: ~97%
- **Validation Accuracy**: ~88%

---

## 📁 Dataset

- **Source**: [PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)
- **Total Classes**: 38 disease categories across various crops such as:
  - Apple
  - Tomato
  - Grape
  - Pepper
  - Corn
  - Potato
  - And many more…

Each class contains leaf images categorized into healthy and diseased conditions.

---

## 🏗️ CNN Architecture

- Multiple **Conv2D** + **MaxPooling2D** layers
- **ReLU** activations
- **Dropout** for regularization
- **Dense** layers
- **Softmax** output for 38-class classification

---

## 🚀 Streamlit Web App

Users can upload an image of a plant leaf and receive an instant disease prediction.

### 💻 How to Run the Web App

1. **Install dependencies**:
   ```bash
   pip install streamlit tensorflow opencv-python numpy pillow
   ```

2. **Run the app**:
   ```bash
   streamlit run app.py
   ```

3. Upload a leaf image and get predictions with confidence scores.

---

## 📊 Results

| Metric              | Value         |
|---------------------|---------------|
| Training Accuracy   | ~97%          |
| Validation Accuracy | ~88%          |
| Total Classes       | 38            |

- more info in notebook

---

## 🖼️ Sample UI

![Streamlit App Screenshot](screenshot.png)
