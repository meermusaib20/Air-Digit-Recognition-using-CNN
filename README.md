# ✋ Air Digit Recognition using CNN

This project is an end-to-end implementation of **air-drawn digit recognition** using a **Convolutional Neural Network (CNN)**.  
Users can draw digits either using a mouse or by drawing in the air with hand gestures, and the trained model predicts the digit in real time.

The main goal of this project was to understand how **computer vision, deep learning, and real-time input** can work together in a practical application.

---

## 🚀 Features

- ✍️ Collect custom digit data (mouse draw & air draw)
- 🧠 CNN trained on user-drawn digits
- ✋ Air drawing using hand gesture tracking
- 📊 Confidence-based prediction (uncertain outputs handled)
- 💻 Runs fully on CPU (no GPU required)

---

## 🛠️ Tech Stack

- **Python**
- **PyTorch** – CNN model and training
- **OpenCV** – drawing canvas & webcam handling
- **MediaPipe (Tasks API)** – hand gesture tracking
- **NumPy** – image processing

---

## 📂 Project Structure
Air Draw Recog/
│
├── draw_dataset/ # Collected digit images (0–9 folders)
├── collect_draw_data_mouse.py # Collect digits using mouse
├── collect_air_draw_data.py # Collect digits using air drawing
├── train_cnn_from_drawn_data.py # Train CNN from scratch
├── finetune_cnn_on_drawn_data.py# Fine-tune CNN on collected data
├── predict_drawn_digit.py # Predict digit from drawn image
├── predict_air_draw.py # Real-time air draw prediction
├── hand_landmarker.task # MediaPipe hand model
├── README.md
└── .gitignore


## 🧪 How It Works

1. **Data Collection**
   - Digits are collected by drawing (mouse or air).
   - Images are saved in class-wise folders (`0` to `9`).

2. **Model Training**
   - A CNN is trained on the collected dataset.
   - Data augmentation is used to improve generalization.
   - Class imbalance is handled using weighted sampling.

3. **Prediction**
   - Drawn digit is resized to `28×28`
   - Model outputs probabilities
   - If confidence is low → result marked as *Uncertain*

---

## ▶️ How to Run

### 1️⃣ Create & activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate

2️⃣ Install dependencies
pip install torch torchvision opencv-python mediapipe numpy

3️⃣ Collect data
python collect_draw_data_mouse.py
# or
python collect_air_draw_data.py

4️⃣ Train the CNN
python train_cnn_from_drawn_data.py

5️⃣ Run air-draw prediction
python predict_air_draw.py

📈 Accuracy Notes

Accuracy improves significantly with more personal samples
Misclassification (like 1 vs 8) was reduced using:
Data augmentation
Class weighting
Confidence thresholding
Final model performs best on user-style drawings, not generic MNIST
