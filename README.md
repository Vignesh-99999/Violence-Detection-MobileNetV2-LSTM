# 🎥 Violence Detection in Videos using MobileNetV2 + LSTM

This repository provides an **end-to-end deep learning pipeline** for **violence detection in video clips** using **transfer learning with MobileNetV2** and **temporal modeling with LSTM**.

The project includes:

- 📦 Dataset loading & preprocessing
- 🧠 Model creation and training
- 💾 Model saving/loading
- 🎯 Video-level inference
- 🖼️ Real-time visualization with OpenCV bounding box and confidence score

---

## 📌 Features

- Uses **MobileNetV2 (ImageNet-pretrained)** for spatial feature extraction
- Uses **Bi-Directional LSTM** for temporal sequence learning
- Handles variable-length videos using **frame padding**
- Supports multiple video formats (`.mp4`, `.avi`, `.mov`, `.mkv`)
- Lightweight and deployable
- OpenCV visualization with **colored alerts**

  - 🔴 Red → Violence
  - 🟢 Green → Non-Violence

---

## 🧠 Model Architecture

```
Input Video (16 frames, 112×112×3)
        │
TimeDistributed MobileNetV2
        │
Global Average Pooling
        │
Bi-Directional LSTM (128)
        │
Dense (64) + ReLU
        │
Dropout (0.4)
        │
Dense (1) + Sigmoid
        │
Binary Classification
```

---

## 📂 Repository Structure

```
.
├── violence_classification_mobilenetv2.py   # Training & model creation
├── inference_and_visualization.py           # Inference + OpenCV display
├── violence_detector_mobilenetv2.keras      # Trained model
├── violence_dataset/
│   ├── violence/
│   │   ├── v1.mp4
│   │   ├── v2.mp4
│   └── non_violence/
│       ├── nv1.mp4
│       ├── nv2.mp4
├── requirements.txt
└── README.md
```

---

## 🧪 Dataset Format

Organize your dataset as:

```
violence_dataset/
├── violence/
│   ├── video1.mp4
│   ├── video2.mp4
└── non_violence/
    ├── video1.mp4
    ├── video2.mp4
```

Each video is automatically:

- Sampled to **16 frames**
- Resized to **112×112**
- Normalized to `[0,1]`

---

## ⚙️ Configuration

Key parameters (modifiable in code):

```python
IMG_SIZE = 112
FRAMES_PER_CLIP = 16
EPOCHS = 10
BATCH_SIZE = 4
MODEL_PATH = "violence_detector_mobilenetv2.keras"
```

---

## 🚀 Installation

### 1️⃣ Clone Repository

```bash
git clone https://github.com/yourusername/violence-detection-video.git
cd violence-detection-video
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

- TensorFlow
- OpenCV
- NumPy

Example `requirements.txt`:

```
tensorflow>=2.10
opencv-python
numpy
```

---

## 🏋️ Model Training

Run the training script:

```bash
python violence_classification_mobilenetv2.py
```

### Training Logic

- Automatically checks if model exists
- Trains **only if no saved model is found**
- Saves model as:

```
violence_detector_mobilenetv2.keras
```

### Output Example

```
Loaded dataset: (120, 16, 112, 112, 3)
Epoch 1/10
accuracy: 0.84 - val_accuracy: 0.87
Model trained and saved
```

---

## 📊 Model Evaluation

Validation accuracy is printed automatically after training:

```text
Validation Accuracy: 89.25%
```

---

## 🎯 Video Inference (Prediction)

To predict violence in a new video:

```python
result = predict_video("test_clip.mp4")
```

### Output

```
test_clip.mp4 → Violence (confidence=0.923)
```

---

## 🖼️ Real-Time Visualization

Run:

```bash
python inference_and_visualization.py
```

### Display Features

- Bounding box overlay
- Confidence percentage
- Color-coded alert

  - 🔴 Violence
  - 🟢 Non-Violence

- Press **ESC** to exit

Example:

```
NV_169.mp4 → Non-Violence (92%)
```

---

## 📈 Performance Notes

- Lightweight CNN (MobileNetV2)
- Suitable for **edge devices** with optimization
- Accuracy depends on:

  - Dataset size
  - Class balance
  - Video quality

- Can be improved using:

  - Fine-tuning MobileNetV2
  - More frames per clip
  - Data augmentation

---

## 🔧 Future Improvements

- Multi-class action recognition
- Frame-wise localization
- Temporal attention
- ONNX / TensorRT export
- Live webcam detection
- REST API deployment

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.
Predictions should **not** be used as the sole basis for security or legal decisions.

---

## 👨‍💻 Author

**Vignesh Mudaliyar**
Deep Learning | Computer Vision | Video Analytics

If you find this useful, ⭐ the repository!
