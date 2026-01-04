# violence_classification_mobilenetv2.py
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np, cv2, os

# ==========================================================
# Configuration
# ==========================================================
DATA_DIR = "violence_dataset"      # contains 'violence/' and 'non_violence/' folders
IMG_SIZE = 112
FRAMES_PER_CLIP = 16
EPOCHS = 10
MODEL_PATH = "violence_detector_mobilenetv2.keras"

# ==========================================================
# Helper: load and preprocess video clips
# ==========================================================
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < FRAMES_PER_CLIP and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Ensure RGB (3 channels)
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
    cap.release()

    # pad or trim to consistent length
    if len(frames) == 0:
        return np.zeros((FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    while len(frames) < FRAMES_PER_CLIP:
        frames.append(frames[-1])
    return np.array(frames[:FRAMES_PER_CLIP], dtype=np.float32)

def make_dataset(root):
    clips, labels = [], []
    classes = ['non_violence', 'violence']
    for label, folder in enumerate(classes):
        folder_path = os.path.join(root, folder)
        for fname in os.listdir(folder_path):
            if fname.lower().endswith(('.mp4','.avi','.mov','.mkv')):
                clips.append(load_video(os.path.join(folder_path, fname)))
                labels.append(label)
    return np.array(clips), np.array(labels)

# ==========================================================
# Load dataset
# ==========================================================
X, y = make_dataset(DATA_DIR)
print("Loaded dataset:", X.shape, y.shape)

split = int(0.8 * len(X))
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ==========================================================
# Build model – Transfer Learning with MobileNetV2
# ==========================================================
base_cnn = tf.keras.applications.MobileNetV2(
    include_top=False, weights="imagenet",
    pooling="avg", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base_cnn.trainable = False   # freeze base weights

model = models.Sequential([
    layers.TimeDistributed(base_cnn, input_shape=(FRAMES_PER_CLIP, IMG_SIZE, IMG_SIZE, 3)),
    layers.Bidirectional(layers.LSTM(128)),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ==========================================================
# Train (or load if already trained)
# ==========================================================
if not os.path.exists(MODEL_PATH):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=4
    )
    model.save(MODEL_PATH)
    print(" Model trained and saved:", MODEL_PATH)
else:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(" Loaded existing trained model:", MODEL_PATH)

# ==========================================================
# Evaluate Accuracy
# ==========================================================
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# ==========================================================
# Inference on New Video
# ==========================================================
def predict_video(video_path):
    clip = load_video(video_path)
    clip = np.expand_dims(clip, axis=0)     # (1,16,112,112,3)
    pred = model.predict(clip, verbose=0)[0][0]
    label = "Violence" if pred > 0.5 else "Non‑Violence"
    print(f"{video_path} → {label} (confidence={pred:.3f})")
    return label

# Example usage:
# result = predict_video("test_clip.mp4")