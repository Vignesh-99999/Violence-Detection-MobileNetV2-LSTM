import tensorflow as tf
import numpy as np, cv2, os

# ==========================================================
# Config
# ==========================================================
MODEL_PATH = "violence_detector_mobilenetv2.keras"
VIDEO_PATH = "NV_169.mp4"
IMG_SIZE = 112
FRAMES_PER_CLIP = 16

# ==========================================================
# Load clip exactly like training
# ==========================================================
def load_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < FRAMES_PER_CLIP and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if len(frame.shape) == 2 or frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype("float32") / 255.0
        frames.append(frame)
    cap.release()
    if not frames:
        raise ValueError("No frames read from video")
    while len(frames) < FRAMES_PER_CLIP:
        frames.append(frames[-1])
    return np.array(frames[:FRAMES_PER_CLIP], dtype=np.float32)

# ==========================================================
# Load model and predict
# ==========================================================
model = tf.keras.models.load_model(MODEL_PATH)

clip = load_video(VIDEO_PATH)
clip = np.expand_dims(clip, axis=0)
pred = model.predict(clip, verbose=0)[0][0]
label_flag = pred > 0.5
confidence = (pred if label_flag else 1 - pred) * 100
label = "Violence" if label_flag else "Non‑Violence"

print(f"{VIDEO_PATH} → {label} ({confidence:.1f}%)")

# ==========================================================
# Display video with colored box and label
# ==========================================================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
color = (0, 0, 255) if label_flag else (0, 255, 0)   # red / green
text = f"{label}  {confidence:.0f}%"                 # Violence 90%

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    # Draw box around entire ROI
    x1, y1, x2, y2 = w // 4, h // 4, 3 * w // 4, 3 * h // 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

    # Thick label background
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
    # Label text
    cv2.putText(frame, text, (x1 + 5, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Violence Detection", frame)
    key = cv2.waitKey(int(1000 / fps))
    if key == 27:  # Esc to stop
        break

cap.release()
cv2.destroyAllWindows()