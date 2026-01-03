import cv2
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------
# CNN model (must match training architecture)
# -------------------------------------------------

class DigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# -------------------------------------------------
# Load trained stable model
# -------------------------------------------------

model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn_stable.pth", map_location="cpu"))
model.eval()

# -------------------------------------------------
# MediaPipe HandLandmarker (Tasks API – stable)
# -------------------------------------------------

base_options = python.BaseOptions(
    model_asset_path="hand_landmarker.task"
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = vision.HandLandmarker.create_from_options(options)

# -------------------------------------------------
# Webcam and canvas
# -------------------------------------------------

cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640), dtype=np.uint8)

prev_x, prev_y = None, None
prediction = None
cooldown = 0  # prevents repeated predictions

# -------------------------------------------------
# Preprocess drawn digit (MNIST style)
# -------------------------------------------------

def preprocess(img):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return None

    digit = img[ys.min():ys.max()+1, xs.min():xs.max()+1]

    h, w = digit.shape
    size = max(h, w) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    y_off = (size - h) // 2
    x_off = (size - w) // 2
    square[y_off:y_off+h, x_off:x_off+w] = digit

    square = cv2.resize(square, (28, 28))
    square = square / 255.0
    square = (square - 0.5) / 0.5

    return torch.tensor(square).unsqueeze(0).unsqueeze(0).float()

# -------------------------------------------------
# Gesture detection: pinch (thumb + index finger)
# -------------------------------------------------

def is_pinch(lm1, lm2, threshold=0.04):
    dx = lm1.x - lm2.x
    dy = lm1.y - lm2.y
    return (dx * dx + dy * dy) < threshold

print("Air Draw Started")
print("Draw with index finger")
print("Pinch (thumb + index) to predict")
print("c → clear | q → quit")

# -------------------------------------------------
# Main loop
# -------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = detector.detect(mp_image)

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]

        index_tip = landmarks[8]
        thumb_tip = landmarks[4]

        x = int(index_tip.x * frame.shape[1])
        y = int(index_tip.y * frame.shape[0])

        cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 8)

        prev_x, prev_y = x, y

        # Gesture-based prediction (pinch)
        if is_pinch(thumb_tip, index_tip) and cooldown == 0:
            processed = preprocess(canvas)
            if processed is not None:
                with torch.no_grad():
                    out = model(processed)
                    probs = torch.softmax(out, dim=1)
                    conf, pred = torch.max(probs, 1)

                    prediction = pred.item() if conf.item() > 0.45 else "Uncertain"
                    cooldown = 20  # debounce

    else:
        prev_x, prev_y = None, None

    if cooldown > 0:
        cooldown -= 1

    combined = cv2.add(frame, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    if prediction is not None:
        cv2.putText(
            combined,
            f"Prediction: {prediction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.putText(
        combined,
        "Air Draw | Pinch to predict | c clear | q quit",
        (10, 470),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    cv2.imshow("Air Draw Digit Recognition", combined)
    key = cv2.waitKey(10) & 0xFF

    if key == ord('c'):
        canvas[:] = 0
        prediction = None

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Air draw closed cleanly")
