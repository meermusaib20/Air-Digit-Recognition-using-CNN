import cv2
import numpy as np
import mediapipe as mp
import os

# Create dataset folders
BASE_DIR = "air_draw_dataset"
for i in range(10):
    os.makedirs(os.path.join(BASE_DIR, str(i)), exist_ok=True)

# MediaPipe Hands (classic, stable)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

canvas = np.zeros((480, 640), dtype=np.uint8)
prev_x, prev_y = None, None

print("Air-Draw Data Collection")
print("Draw digit in air")
print("Press 0–9 to save")
print("c → clear | q → quit")


def preprocess_canvas(canvas):
    ys, xs = np.where(canvas > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    digit = canvas[y_min:y_max + 1, x_min:x_max + 1]

    digit = cv2.dilate(digit, np.ones((3, 3), np.uint8), iterations=1)

    h, w = digit.shape
    size = max(h, w) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset + h, x_offset:x_offset + w] = digit

    square = cv2.resize(square, (28, 28))
    return square


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        lm = hand_landmarks.landmark[8]  # index finger tip

        x = int(lm.x * w)
        y = int(lm.y * h)

        if prev_x is not None:
            cv2.line(canvas, (prev_x, prev_y), (x, y), 255, 10)

        prev_x, prev_y = x, y
    else:
        prev_x, prev_y = None, None

    combined = cv2.add(frame, cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR))

    cv2.putText(
        combined,
        "Draw digit | press 0-9 to save | c: clear | q: quit",
        (10, 460),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    cv2.imshow("Collect Air-Draw Data", combined)

    key = cv2.waitKey(5) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == ord('q'):
        break

    if ord('0') <= key <= ord('9'):
        label = chr(key)
        processed = preprocess_canvas(canvas)

        if processed is not None:
            count = len(os.listdir(os.path.join(BASE_DIR, label)))
            filename = f"{label}_{count}.png"
            path = os.path.join(BASE_DIR, label, filename)
            cv2.imwrite(path, processed)
            print(f"Saved: {path}")

        canvas[:] = 0


cap.release()
hands.close()
cv2.destroyAllWindows()
print("Data collection finished")
