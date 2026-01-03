import cv2
import numpy as np
import os

# Create dataset folders
BASE_DIR = "draw_dataset"
for i in range(10):
    os.makedirs(os.path.join(BASE_DIR, str(i)), exist_ok=True)

drawing = False
canvas = np.zeros((400, 400), dtype=np.uint8)

print("Mouse Draw Data Collection")
print("Draw digit using mouse")
print("Press 0–9 to save")
print("c -> clear | q -> quit")


def draw(event, x, y, flags, param):
    global drawing, canvas

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.circle(canvas, (x, y), 8, 255, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def preprocess(canvas):
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


cv2.namedWindow("Draw Digit")
cv2.setMouseCallback("Draw Digit", draw)

while True:
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    cv2.putText(
        display,
        "Draw digit | 0-9 save | c clear | q quit",
        (10, 390),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1
    )

    cv2.imshow("Draw Digit", display)

    key = cv2.waitKey(10) & 0xFF

    if key == ord('c'):
        canvas[:] = 0

    if key == ord('q'):
        break

    if ord('0') <= key <= ord('9'):
        label = chr(key)
        processed = preprocess(canvas)

        if processed is not None:
            count = len(os.listdir(os.path.join(BASE_DIR, label)))
            filename = f"{label}_{count}.png"
            path = os.path.join(BASE_DIR, label, filename)
            cv2.imwrite(path, processed)
            print(f"Saved: {path}")

        canvas[:] = 0


cv2.destroyAllWindows()
print("Dataset collection finished")
