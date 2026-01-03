import cv2
import numpy as np
import torch
import torch.nn as nn

# -------------------------------------------------
# This is the same CNN architecture used during training.
# It is important that training and prediction models match.
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
# Load the trained model from disk
# -------------------------------------------------

model = DigitCNN()
model.load_state_dict(torch.load("digit_cnn_stable.pth", map_location="cpu"))
model.eval()

# -------------------------------------------------
# Create a blank canvas where the user will draw
# -------------------------------------------------

canvas = np.zeros((400, 400), dtype=np.uint8)
drawing = False
prediction = None

# -------------------------------------------------
# Mouse callback function for drawing
# -------------------------------------------------

def draw(event, x, y, flags, param):
    global drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        # Small brush size helps keep digit 1 thin
        cv2.circle(canvas, (x, y), 7, 255, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# -------------------------------------------------
# Preprocess the drawn digit
# This keeps aspect ratio and centers the digit
# similar to how MNIST images are prepared
# -------------------------------------------------

def preprocess(img):
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return None

    digit = img[ys.min():ys.max()+1, xs.min():xs.max()+1]

    h, w = digit.shape
    size = max(h, w) + 20
    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2
    square[y_offset:y_offset+h, x_offset:x_offset+w] = digit

    square = cv2.resize(square, (28, 28))

    # Normalize exactly the same way as training
    square = square / 255.0
    square = (square - 0.5) / 0.5

    tensor = torch.tensor(square).unsqueeze(0).unsqueeze(0).float()
    return tensor

# -------------------------------------------------
# Setup OpenCV window and mouse callback
# -------------------------------------------------

cv2.namedWindow("Digit Predictor")
cv2.setMouseCallback("Digit Predictor", draw)

print("Draw a digit using the mouse")
print("p → predict | c → clear | q → quit")

# -------------------------------------------------
# Main application loop
# -------------------------------------------------

while True:
    display = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if prediction is not None:
        cv2.putText(
            display,
            f"Prediction: {prediction}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3
        )

    cv2.putText(
        display,
        "Draw | p predict | c clear | q quit",
        (10, 380),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2
    )

    cv2.imshow("Digit Predictor", display)
    key = cv2.waitKey(10) & 0xFF

    # When user presses 'p', run prediction
    if key == ord('p'):
        processed = preprocess(canvas)
        if processed is not None:
            with torch.no_grad():
                output = model(processed)

                # Convert raw outputs to probabilities
                probs = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probs, 1)

                # If confidence is low, say "Uncertain"
                if confidence.item() < 0.50:

                    prediction = "Uncertain"
                else:
                    prediction = predicted_class.item()

    # Clear the canvas
    if key == ord('c'):
        canvas[:] = 0
        prediction = None

    # Quit the program
    if key == ord('q'):
        break

cv2.destroyAllWindows()
print("Program exited cleanly")
print("Confidence:", confidence.item())

