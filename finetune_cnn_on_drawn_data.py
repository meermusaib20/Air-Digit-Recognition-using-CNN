import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Dataset path
DATA_DIR = "draw_dataset"

# Transform (same size & normalization as MNIST)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load drawn digit dataset
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

print("Total drawn samples:", len(dataset))

loader = DataLoader(dataset, batch_size=32, shuffle=True)

# CNN architecture (same as before)
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

# Load pretrained CNN
model = DigitCNN().to(device)
model.load_state_dict(torch.load("digit_cnn_model.pth", map_location=device))
model.train()

# Loss & optimizer (low LR for fine-tuning)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# Fine-tuning
epochs = 5

for epoch in range(epochs):
    running_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}] - Loss: {running_loss / len(loader):.4f}")

# Save fine-tuned model
torch.save(model.state_dict(), "digit_cnn_drawn_finetuned.pth")
print("Fine-tuned model saved as digit_cnn_drawn_finetuned.pth")
