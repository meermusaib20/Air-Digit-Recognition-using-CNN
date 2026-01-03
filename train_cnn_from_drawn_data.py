import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler

torch.set_num_threads(1)
device = torch.device("cpu")

DATA_DIR = "draw_dataset"

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.1, 0.1),
        scale=(0.8, 1.1)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

targets = [label for _, label in dataset.samples]
class_counts = torch.bincount(torch.tensor(targets))
class_weights = 1.0 / class_counts.float()
sample_weights = [class_weights[t] for t in targets]

sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

loader = DataLoader(
    dataset,
    batch_size=8,
    sampler=sampler,
    num_workers=0
)

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

model = DigitCNN().to(device)

weights = torch.tensor(
    [1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)
criterion = nn.CrossEntropyLoss(weight=weights)


optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    loss_sum = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss_sum/len(loader):.4f}")

torch.save(model.state_dict(), "digit_cnn_stable.pth")
print("Saved model as digit_cnn_stable.pth")
print("Training complete.")