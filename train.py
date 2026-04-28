import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import matplotlib.pyplot as plt

from utils.dataloader import get_loaders


# ================= CONFIG =================
data_dir = r"C:\Users\HP\OneDrive\Desktop\Material\Experimental design project\eco_bot\data"
epochs = 50
lr = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================= LOAD DATA =================
train_loader, val_loader, test_loader, classes = get_loaders(data_dir)
print("Classes:", classes)


# ================= MODEL =================
model = models.mobilenet_v2(pretrained=True)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Replace classifier
model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(device)


# ================= LOSS & OPTIMIZER =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


# ================= METRIC STORAGE =================
train_losses = []
train_accuracies = []
val_accuracies = []


# ================= TRAIN FUNCTION =================
def train_one_epoch():
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    acc = 100 * correct / total
    return total_loss / len(train_loader), acc


# ================= VALIDATION =================
def validate():
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    return acc


# ================= TRAIN LOOP =================
best_acc = 0

for epoch in range(epochs):
    train_loss, train_acc = train_one_epoch()
    val_acc = validate()

    # Store metrics
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"\nEpoch {epoch+1}/{epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Acc : {train_acc:.2f}%")
    print(f"Val Acc   : {val_acc:.2f}%")

    # Save best model
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Model saved!")


print("\n🎯 Training Complete")


# ================= GRAPH PLOTTING =================

# Loss Graph
plt.figure()
plt.plot(train_losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.savefig("loss.png")

# Accuracy Graph
plt.figure()
plt.plot(train_accuracies, label="Train Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.title("Accuracy vs Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("accuracy.png")

print("📊 Graphs saved as loss.png and accuracy.png")
