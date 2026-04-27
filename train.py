import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# ================= CONFIG =================
DATA_DIR = "data"
EPOCHS = 15
LR = 0.0003

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= TRANSFORMS =================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= DATA =================
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
val_data = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transform)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=32)

classes = train_data.classes
print("Classes:", classes)

# save class order
os.makedirs("models", exist_ok=True)
with open("models/classes.json", "w") as f:
    json.dump(classes, f)

# ================= MODEL =================
model = models.mobilenet_v2(weights="DEFAULT")

# 🔥 DO NOT freeze everything
for param in model.features.parameters():
    param.requires_grad = True

model.classifier[1] = nn.Linear(model.last_channel, len(classes))
model = model.to(DEVICE)

# ================= LOSS =================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ================= TRAIN =================
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # ===== VALIDATION =====
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Acc: {train_acc:.2f}%")
    print(f"Val Acc  : {val_acc:.2f}%")

    # SAVE BEST MODEL
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "models/best_model.pth")
        print("✅ Model Saved!")

print("\n🎯 Training Complete")
