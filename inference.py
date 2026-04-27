import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD CLASSES =================
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# ================= MODEL =================
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ================= TRANSFORM =================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# ================= PREDICT =================
def predict(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    return classes[pred.item()], conf.item()

# ================= MAIN =================
if __name__ == "__main__":
    path = input("Enter image path: ").strip().strip('"')

    label, conf = predict(path)

    print("\nPrediction:", label)
    print(f"Confidence: {conf*100:.2f}%")
