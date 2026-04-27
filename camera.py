import os
import json
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from collections import deque, Counter

# ================= PATH =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "classes.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= LOAD =================
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

model = models.mobilenet_v2(weights=None)
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

# ================= SETTINGS =================
CONF_THRESHOLD = 0.75        # stricter = fewer wrong predictions
BUFFER_SIZE = 12            # more frames = more stable
STABLE_REQUIRED = 7         # votes needed to confirm

pred_buffer = deque(maxlen=BUFFER_SIZE)
stable_label = "Show object"

# ================= CAMERA =================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # ===== VERY TIGHT CENTER CROP =====
    size = int(min(h, w) * 0.25)   # tighter crop = better accuracy
    cx, cy = w // 2, h // 2

    crop = frame[cy-size:cy+size, cx-size:cx+size]

    # Zoom crop (simulate closer object)
    crop = cv2.resize(crop, (224, 224))

    # Draw guide box
    cv2.rectangle(frame,
                  (cx-size, cy-size),
                  (cx+size, cy+size),
                  (255, 0, 0), 2)

    # OPTIONAL debug view
    cv2.imshow("CROP", crop)

    # ===== PREPROCESS =====
    img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    tensor = transform(img).unsqueeze(0).to(DEVICE)

    # ===== PREDICT =====
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    confidence = conf.item()
    label = classes[pred.item()]

    # ===== CONFIDENCE FILTER =====
    if confidence < CONF_THRESHOLD:
        label = "Uncertain"

    # ===== BUFFER (multi-frame voting) =====
    pred_buffer.append(label)

    most_common, count = Counter(pred_buffer).most_common(1)[0]

    if count >= STABLE_REQUIRED:
        stable_label = most_common
    else:
        stable_label = "Hold steady..."

    # ===== DISPLAY =====
    text = f"{stable_label} ({confidence*100:.1f}%)"

    cv2.putText(frame, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2)

    cv2.imshow("EcoBot", frame)

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
