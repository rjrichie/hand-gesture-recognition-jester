import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict

from jester_dataset import JesterDataset
from models.CNN3D import C3D  # adjust import if needed

# -----------------------------
# Configuration
# -----------------------------
val_csv = "dataset/modified/annotations/val.csv"   # or test.csv
root_dir = "dataset/modified/data"
checkpoint_path = "checkpoints/c3d_epoch16.pth"
num_classes = 9
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# Optional: gesture label mapping
label_map = [7, 8, 9, 10, 17, 20, 24, 26, 27]

# -----------------------------
# Load dataset and dataloader
# -----------------------------
val_dataset = JesterDataset(csv_file=val_csv, root_dir=root_dir, num_frames=32)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Load model
# -----------------------------
print("🔧 Loading model and checkpoint...")
model = C3D(num_classes=num_classes, sample_size=128, sample_duration=32)
checkpoint = torch.load(checkpoint_path, map_location=device)

# Handle DataParallel checkpoint
state_dict = checkpoint.get('model_state_dict', checkpoint)
new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())

model.load_state_dict(new_state_dict)
model = model.to(device)
model.eval()
print("✅ Model loaded successfully.\n")

# -----------------------------
# Evaluation loop
# -----------------------------
criterion = nn.CrossEntropyLoss()
total, correct = 0, 0
all_preds, all_labels = [], []
running_loss = 0.0

print("🚀 Evaluating model...")
with torch.no_grad():
    for videos, labels in val_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# -----------------------------
# Metrics
# -----------------------------
accuracy = 100 * correct / total
avg_loss = running_loss / len(val_loader)

print("📊 Evaluation Results")
print("=" * 50)
print(f"Validation Loss     : {avg_loss:.4f}")
print(f"Validation Accuracy : {accuracy:.2f}%")
print("=" * 50)

print("\n🧠 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(x) for x in label_map], digits=3))

# -----------------------------
# Confusion Matrix Plot
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=label_map, yticklabels=label_map)
plt.title("Confusion Matrix (Normalized)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
