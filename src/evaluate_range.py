import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict

from jester_dataset import JesterDataset
from models.CNN3D import C3D

# -----------------------------
# Configuration
# -----------------------------
val_csv = "dataset/modified/annotations/val.csv"
root_dir = "dataset/modified/data"
checkpoint_dir = "checkpoints"
num_classes = 9
batch_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
label_map = [7, 8, 9, 10, 17, 20, 24, 26, 27]

# Dataset & loader
val_dataset = JesterDataset(csv_file=val_csv, root_dir=root_dir, num_frames=32)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Evaluation Loop for Multiple Epochs
# -----------------------------
for epoch in range(15, 21):
    checkpoint_path = os.path.join(checkpoint_dir, f"c3d_epoch{epoch}.pth")
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Skipping epoch {epoch}: checkpoint not found.")
        continue

    print(f"\n🔧 Loading model from epoch {epoch}...")
    model = C3D(num_classes=num_classes, sample_size=128, sample_duration=32)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle DataParallel
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    new_state_dict = OrderedDict((k.replace("module.", ""), v) for k, v in state_dict.items())
    model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    # Evaluate
    criterion = nn.CrossEntropyLoss()
    total, correct = 0, 0
    all_preds, all_labels = [], []
    running_loss = 0.0

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

    accuracy = 100 * correct / total
    avg_loss = running_loss / len(val_loader)
    print(f"✅ Epoch {epoch} — Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    # -----------------------------
    # Save results
    # -----------------------------
    os.makedirs("results", exist_ok=True)

    # 1️⃣ Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=label_map, yticklabels=label_map)
    plt.title(f"Confusion Matrix (Normalized) — Epoch {epoch}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    cm_path = os.path.join("results", f"confusion_matrix_epoch{epoch}.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2️⃣ Classification Report
    report = classification_report(
        all_labels, all_preds,
        target_names=[str(x) for x in label_map],
        digits=3,
        output_dict=True
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = os.path.join("results", f"classification_report_epoch{epoch}.csv")
    report_df.to_csv(report_path, index=True)

    # 3️⃣ Summary
    summary_path = os.path.join("results", f"summary_metrics_epoch{epoch}.txt")
    with open(summary_path, "w") as f:
        f.write(f"=== EPOCH {epoch} EVALUATION SUMMARY ===\n")
        f.write(f"Validation Loss     : {avg_loss:.4f}\n")
        f.write(f"Validation Accuracy : {accuracy:.2f}%\n")
        f.write("==========================\n\n")
        f.write("Per-Class Metrics (see CSV for details)\n")

    print(f"🖼️ Saved: {cm_path}")
    print(f"📄 Saved: {report_path}")
    print(f"🧾 Saved: {summary_path}")
