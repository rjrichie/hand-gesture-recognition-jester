import re
import matplotlib.pyplot as plt
import pandas as pd
import os

# Path to your training log file
log_path = "train_log.txt"  # put this file in your base directory
os.makedirs("results", exist_ok=True)

# Read the entire file as one string (since everything is in one long line)
with open(log_path, "r") as f:
    text = f.read()

# Regex pattern that finds all epochs in the continuous text
pattern = re.compile(
    r"Epoch\s*\[(\d+)/\d+\]\s*Train Loss:\s*([\d.]+),\s*Train Acc:\s*([\d.]+)\s*\|\s*Val Loss:\s*([\d.]+),\s*Val Acc:\s*([\d.]+)"
)

epochs, train_loss, val_loss, train_acc, val_acc = [], [], [], [], []

for match in pattern.finditer(text):
    epochs.append(int(match.group(1)))
    train_loss.append(float(match.group(2)))
    train_acc.append(float(match.group(3)))
    val_loss.append(float(match.group(4)))
    val_acc.append(float(match.group(5)))

# Convert to dataframe
df = pd.DataFrame({
    "Epoch": epochs,
    "Train Loss": train_loss,
    "Val Loss": val_loss,
    "Train Acc": train_acc,
    "Val Acc": val_acc
})

df.to_csv("results/training_metrics.csv", index=False)
print("✅ Saved extracted metrics to results/training_metrics.csv")

# -----------------------------
# Plot 1: Loss Curves
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Epoch"], df["Train Loss"], 'o-', label="Train Loss")
plt.plot(df["Epoch"], df["Val Loss"], 's--', label="Validation Loss")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png", dpi=300)
plt.close()
print("📉 Saved results/loss_curve.png")

# -----------------------------
# Plot 2: Accuracy Curves
# -----------------------------
plt.figure(figsize=(8,5))
plt.plot(df["Epoch"], df["Train Acc"], 'o-', label="Train Accuracy")
plt.plot(df["Epoch"], df["Val Acc"], 's--', label="Validation Accuracy")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/accuracy_curve.png", dpi=300)
plt.close()
print("📈 Saved results/accuracy_curve.png")
