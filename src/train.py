import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
from jester_dataset import JesterDataset
from models.CNN3D import C3D

# -----------------------------
# Parameters
# -----------------------------
train_csv = "dataset/modified/annotations/train.csv"
val_csv = "dataset/modified/annotations/val.csv"
root_dir = "dataset/modified/data"

batch_size = 4
num_epochs = 20
learning_rate = 1e-4
num_classes = 9
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "mps")

# -----------------------------
# Datasets and Dataloaders
# -----------------------------
train_dataset = JesterDataset(csv_file=train_csv, root_dir=root_dir, num_frames=32)
val_dataset = JesterDataset(csv_file=val_csv, root_dir=root_dir, num_frames=32)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Model, loss, optimizer
# -----------------------------
model = C3D(sample_size=128, sample_duration=32, num_classes=num_classes)
model = model.to(device)
model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# -----------------------------
# Resume from latest checkpoint if exists
# -----------------------------
start_epoch = 1
checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
if checkpoint_files:
    latest_ckpt = max(checkpoint_files, key=lambda x: int(x.split("epoch")[1].split(".")[0]))
    ckpt_path = os.path.join(checkpoint_dir, latest_ckpt)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resumed from checkpoint: {ckpt_path} (starting epoch {start_epoch})")

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    train_loader_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training", leave=False)
    for videos, labels in train_loader_iter:
        videos = videos.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        train_loader_iter.set_postfix({"loss": running_loss / total, "acc": correct / total})

    train_loss = running_loss / total
    train_acc = correct / total

    # -----------------------------
    # Validation
    # -----------------------------
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_loader_iter = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation", leave=False)
    with torch.no_grad():
        for videos, labels in val_loader_iter:
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * videos.size(0)
            preds = outputs.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

            val_loader_iter.set_postfix({"loss": val_loss / val_total, "acc": val_correct / val_total})

    val_loss /= val_total
    val_acc = val_correct / val_total

    print(f"Epoch [{epoch}/{num_epochs}] "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # -----------------------------
    # Save checkpoint
    # -----------------------------
    checkpoint_path = os.path.join(checkpoint_dir, f"c3d_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_acc': train_acc,
        'val_loss': val_loss,
        'val_acc': val_acc
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}\n")
