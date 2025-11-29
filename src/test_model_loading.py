import torch
from torch.utils.data import DataLoader
from jester_dataset import JesterDataset  # Adjust import path if necessary
from models.CNN3D import C3D          # Adjust import path if necessary

# -----------------------------
# Parameters
# -----------------------------
csv_file = "dataset/modified/annotations/test.csv"  # Use train/test CSV
root_dir = "dataset/modified/data"
batch_size = 2  # small batch for testing
num_classes = 9

# -----------------------------
# Dataset and DataLoader
# -----------------------------
dataset = JesterDataset(csv_file=csv_file, root_dir=root_dir, num_frames=32)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Load C3D model
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = C3D(sample_size=128, sample_duration=32, num_classes=num_classes)
model = model.to(device)
model.eval()  # set to evaluation mode

# -----------------------------
# Run a single batch through the model
# -----------------------------
with torch.no_grad():
    for batch_idx, (videos, labels) in enumerate(dataloader):
        # videos: [B, 3, 32, 128, 128]
        videos = videos.to(device)
        labels = labels.to(device)

        outputs = model(videos)
        print("Video batch shape:", videos.shape)
        print("Model output shape:", outputs.shape)

        # Predicted class for each video
        preds = torch.argmax(outputs, dim=1)
        print("Predicted classes:", preds)
        print("Ground truth labels:", labels)

        break  # only process first batch
