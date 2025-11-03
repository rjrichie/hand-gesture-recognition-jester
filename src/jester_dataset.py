import os
import torch
from PIL import Image
import torchvision

class JesterDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, num_frames=32, transform=None):
        """
        Args:
            csv_file (str): Path to CSV file with 'gesture_dir' and 'label'.
            root_dir (str): Root directory containing gesture folders.
            num_frames (int): Number of frames to sample per video.
            transform (callable, optional): Transform to apply to each frame.
        """
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            torchvision.transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.data.iloc[idx]
        
        gesture_dir = str(item_data['gesture_dir'])
        label = int(item_data['label'])

        folder_path = os.path.join(self.root_dir, gesture_dir)

        # Get sorted list of frames
        frame_paths = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.jpg')])

        # Uniformly sample num_frames
        total_frames = int(item_data['num_frames'])
        if total_frames >= self.num_frames:
            # Uniform sampling indices
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()

        # Load frames
        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert('RGB')
            img = self.transform(img)
            frames.append(img)

        # Stack frames into tensor: [num_frames, C, H, W]
        video_tensor = torch.stack(frames)

        # Permute to [C, D, H, W] for 3D CNN
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor, label
