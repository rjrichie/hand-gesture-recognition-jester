import os
import torch
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import random


class JesterDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, num_frames=32, transform=None, train=False):
        """
        Args:
            csv_file (str): Path to CSV file with 'gesture_dir' and 'label'.
            root_dir (str): Root directory containing gesture folders.
            num_frames (int): Number of frames to sample per video.
            transform (callable, optional): Transform to apply to each frame.
            train (bool): If True, apply training augmentations. If False, apply validation transforms.
        """
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        uniq = sorted(self.data['label'].unique().tolist())
        self.label2id = {lab: i for i, lab in enumerate(uniq)}
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.train = train
        
        # Use default transforms if none provided
        if transform is None:
            self.use_default_transforms = True
        else:
            self.use_default_transforms = False
            self.transform = transform

    def __len__(self):
        return len(self.data)

    def apply_consistent_transforms(self, frames):
        """Apply the same random transformation to all frames in the video."""
        if not self.train:
            # Validation: deterministic transforms
            frames = [TF.resize(f, (128, 128)) for f in frames]
            frames = [TF.to_tensor(f) for f in frames]
            return frames
        
        # Training: random but consistent transforms across all frames
        
        # 1. Resize to larger size for cropping
        frames = [TF.resize(f, (144, 144)) for f in frames]
        
        # 2. Random crop - same crop for all frames
        i, j, h, w = torchvision.transforms.RandomCrop.get_params(
            frames[0], output_size=(128, 128)
        )
        frames = [TF.crop(f, i, j, h, w) for f in frames]
        
        # 3. Color jitter - same parameters for all frames
        brightness_factor = random.uniform(0.8, 1.2)
        contrast_factor = random.uniform(0.8, 1.2)
        saturation_factor = random.uniform(0.8, 1.2)
        hue_factor = random.uniform(-0.1, 0.1)
        
        frames = [TF.adjust_brightness(f, brightness_factor) for f in frames]
        frames = [TF.adjust_contrast(f, contrast_factor) for f in frames]
        frames = [TF.adjust_saturation(f, saturation_factor) for f in frames]
        frames = [TF.adjust_hue(f, hue_factor) for f in frames]
        
        # 4. Random rotation - same angle for all frames
        angle = random.uniform(-5, 5)
        frames = [TF.rotate(f, angle) for f in frames]
        
        # 5. Convert to tensor
        frames = [TF.to_tensor(f) for f in frames]
        
        return frames

    def __getitem__(self, idx):
        item_data = self.data.iloc[idx]
        
        gesture_dir = str(item_data['gesture_dir'])
        label = self.label2id[int(item_data['label'])]
        label = torch.tensor(label, dtype=torch.long)

        folder_path = os.path.join(self.root_dir, gesture_dir)

        # Get sorted list of frames
        frame_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.jpg') and not f.startswith('._')
        ])

        # Uniformly sample num_frames
        total_frames = int(item_data['num_frames'])
        if total_frames >= self.num_frames:
            # Uniform sampling indices
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        else:
            # If video is shorter, repeat last frame
            indices = torch.cat([
                torch.arange(total_frames),
                torch.full((self.num_frames - total_frames,), total_frames - 1)
            ]).long()

        # Load all frames first
        frames = []
        for i in indices:
            img = Image.open(frame_paths[i]).convert('RGB')
            frames.append(img)

        # Apply consistent transforms to all frames
        if self.use_default_transforms:
            frames = self.apply_consistent_transforms(frames)
        else:
            # If custom transform provided, assume it handles consistency
            frames = [self.transform(f) for f in frames]

        # Stack frames into tensor: [num_frames, C, H, W]
        video_tensor = torch.stack(frames)

        # Permute to [C, D, H, W] for 3D CNN
        video_tensor = video_tensor.permute(1, 0, 2, 3)

        return video_tensor, label
