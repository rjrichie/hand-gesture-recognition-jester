import os
import pandas as pd
from tqdm import tqdm
from collections import Counter
from PIL import Image

# Load in data from txts
train_df = pd.read_csv('dataset/original/annotations/trainlist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])
test_df = pd.read_csv('dataset/original/annotations/testlist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])
val_df = pd.read_csv('dataset/original/annotations/vallist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])

# Combine all dirs
all_dirs = pd.concat([train_df, test_df, val_df])["gesture_dir"]

original_data_root = "dataset/original/data"

num_frames_list = []
video_dimensions = []

for gesture_dir in tqdm(all_dirs, desc="Analysing gesture data"):
    src_path = os.path.join(original_data_root, str(gesture_dir))
    
    if os.path.exists(src_path):
        # Count number of files in the directory
        files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
        num_frames = len(files)
        
        # Check first frame for video dimensions
        first_frame_path = os.path.join(src_path, "00001.jpg")
        if os.path.exists(first_frame_path):
            with Image.open(first_frame_path) as img:
                video_dimensions.append(img.size)  # size returns (width, height)
    else:
        num_frames = 0
    
    num_frames_list.append(num_frames)

# Get frame info
num_frames_counter = Counter(num_frames_list)
print("Number of videos with different frame counts:")
for num_frames, count in sorted(num_frames_counter.items()):
    print(f"{num_frames} frames: {count} videos")

# Get dimension info
dimensions_counter = Counter(video_dimensions)
print("All types of dimensions for images:")
for dimensions, count in sorted(dimensions_counter.items()):
    print(f"{dimensions} files: {count} videos")
