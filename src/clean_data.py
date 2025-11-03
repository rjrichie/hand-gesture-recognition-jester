import os
import shutil
import pandas as pd
from sklearn.utils import shuffle
from tqdm import tqdm
from PIL import Image

random_seed = 0
original_data_root = "dataset/original/data"
modified_data_root = "dataset/modified/data"

# Load in data from txts
train_df = pd.read_csv('dataset/original/annotations/trainlist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])
test_df = pd.read_csv('dataset/original/annotations/testlist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])
val_df = pd.read_csv('dataset/original/annotations/vallist01.txt', sep=' ', header=None, names=['gesture_dir', 'label'])

# Enhance the data
def add_metadata(df, name):
    # Initialize new columns
    df["num_frames"] = 0
    df["height"] = None
    df["width"] = None

    for idx, video in tqdm(df.iterrows(), total=len(df), desc=f"Analysing {name} dataset"):
        src_path = os.path.join(original_data_root, str(video["gesture_dir"]))

        if os.path.exists(src_path):
            # Count number of files in the directory
            files = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
            num_frames = len(files)
            
            # Check first frame for video dimensions
            first_frame_path = os.path.join(src_path, "00001.jpg")
            if os.path.exists(first_frame_path):
                with Image.open(first_frame_path) as img:
                    video_dimensions = img.size  # (width, height)
            else:
                video_dimensions = None

            df.at[idx, "num_frames"] = num_frames
            df.at[idx, "height"] = video_dimensions[1]
            df.at[idx, "width"] = video_dimensions[0]

    return df

enhanced_train_df = add_metadata(train_df, "train")
enhanced_test_df = add_metadata(test_df, "test")
enhanced_val_df = add_metadata(val_df, "validation")

# Filter videos by number of frames (keep only 32 to 40)
frame_min, frame_max = 32, 40
filtered_train_df = enhanced_train_df[(enhanced_train_df['num_frames'] >= frame_min) & (enhanced_train_df['num_frames'] <= frame_max)]
filtered_test_df = enhanced_test_df[(enhanced_test_df['num_frames'] >= frame_min) & (enhanced_test_df['num_frames'] <= frame_max)]
filtered_val_df = enhanced_val_df[(enhanced_val_df['num_frames'] >= frame_min) & (enhanced_val_df['num_frames'] <= frame_max)]

# Remove all classes except for 7, 8, 9, 10, 17, 20, 24, 26, 27
classes_to_keep = [7, 8, 9, 10, 17, 20, 24, 26, 27]
filtered_train_df = filtered_train_df[filtered_train_df['label'].isin(classes_to_keep)]
filtered_test_df = filtered_test_df[filtered_test_df['label'].isin(classes_to_keep)]
filtered_val_df = filtered_val_df[filtered_val_df['label'].isin(classes_to_keep)]

# Reduce each class size to 1000 for training, 300 for testing and 30 for validation (randomly)
reduced_train_df = filtered_train_df.groupby('label').sample(n=1000, random_state=random_seed)
reduced_test_df = filtered_test_df.groupby('label').sample(n=300, random_state=random_seed)
reduced_val_df = filtered_val_df.groupby('label').sample(n=30, random_state=random_seed)

# Reshuffle
final_train_df = shuffle(reduced_train_df, random_state=random_seed)
final_test_df = shuffle(reduced_test_df, random_state=random_seed)
final_val_df = shuffle(reduced_val_df, random_state=random_seed)

# Save the csvs
final_train_df.to_csv("dataset/modified/annotations/train.csv", index = False)
final_test_df.to_csv("dataset/modified/annotations/test.csv", index = False)
final_val_df.to_csv("dataset/modified/annotations/val.csv", index = False)

# Create a new folder of all the image data with the reduced data
all_dirs = pd.concat([final_train_df, final_test_df, final_val_df])["gesture_dir"]

for gesture_dir in tqdm(all_dirs, desc="Copying gesture data"):
    src_path = os.path.join(original_data_root, str(gesture_dir))
    dst_path = os.path.join(modified_data_root, str(gesture_dir))
    
    # Ensure destination folder structure exists
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    # Copy folder
    if os.path.isdir(src_path):
        if not os.path.exists(dst_path):
            shutil.copytree(src_path, dst_path)
    else:
        print(f"Path not found: {src_path}")