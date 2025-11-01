# hand-gesture-recognition-jester

Deep learning project for hand gesture recognition using the Jester dataset.

## Project Structure

```
hand-gesture-recognition-jester/
├── data/                    # Data directory (ignored by git)
│   ├── raw/                # Raw video data
│   ├── processed/          # Preprocessed data
│   └── interim/            # Intermediate data
├── src/                    # Source code
│   ├── data/               # Data preprocessing and loading
│   │   └── preprocessing.py
│   ├── models/             # Model architectures
│   ├── training/           # Training scripts
│   ├── evaluation/         # Evaluation utilities
│   └── utils/              # Utility functions
│       └── helpers.py
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## Setup

1. Clone the repository
2. Install dependencies (coming soon)
3. Download the Jester dataset and place it in `data/raw/`
4. Run preprocessing scripts

## Data Preprocessing

The `src/data/preprocessing.py` module provides classes for preprocessing video data:

- `VideoPreprocessor`: Handles frame extraction, resizing, and normalization
- `DatasetLoader`: Loads preprocessed data for training

## Usage

```python
from src.data.preprocessing import VideoPreprocessor, DatasetLoader

# Initialize preprocessor
preprocessor = VideoPreprocessor(
    raw_data_path="data/raw",
    processed_data_path="data/processed",
    frame_height=224,
    frame_width=224,
    num_frames=16
)

# Preprocess dataset
preprocessor.process_dataset(split="train")

# Load data
loader = DatasetLoader(data_path="data/processed", batch_size=32)
```

## Contributing

This is a work in progress. Contributions are welcome!