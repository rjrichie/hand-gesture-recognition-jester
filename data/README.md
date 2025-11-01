# Data Directory

This directory contains the hand gesture recognition dataset.

## Structure

- `raw/`: Original, unprocessed video data from the Jester dataset
- `processed/`: Preprocessed frames and features ready for training
- `interim/`: Intermediate data that has been transformed

## Usage

1. Place the Jester dataset videos in the `raw/` directory
2. Run the preprocessing script to populate the `processed/` directory
3. Use the data loaders from `src/data/preprocessing.py` to load data for training

Note: These directories are ignored by git (see .gitignore) as they typically contain large files.
