import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def load_results(results_file):
    """Load evaluation results from JSON file."""
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_confusion_matrix(labels, predictions, class_names, output_path, normalize=True):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title_suffix = " (Normalized)"
    else:
        fmt = 'd'
        title_suffix = ""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap="Blues", 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion' if normalize else 'Count'})
    plt.title(f"Confusion Matrix{title_suffix}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot confusion matrix from evaluation results")
    parser.add_argument("--results_file", type=str, required=True, 
                       help="Path to JSON results file from evaluate.py")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output path for confusion matrix image (default: same as results_file with .png)")
    parser.add_argument("--normalize", action="store_true", default=True,
                       help="Normalize confusion matrix (default: True)")
    parser.add_argument("--no_normalize", dest="normalize", action="store_false",
                       help="Don't normalize confusion matrix")
    
    args = parser.parse_args()
    
    # Load results
    results = load_results(args.results_file)
    labels = np.array(results['labels'])
    predictions = np.array(results['predictions'])
    class_names = results['class_names']
    
    # Determine output path
    if args.output_path is None:
        base_name = os.path.splitext(args.results_file)[0]
        suffix = "_normalized" if args.normalize else "_counts"
        args.output_path = f"{base_name}_confusion_matrix{suffix}.png"
    
    # Plot
    plot_confusion_matrix(labels, predictions, class_names, args.output_path, args.normalize)
    
    # Also plot non-normalized if normalized was requested
    if args.normalize:
        base_name = os.path.splitext(args.output_path)[0]
        count_path = f"{base_name.replace('_normalized', '')}_counts.png"
        plot_confusion_matrix(labels, predictions, class_names, count_path, normalize=False)


if __name__ == "__main__":
    main()

