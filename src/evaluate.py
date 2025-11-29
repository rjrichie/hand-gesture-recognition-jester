import argparse
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_processing.jester_dataset import JesterDataset
from models.CNN3D import C3D
from models.CLSTM import CLSTM
from models.R2plus1D import R2plus1D


CLASS_MAPPING = {
    1: "Doing_other_things",
    3: "No_gesture",
    11: "Sliding_Two_Fingers_Down",
    12: "Sliding_Two_Fingers_Left",
    13: "Sliding_Two_Fingers_Right",
    14: "Sliding_Two_Fingers_Up",
    15: "Stop_Sign",
    24: "Zooming_In_With_Full_Hand",
    27: "Zooming_Out_With_Two_Fingers"
}


def get_model(model_type, sample_size, sample_duration, num_classes):
    """Create model based on model type."""
    if model_type == "c3d":
        return C3D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    elif model_type == "clstm":
        return CLSTM(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    elif model_type == "r2plus1d":
        return R2plus1D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes)
    elif model_type == "r2plus1d_pretrained":
        return R2plus1D(sample_size=sample_size, sample_duration=sample_duration, num_classes=num_classes, pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model):
    """Count total and trainable parameters in the model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def measure_inference_time(model, sample_input, device, num_warmup=10, num_iterations=100):
    """Measure average inference time for a single sample."""
    model.eval()
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(sample_input)
    
    # Synchronize GPU before timing
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            _ = model(sample_input)
            
            # Synchronize GPU to ensure operation is complete
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            times.append(end_time - start_time)
    
    times = np.array(times)
    return {
        'mean_ms': float(np.mean(times) * 1000),
        'std_ms': float(np.std(times) * 1000),
        'min_ms': float(np.min(times) * 1000),
        'max_ms': float(np.max(times) * 1000),
        'median_ms': float(np.median(times) * 1000),
        'num_iterations': num_iterations
    }


def load_model(checkpoint_path, model_type, sample_size, sample_duration, num_classes, device):
    model = get_model(model_type, sample_size, sample_duration, num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    
    model.load_state_dict(new_state_dict)
    model = model.to(device).eval()
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    all_preds, all_labels = [], []
    running_loss = 0.0
    total = 0
    
    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Evaluating"):
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * videos.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Compute overall metrics
    metrics = {
        'loss': float(running_loss / total),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro')),
        'f1_weighted': float(f1_score(all_labels, all_preds, average='weighted'))
    }
    
    # Compute per-class metrics
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    metrics['per_class'] = {}
    for i in range(len(np.unique(all_labels))):
        if str(i) in report:
            metrics['per_class'][i] = {
                'precision': float(report[str(i)]['precision']),
                'recall': float(report[str(i)]['recall']),
                'f1': float(report[str(i)]['f1-score']),
                'support': int(report[str(i)]['support'])
            }
    
    return metrics, all_preds, all_labels, cm


def get_class_names(dataset):
    original_labels = sorted(dataset.label2id.keys())
    return [CLASS_MAPPING.get(label, f"Class_{label}") for label in original_labels]


def main():
    parser = argparse.ArgumentParser(description="Evaluate gesture recognition model")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--csv_file", type=str, required=True, 
                       help="CSV file name (val or test)")
    parser.add_argument("--model_type", type=str, default="c3d", choices=["c3d", "clstm", "r2plus1d", "r2plus1d_pretrained"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_inference_runs", type=int, default=100,
                       help="Number of iterations for inference timing")
    
    args = parser.parse_args()
    
    # Fixed parameters
    root_dir = "dataset/modified/data"
    num_classes = 9
    sample_size = 128
    num_frames = 32
    
    # Setup paths
    if not args.csv_file.endswith('.csv'):
        args.csv_file += '.csv'
    csv_path = os.path.join("dataset/modified/annotations", args.csv_file)
    dataset_name = args.csv_file.replace(".csv", "")
    
    # Auto-calculate output directory
    checkpoint_name = os.path.basename(args.checkpoint_path).replace(".pth", "")
    output_dir = os.path.join("results", checkpoint_name)
    os.makedirs(output_dir, exist_ok=True)
    
    inference_device = torch.device("cpu")
    testing_device = torch.device("mps" if torch.mps.is_available() else "cpu")
    print(f"Using device (inference test): {inference_device}")
    print(f"Using device (testing): {testing_device}")
    
    # Load dataset and model
    dataset = JesterDataset(csv_file=csv_path, root_dir=root_dir, num_frames=num_frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    inference_model = load_model(args.checkpoint_path, args.model_type, sample_size, num_frames, num_classes, inference_device)
    
    # Count parameters
    print("\n" + "="*60)
    print("Model Information:")
    print("="*60)
    param_counts = count_parameters(inference_model)
    print(f"Total parameters:       {param_counts['total']:,}")
    print(f"Trainable parameters:   {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"Model size (MB):        {param_counts['total'] * 4 / (1024**2):.2f}")  # Assuming float32
    
    # Measure inference time with batch size 1 on cpu
    print("\n" + "="*60)
    print("Measuring Inference Time (batch_size=1):")
    print("="*60)
    sample_input = torch.randn(1, 3, num_frames, sample_size, sample_size).to(inference_device)
    inference_stats = measure_inference_time(
        inference_model, 
        sample_input, 
        inference_device, 
        num_warmup=10, 
        num_iterations=args.num_inference_runs
    )
    print(f"Mean inference time:    {inference_stats['mean_ms']:.2f} ± {inference_stats['std_ms']:.2f} ms")
    print(f"Median inference time:  {inference_stats['median_ms']:.2f} ms")
    print(f"Min inference time:     {inference_stats['min_ms']:.2f} ms")
    print(f"Max inference time:     {inference_stats['max_ms']:.2f} ms")
    print(f"Iterations:             {inference_stats['num_iterations']}")
    print(f"FPS (mean):             {1000 / inference_stats['mean_ms']:.2f}")

    testing_model = load_model(args.checkpoint_path, args.model_type, sample_size, num_frames, num_classes, testing_device)

    # Measure inference time with batch size 1 on mps
    print("\n" + "="*60)
    print("Measuring Inference Time (batch_size=1):")
    print("="*60)
    sample_input_mps = torch.randn(1, 3, num_frames, sample_size, sample_size).to(testing_device)
    inference_stats_mps = measure_inference_time(
        testing_model, 
        sample_input_mps, 
        testing_device, 
        num_warmup=10, 
        num_iterations=args.num_inference_runs
    )
    print(f"Mean inference time:    {inference_stats_mps['mean_ms']:.2f} ± {inference_stats_mps['std_ms']:.2f} ms")
    print(f"Median inference time:  {inference_stats_mps['median_ms']:.2f} ms")
    print(f"Min inference time:     {inference_stats_mps['min_ms']:.2f} ms")
    print(f"Max inference time:     {inference_stats_mps['max_ms']:.2f} ms")
    print(f"Iterations:             {inference_stats_mps['num_iterations']}")
    print(f"FPS (mean):             {1000 / inference_stats_mps['mean_ms']:.2f}")
    
    # Evaluate
    print("\n" + "="*60)
    print(f"Evaluating on {dataset_name}:")
    print("="*60)
    criterion = nn.CrossEntropyLoss()
    metrics, predictions, labels, conf_matrix = evaluate(testing_model, loader, criterion, testing_device)
    class_names = get_class_names(dataset)
    
    # Print results
    print(f"\n{dataset_name.upper()} Results:")
    print(f"  Loss:          {metrics['loss']:.4f}")
    print(f"  Accuracy:      {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  F1 (Macro):    {metrics['f1_macro']:.4f}")
    print(f"  F1 (Weighted): {metrics['f1_weighted']:.4f}")
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        if i in metrics['per_class']:
            pc = metrics['per_class'][i]
            print(f"  {class_name:30s} | Precision: {pc['precision']:.3f} | Recall: {pc['recall']:.3f} | F1: {pc['f1']:.3f} | Support: {pc['support']}")
    print()
    
    # Save results
    results = {
        'predictions': predictions.tolist(),
        'labels': labels.tolist(),
        'metrics': metrics,
        'confusion_matrix': conf_matrix.tolist(),
        'class_names': class_names,
        'model_info': {
            'parameters': param_counts,
            'inference_cpu': inference_stats,
            'inference_mps': inference_stats_mps,
        }
    }
    
    output_file = os.path.join(output_dir, f"{dataset_name}_results.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()