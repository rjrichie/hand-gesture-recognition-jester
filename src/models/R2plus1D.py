import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2plus1D(nn.Module):
    """
    R(2+1)D-18 wrapper for video classification.
    
    R(2+1)D factorizes 3D convolutions into separate 2D spatial and 1D temporal convolutions.
    This increases non-linearity and optimization efficiency compared to standard C3D.
    
    Args:
        sample_size (int): Spatial resolution of input frames (e.g., 128 for 128x128)
        sample_duration (int): Number of frames per video clip (temporal dimension)
        num_classes (int): Number of output classes for classification
        pretrained (bool): If True, load weights pretrained on Kinetics-400. 
                           If False, initialize randomly (for training from scratch).
    """
    
    def __init__(self, sample_size, sample_duration, num_classes=9, pretrained = False):
        super(R2plus1D, self).__init__()
        
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes

        if pretrained:
            print(f"Loading R(2+1)D-18 with Kinetics-400 pretrained weights...")
            weights = R2Plus1D_18_Weights.DEFAULT
        else:
            print(f"Initializing R(2+1)D-18 from scratch (Random Init)...")
            weights = None
            
        self.model = r2plus1d_18(weights=weights)
        
        # Modify the classification head
        # The default model has 400 classes (Kinetics). We replace it with our num_classes.
        # R(2+1)D in torchvision uses 'fc' as the final layer name.
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        """
        Forward pass through R(2+1)D.
        
        Args:
            x: Input tensor of shape [Batch, Channels, Frames, Height, Width]
               e.g., [8, 3, 32, 128, 128]
        
        Returns:
            output: Classification logits of shape [Batch, num_classes]
        """
        # Our input: [Batch, Channels, Frames, Height, Width]
        # Torchvision R(2+1)D expects: [Batch, Channels, Frames, Height, Width]
        
        output = self.model(x)
        
        return output


if __name__ == '__main__':
    print("\nTesting R(2+1)D-18 Wrapper")
    print("="*60)
    
    # Initialize from scratch
    model_scratch = R2plus1D(
        sample_size=128, 
        sample_duration=32, 
        num_classes=9, 
    )
    
    # Move to GPU if available for accurate memory estimation (optional)
    if torch.cuda.is_available():
        model_scratch = model_scratch.cuda()
    
    total_params = sum(p.numel() for p in model_scratch.parameters())
    trainable_params = sum(p.numel() for p in model_scratch.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)

    # Test forward pass
    batch_size = 2
    # [Batch, Channels, Frames, Height, Width]
    input_tensor = torch.randn(batch_size, 3, 32, 128, 128)
    
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    output = model_scratch(input_tensor)
    
    print(f"\nInput shape:  {list(input_tensor.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Expected:     [{batch_size}, 9]")
    print("="*60)