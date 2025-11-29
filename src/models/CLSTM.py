import torch
import torch.nn as nn
from torchvision import models


class CLSTM(nn.Module):
    """
    CNN + LSTM Hybrid Model for Video Classification
    
    Architecture:
    1. 2D ResNet-18 backbone extracts spatial features from each frame independently
    2. LSTM processes the sequence of frame features to capture temporal dynamics
    3. Final classification head predicts gesture class
    
    Args:
        sample_size (int): Spatial resolution of input frames (e.g., 128 for 128x128)
        sample_duration (int): Number of frames per video clip (temporal dimension)
        num_classes (int): Number of output classes for classification
        hidden_dim (int): Hidden state dimension of LSTM
        num_layers (int): Number of stacked LSTM layers
    """
    
    def __init__(self, sample_size, sample_duration, num_classes=9, hidden_dim=256, num_layers=2):
        super(CLSTM, self).__init__()
        
        self.sample_size = sample_size
        self.sample_duration = sample_duration
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 1. Spatial Feature Extractor (2D CNN Backbone)
        # Use ResNet-18 without pre-trained weights (training from scratch)
        resnet = models.resnet18(weights=None)
        self.feature_dim = 512  # ResNet-18 output feature dimension
        
        # Remove the final fully connected layer from ResNet
        # We only want the feature extraction part
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # 2. Temporal Sequence Modeling (LSTM)
        # Processes sequence of frame features to capture temporal patterns
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.5 if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # 3. Classification Head
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize LSTM and FC layer weights."""
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                # Set forget gate bias to 1 (common LSTM trick)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
        
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through CRNN.
        
        Args:
            x: Input tensor of shape [Batch, Channels, Frames, Height, Width]
               e.g., [8, 3, 32, 128, 128]
        
        Returns:
            output: Classification logits of shape [Batch, num_classes]
        """
        # Input shape: [Batch, Channels, Frames, Height, Width]
        b, c, t, h, w = x.shape
        
        # Reshape to process all frames through 2D CNN in parallel
        # Permute to [Batch, Frames, Channels, Height, Width]
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        # Flatten batch and time: [Batch * Frames, Channels, Height, Width]
        x = x.view(b * t, c, h, w)
        
        # Extract spatial features using ResNet backbone
        # Output shape: [Batch * Frames, Feature_Dim, 1, 1]
        x = self.backbone(x)
        # Flatten spatial dimensions: [Batch * Frames, Feature_Dim]
        x = x.flatten(1)
        
        # Reshape back to sequence: [Batch, Frames, Feature_Dim]
        x = x.view(b, t, self.feature_dim)
        
        # Process temporal sequence with LSTM
        # lstm_out: [Batch, Frames, Hidden_Dim]
        # h_n: [Num_Layers, Batch, Hidden_Dim] (final hidden state)
        # c_n: [Num_Layers, Batch, Hidden_Dim] (final cell state)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use the output of the last time step for classification
        # Shape: [Batch, Hidden_Dim]
        last_frame_feat = lstm_out[:, -1, :]
        
        # Final classification
        # Output shape: [Batch, num_classes]
        output = self.fc(last_frame_feat)
        
        return output


if __name__ == '__main__':
    # Test the model
    model = CLSTM(sample_size=128, sample_duration=32, num_classes=9)
    model = model.cuda() if torch.cuda.is_available() else model
    
    print("="*60)
    print("CRNN Model Architecture")
    print("="*60)
    print(model)
    print("="*60)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("="*60)
    
    # Test forward pass
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 32, 128, 128)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()
    
    output = model(input_tensor)
    print(f"\nInput shape:  {list(input_tensor.shape)}")
    print(f"Output shape: {list(output.shape)}")
    print(f"Expected:     [{batch_size}, 9]")
    print("="*60)