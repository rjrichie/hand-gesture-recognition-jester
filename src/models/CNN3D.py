import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class C3D(nn.Module):
    def __init__(self, sample_size, sample_duration, num_classes=9):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group3 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group4 = nn.Sequential(
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        # Dynamically compute input size for fully connected layers
        self._to_linear = None
        self._initialize_fc_input(sample_size, sample_duration)

        # Fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(self._to_linear, 2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc = nn.Linear(2048, num_classes)

    def _initialize_fc_input(self, sample_size, sample_duration):
        with torch.no_grad():
            x = torch.zeros(1, 3, sample_duration, sample_size, sample_size)
            x = self.group1(x)
            x = self.group2(x)
            x = self.group3(x)
            x = self.group4(x)
            x = self.group5(x)
            self._to_linear = x.view(1, -1).size(1)

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        out = self.group3(out)
        out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


if __name__ == '__main__':
    model = C3D(sample_size=128, sample_duration=32, num_classes=9)
    model = model
    model = nn.DataParallel(model)
    print(model)

    # Example batch: 8 videos of 32 frames, 128x128
    input_var = torch.randn(8, 3, 32, 128, 128)
    output = model(input_var)
    print(output.shape)  # Expected: [8, 9]
