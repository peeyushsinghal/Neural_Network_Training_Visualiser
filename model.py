import torch.nn as nn

class MNISTNet(nn.Module):
    def __init__(self, layer_sizes=[32, 64, 128, 256]):
        """
        Initialize CNN with custom layer sizes
        Args:
            layer_sizes (list): List of 4 integers specifying the number of filters in each conv layer
        """
        super(MNISTNet, self).__init__()
        
        self.layer_sizes = layer_sizes
        self.features = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, layer_sizes[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(layer_sizes[0], layer_sizes[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(layer_sizes[1], layer_sizes[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth conv layer
            nn.Conv2d(layer_sizes[2], layer_sizes[3], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(layer_sizes[3], 10)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_layer_sizes(self):
        """Return the layer sizes for model comparison"""
        return self.layer_sizes