import torch
import torch.nn as nn

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        """
        Initialize the multi-scale detector.
        
        Args:
            num_classes: Number of object classes (not including background)
            num_anchors: Number of anchors per spatial location
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Feature extraction backbone
        # Extract features at 3 different scales
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), # nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2), # nn.MaxPool2d(kernel_size, stride)

            nn.Conv2d(32, 64, 3, 2, 1), 
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(2, 2)
        )
        

        # Detection heads for each scale
        # Each head outputs: [batch, num_anchors * (4 + 1 + num_classes), H, W]
        out_channels = num_anchors * (5 + num_classes)

        self.det_head1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, out_channels, kernel_size=1)
        )
        self.det_head2 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, out_channels, kernel_size=1)
        )
        self.det_head3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.Conv2d(512, out_channels, kernel_size=1)
        )
        

    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
            where 5 = 4 bbox coords + 1 objectness score
        """
        
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)


        pred1 = self.det_head1(x2)
        pred2 = self.det_head2(x3)
        pred3 = self.det_head3(x4)


        return [pred1, pred2, pred3]



