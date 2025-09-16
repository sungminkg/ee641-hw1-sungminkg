import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """Basic Conv-BN-ReLU block."""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """Head that predicts bbox, objectness, and class scores."""
    def __init__(self, in_channels, num_anchors, num_classes):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        out_channels = num_anchors * (5 + num_classes)  
        self.pred = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return self.pred(x)


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
        # Block 1 (Stem)
        self.block1_conv1 = ConvBlock(3, 32, stride=1)
        self.block1_conv2 = ConvBlock(32, 64, stride=2)  # 224 -> 112

        # Block 2
        self.block2 = ConvBlock(64, 128, stride=2)       # 112 -> 56 (Scale 1)
        # Block 3
        self.block3 = ConvBlock(128, 256, stride=2)      # 56 -> 28 (Scale 2)
        # Block 4
        self.block4 = ConvBlock(256, 512, stride=2)      # 28 -> 14 (Scale 3)

        # Detection heads
        self.head1 = DetectionHead(128, num_anchors, num_classes)  # Scale 1
        self.head2 = DetectionHead(256, num_anchors, num_classes)  # Scale 2
        self.head3 = DetectionHead(512, num_anchors, num_classes)  # Scale 3
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch, 3, 224, 224]
            
        Returns:
            List of 3 tensors (one per scale), each containing predictions
            Shape: [batch, num_anchors * (5 + num_classes), H, W]
        """
        # Backbone forward
        x = self.block1_conv1(x)
        x = self.block1_conv2(x)

        scale1 = self.block2(x)   # [B, 128, 56, 56]
        scale2 = self.block3(scale1)  # [B, 256, 28, 28]
        scale3 = self.block4(scale2)  # [B, 512, 14, 14]

        # Detection heads
        pred1 = self.head1(scale1)  # [B, A*(5+C), 56, 56]
        pred2 = self.head2(scale2)  # [B, A*(5+C), 28, 28]
        pred3 = self.head3(scale3)  # [B, A*(5+C), 14, 14]

        return [pred1, pred2, pred3]
