import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np

class ShapeDetectionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Initialize the dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to COCO-style JSON annotations
            transform: Optional transform to apply to images
        """
        self.image_dir = image_dir
        self.transform = transform

        # Load and parse annotations
        with open(annotation_file, "r") as f:
            data = json.load(f)

        # Store image paths and corresponding annotations
        self.images = {img["id"]: img["file_name"] for img in data["images"]}

        self.annotations = {img_id: [] for img_id in self.images.keys()}
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            x, y, w, h = ann["bbox"]  # COCO format: [x, y, w, h]
            box = [x, y, x + w, y + h]  # convert to [x1, y1, x2, y2]
            label = ann["category_id"]
            self.annotations[img_id].append({"box": box, "label": label})

        self.image_ids = list(self.images.keys())

    def __len__(self):
        """Return the total number of samples."""
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        img_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, self.images[img_id])

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transform if provided
        if self.transform:
            image = self.transform(image)
        else:
            # Default: convert to tensor [3, H, W], normalized to [0,1]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        # Load targets
        annos = self.annotations[img_id]
        boxes = torch.tensor([a["box"] for a in annos], dtype=torch.float32)
        labels = torch.tensor([a["label"] for a in annos], dtype=torch.long)

        targets = {"boxes": boxes, "labels": labels}

        return image, targets
