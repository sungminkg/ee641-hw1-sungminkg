import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import os


class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap',
                 heatmap_size=64, sigma=2.0, transform=None):
        """
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
            transform: Optional image transform
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.transform = transform

        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.annotations = data["images"]
        self.id_to_filename = {ann["id"]: ann["file_name"] for ann in self.annotations}

    def __len__(self):
        return len(self.annotations)

    def generate_heatmap(self, keypoints, height, width):
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, height, width), dtype=np.float32)

        tmp_size = self.sigma * 3
        for i, (x, y) in enumerate(keypoints):
            if x < 0 or y < 0:
                continue
            mu_x = int(x * (width / 128.0))
            mu_y = int(y * (height / 128.0))

            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]

            if ul[0] >= width or ul[1] >= height or br[0] < 0 or br[1] < 0:
                continue

            size = int(2 * tmp_size + 1)
            x_coords = np.arange(0, size, 1, np.float32)
            y_coords = x_coords[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x_coords - x0) ** 2 + (y_coords - y0) ** 2) / (2 * self.sigma ** 2))

            g_x = max(0, -ul[0]), min(br[0], width) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], height) - ul[1]
            img_x = max(0, ul[0]), min(br[0], width)
            img_y = max(0, ul[1]), min(br[1], height)

            heatmaps[i][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return torch.from_numpy(heatmaps)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        file_name = ann["file_name"]
        img_path = os.path.join(self.image_dir, file_name)

        img = Image.open(img_path).convert("L")
        img = img.resize((128, 128))
        img = np.array(img, dtype=np.float32) / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        keypoints = np.array(ann["keypoints"], dtype=np.float32) 

        if self.output_type == "heatmap":
            targets = self.generate_heatmap(keypoints, self.heatmap_size, self.heatmap_size)
        elif self.output_type == "regression":
            targets = keypoints / 128.0
            targets = torch.from_numpy(targets.flatten()).float()
        else:
            raise ValueError(f"Unknown output_type {self.output_type}")

        return img, targets
