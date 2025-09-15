import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import json
import torchvision.transforms as transforms

class KeypointDataset(Dataset):
    def __init__(self, image_dir, annotation_file, output_type='heatmap', 
                 heatmap_size=64, sigma=2.0):
        """
        Initialize the keypoint dataset.
        
        Args:
            image_dir: Path to directory containing images
            annotation_file: Path to JSON annotations
            output_type: 'heatmap' or 'regression'
            heatmap_size: Size of output heatmaps (for heatmap mode)
            sigma: Gaussian sigma for heatmap generation
        """
        self.image_dir = image_dir
        self.output_type = output_type
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        assert output_type in ("heatmap", "regression")

        # Load annotations
        
        try:
            with open(annotation_file, "r", encoding="utf-8") as file:
                annotation_dict = json.load(file)
            print("Successfully loaded the annotation file:")
            print(type(annotation_dict))
            # print(annotation_dict)


            images = annotation_dict["images"] # list[dict]
            keypoint_names = annotation_dict["keypoint_names"]
            num_keypoints = annotation_dict["num_keypoints"]

        except Exception as e:
            print(f"Cannot open the annotation file: {e}")
            raise


        self.samples = []

        for i in range(len(images)):
            img = images[i]

            image_full_path = image_dir + "/" + img["file_name"]

            element = {
                "path": image_full_path,
                "keypoints": img["keypoints"]
            }
            self.samples.append(element)



    
    def generate_heatmap(self, keypoints, height, width):
        """
        Generate gaussian heatmaps for keypoints.
        
        Args:
            keypoints: Array of shape [num_keypoints, 2] in (x, y) format
            height, width: Dimensions of the heatmap
            
        Returns:
            heatmaps: Tensor of shape [num_keypoints, height, width]
        """
        # For each keypoint:
        # 1. Create 2D gaussian centered at keypoint location
        # 2. Handle boundary cases

        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        

        num_keypoints = keypoints.shape[0]
        heatmaps = torch.zeros((num_keypoints, height, width), dtype=torch.float32)

        # Create meshgrid for the entire heatmap
        y = torch.arange(0, height, dtype=torch.float32).unsqueeze(1)  # shape [H, 1]
        x = torch.arange(0, width, dtype=torch.float32).unsqueeze(0)   # shape [1, W]
        yy = y.repeat(1, width)  # [H, W]
        xx = x.repeat(height, 1) # [H, W]

        for i, (px, py) in enumerate(keypoints):
            if px < 0 or py < 0 or px >= width or py >= height:
                continue  # skip invalid keypoints

            # Calculate gaussian heatmap centered at (px, py)
            d2 = (xx - px)**2 + (yy - py)**2
            exponent = -d2 / (2.0 * self.sigma**2)
            gaussian = torch.exp(exponent)

            # Clamp value between 0~1 for numerical stability
            gaussian = torch.clamp(gaussian, 0.0, 1.0)
            heatmaps[i] = gaussian


        return heatmaps 

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):
        """
        Return a sample from the dataset.
        
        Returns:
            image: Tensor of shape [1, 128, 128] (grayscale)
            If output_type == 'heatmap':
                targets: Tensor of shape [5, 64, 64] (5 heatmaps)
            If output_type == 'regression':
                targets: Tensor of shape [10] (x,y for 5 keypoints, normalized to [0,1])
        """
        

        sample = self.samples[idx] # dict

        img = Image.open(sample["path"])

        transform = transforms.ToTensor() 
        tensor_img = transform(img)


        if self.output_type == "heatmap":
            
            targets = self.generate_heatmap(sample["keypoints"], self.heatmap_size, self.heatmap_size)

            return tensor_img, targets

        else:
            width, height = 128.0, 128.0

            keypoints = torch.as_tensor(sample["keypoints"], dtype=torch.float32)

            keypoints = keypoints.clone().float()  
            keypoints[:, 0] /= width   # normalize x
            keypoints[:, 1] /= height  # normalize y
            
            targets = keypoints.view(-1)

            return tensor_img, targets

