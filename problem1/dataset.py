import torch
from torch.utils.data import Dataset
from PIL import Image
import json

import torchvision.transforms as transforms

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
        try:
            with open(annotation_file, "r", encoding="utf-8") as file:
                annotation_dict = json.load(file)
            print("Successfully loaded the annotation file:")
            print(type(annotation_dict))
            # print(annotation_dict)


            images = annotation_dict["images"] # list
            annotations = annotation_dict["annotations"] # list
            categories = annotation_dict["categories"]
            print("images size:", len(images))
            print("annotations size:", len(annotations))

            # if len(images) != len(annotations):
            #     print("Error input file format!")
            #     return

        except Exception as e:
            print(f"Cannot open the annotation file: {e}")
            raise


        # Store image paths and corresponding annotations 
        self.samples = []
        

        for i in range(len(images)):
            img = images[i]
            # ano = annotations[i]

            bboxs = []
            category_ids = []
            for j in range(len(annotations)):
                ano = annotations[j]

                if ano["image_id"] == img["id"]:
                    bboxs.append(ano["bbox"]) # List[List]
                    category_ids.append(ano["category_id"]) # List[int]

            

            image_full_path = image_dir + "/" + img["file_name"]

            element = {
                "path": image_full_path,
                "boxes": bboxs,
                "label": category_ids

            }
            self.samples.append(element)



    def __len__(self):
        """Return the total number of samples.""" 
        
        return len(self.samples)


    def __getitem__(self, idx): 
        """
        Return a sample from the dataset.

        Returns:
            image: Tensor of shape [3, H, W]
            targets: Dict containing:
                - boxes: Tensor of shape [N, 4] in [x1, y1, x2, y2] format
                - labels: Tensor of shape [N] with class indices (0, 1, 2)
        """
        
        sample = self.samples[idx] # dict

        img = Image.open(sample["path"])

        transform = transforms.ToTensor() 
        tensor_img = transform(img)


        boxes = torch.tensor(sample["boxes"], dtype=torch.float32)
        # labels = torch.tensor(sample["label"])
        labels = torch.tensor(sample["label"], dtype=torch.long)
        targets = {
            "boxes": boxes,
            "labels": labels

        }
        # print("boxes shape:", boxes.shape)
        # print("labels shape:", labels.shape)

        return tensor_img, targets






