import pandas as pd
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VietnameseFoodDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.image_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        # Debug print
        print(f"Image directory:  {img_dir}")
        print(f"Length of datset: {len(self.image_labels)}")
    def __len__(self):
        return len(self.image_labels)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.image_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

        
