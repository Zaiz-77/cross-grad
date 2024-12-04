import os

from PIL import Image
from torch.utils.data import Dataset


class Office31(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = str(os.path.join(root_dir, domain))
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(self.root_dir)
                               if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label


class OfficeHome(Dataset):
    def __init__(self, root_dir, domain, transform=None):
        self.root_dir = str(os.path.join(root_dir, domain))
        self.transform = transform

        self.classes = sorted([d for d in os.listdir(self.root_dir)
                               if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, label
