import torch
from torchvision import transforms
from torch.utils.data import Dataset

class RainPlaceDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.root_dir)

    def __getitem__(self, idx):
        image_path = self.root_dir[idx]['image']
        label = self.root_dir[idx]['label']
        image = self.transform(Image.open(image_path).convert("RGB"))
        return {'image': image, 'label': label}
