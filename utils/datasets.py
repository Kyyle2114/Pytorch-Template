import os
import cv2
from typing import Type, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets

class CustomDataset(Dataset):
    def __init__(self, dataset_path, transform=None):
        self.image_dir = dataset_path
        self.transform = transform
        
        self.data_images = os.listdir(self.image_dir)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.data_images[idx])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
            
        return image
    
    def __len__(self):
        return len(self.data_images)
    

# --- CIFAR10 for testing ---

class CIFAR10Dataset(Dataset):
    def __init__(self, root: str, train: bool = True, transform: Optional[torch.nn.Module] = None):
        """
        CIFAR10 Dataset wrapper.
        
        Args:
            root (str): Root directory of dataset where CIFAR10 will be downloaded.
            train (bool, optional): If True, creates dataset from training set, otherwise from test set. Defaults to True.
            transform (Optional[torch.nn.Module], optional): Optional transform to be applied on a sample. Defaults to None.
        """
        self.dataset = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform
        )
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            Tuple[torch.Tensor, int]: (image, target) where target is the index of the target class
        """
        return self.dataset[idx]
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    @property
    def classes(self) -> list:
        """Get the list of class names."""
        return self.dataset.classes


def make_cifar10_dataset(
    dataset_path: str,
    train: bool = True,
    transform: Optional[torch.nn.Module] = None
) -> Type[torch.utils.data.Dataset]:
    """
    Make PyTorch Dataset for CIFAR10.
    
    Args:
        dataset_path (str): Root directory where CIFAR10 will be downloaded.
        train (bool, optional): If True, creates dataset from training set, otherwise from test set. Defaults to True.
        transform (Optional[torch.nn.Module], optional): Optional transform to be applied on a sample. Defaults to None.
        
    Returns:
        torch.Dataset: PyTorch Dataset for CIFAR10
    """
    dataset = CIFAR10Dataset(
        root=dataset_path,
        train=train,
        transform=transform
    )
    
    return dataset