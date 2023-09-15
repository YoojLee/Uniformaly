import os
from typing import Union, List

try:
    from datasets.base import BaseDataset
    from datasets.transforms import *
except:
    from base import BaseDataset
    from transforms import *

import torch
import torchvision
from PIL import Image

import numpy as np

class CIFAR10Dataset(BaseDataset):
    class_to_idx = {
        'airplane': 0, 'automobile': 1,
        'bird': 2, 'cat': 3,
        'deer': 4, 'dog': 5,
        'frog': 6, 'horse': 7,
        'ship': 8, 'truck': 9
    }
    def __init__(self, root:str, classname:Union[str, List[int], List[str]], resize=256, imagesize=224, split="train", train_val_split=1.0, **kwargs):
        super().__init__(root, classname, resize, imagesize, split, **kwargs)
        
        self.root = root
        self.split = split
        
        if type(classname) != list:
            classname=[classname]
        
        if not classname[0].isdigit():
            self.normal_classes=[self.class_to_idx[c] for c in classname]
        else:
            self.normal_classes = list(map(int, classname))
        
        assert train_val_split == 1.0 # deactivate validation mode
        
        self.transform_img = PadTransform(resize, imagesize)
        self.target_transform = TargetTransform(self.normal_classes)
        
        if self.split == "train":
            self.dataset = CustomCIFAR10(
                self.root, train=True, download=True, target_transform=self.target_transform, transform=self.transform_img
            )
                
            # subset for normal
            normal_indices = np.argwhere(
                np.isin(np.asarray(self.dataset.targets), self.normal_classes)
            ).flatten().tolist()
            
            self.dataset = torch.utils.data.Subset(self.dataset, normal_indices)
        
        else:
            self.dataset = CustomCIFAR10(
                root = self.root, train=False, download=True, transform=self.transform_img, target_transform=self.target_transform
            )
            
            self.dataset = torch.utils.data.Subset(self.dataset, list(range(len(self.dataset)))) # Subset 객체로 맞춰주기 위한 과정
            
        self.imagesize = (3, imagesize, imagesize)
            
    def __len__(self):
        return len(self.dataset)
            
class CustomCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR10, self).__init__(*args, **kwargs)
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        
    def __getitem__(self, idx):
        image, is_anomaly = self.data[idx], self.targets[idx]
        
        # apply transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))
                    
        else:
            image = torch.from_numpy(image).float().div(255).permute(2,0,1)
        
        if self.target_transform:
            is_anomaly = self.target_transform(is_anomaly)
        
        mask = torch.zeros([1, *image.size()[1:]])
        
        return {
            "image": image,
            "mask": mask,
            "classname": self.idx_to_class[self.targets[idx]],
            "anomaly": self.targets[idx],
            "is_anomaly": is_anomaly
        }
        
        
class CIFAR100Dataset(BaseDataset):
    class_to_idx = {'apple': 0, 'aquarium_fish': 1, 'baby': 2, 'bear': 3, 'beaver': 4, 'bed': 5, 'bee': 6, 'beetle': 7, 'bicycle': 8, 'bottle': 9, 
                    'bowl': 10, 'boy': 11, 'bridge': 12, 'bus': 13, 'butterfly': 14, 'camel': 15, 'can': 16, 'castle': 17, 'caterpillar': 18, 'cattle': 19, 
                    'chair': 20, 'chimpanzee': 21, 'clock': 22, 'cloud': 23, 'cockroach': 24, 'couch': 25, 'crab': 26, 'crocodile': 27, 'cup': 28, 
                    'dinosaur': 29, 'dolphin': 30, 'elephant': 31, 'flatfish': 32, 'forest': 33, 'fox': 34, 'girl': 35, 'hamster': 36, 'house': 37, 
                    'kangaroo': 38, 'keyboard': 39, 'lamp': 40, 'lawn_mower': 41, 'leopard': 42, 'lion': 43, 'lizard': 44, 'lobster': 45, 'man': 46, 
                    'maple_tree': 47, 'motorcycle': 48, 'mountain': 49, 'mouse': 50, 'mushroom': 51, 'oak_tree': 52, 'orange': 53, 'orchid': 54, 'otter': 55, 
                    'palm_tree': 56, 'pear': 57, 'pickup_truck': 58, 'pine_tree': 59, 'plain': 60, 'plate': 61, 'poppy': 62, 'porcupine': 63, 'possum': 64, 
                    'rabbit': 65, 'raccoon': 66, 'ray': 67, 'road': 68, 'rocket': 69, 'rose': 70, 'sea': 71, 'seal': 72, 'shark': 73, 'shrew': 74, 
                    'skunk': 75, 'skyscraper': 76, 'snail': 77, 'snake': 78, 'spider': 79, 'squirrel': 80, 'streetcar': 81, 'sunflower': 82, 
                    'sweet_pepper': 83, 'table': 84, 'tank': 85, 'telephone': 86, 'television': 87, 'tiger': 88, 'tractor': 89, 'train': 90, 'trout': 91, 
                    'tulip': 92, 'turtle': 93, 'wardrobe': 94, 'whale': 95, 'willow_tree': 96, 'wolf': 97, 'woman': 98, 'worm': 99}
    superclass_dict = {4: 0, 31: 0, 55: 0, 72: 0, 95: 0, 1: 1, 33: 1, 67: 1, 73: 1, 91: 1, 
                           54: 2, 62: 2, 70: 2, 82: 2, 92: 2, 9: 3, 10: 3, 16: 3, 29: 3, 61: 3, 
                           0: 4, 51: 4, 53: 4, 57: 4, 83: 4, 22: 5, 25: 5, 40: 5, 86: 5, 87: 5, 
                           5: 6, 20: 6, 26: 6, 84: 6, 94: 6, 6: 7, 7: 7, 14: 7, 18: 7, 24: 7, 
                           3: 8, 42: 8, 43: 8, 88: 8, 97: 8, 12: 9, 17: 9, 38: 9, 68: 9, 76: 9, 
                           23: 10, 34: 10, 49: 10, 60: 10, 71: 10, 15: 11, 19: 11, 21: 11, 32: 11, 39: 11, 
                           35: 12, 63: 12, 64: 12, 66: 12, 75: 12, 27: 13, 45: 13, 77: 13, 79: 13, 99: 13, 
                           2: 14, 11: 14, 36: 14, 46: 14, 98: 14, 28: 15, 30: 15, 44: 15, 78: 15, 93: 15, 
                           37: 16, 50: 16, 65: 16, 74: 16, 80: 16, 47: 17, 52: 17, 56: 17, 59: 17, 96: 17, 
                           8: 18, 13: 18, 48: 18, 58: 18, 90: 18, 41: 19, 69: 19, 81: 19, 85: 19, 89: 19}    
    def __init__(self, root:str, classname:Union[str, List[int], List[str]], resize=256, imagesize=224, split="train", train_val_split=1.0, **kwargs):
        super().__init__(root, classname, resize, imagesize, split, **kwargs)
        
        self.root = root
        self.split = split
        
        if type(classname) != list:
            classname=[classname]
        
        if type(classname[0]) == int or classname[0].isdigit():
            self.normal_classes = list(map(int, classname))
        else:
            self.normal_classes=[self.superclass_dict[self.class_to_idx[c]] for c in classname]
        
        assert train_val_split == 1.0 # deactivate validation mode
        
        self.transform_img = PadTransform(resize, imagesize)
        self.target_transform = CIFAR100TargetTransform(self.normal_classes)
        
        if self.split == "train":
            self.dataset = CustomCIFAR100(
                self.root, train=True, download=True, target_transform=self.target_transform, transform=self.transform_img
            )
            
            # subset for normal
            normal_indices = np.argwhere(
                np.isin(np.asarray(list(map(lambda x: self.superclass_dict[x], self.dataset.targets))), self.normal_classes)
            ).flatten().tolist()
            
            self.dataset = torch.utils.data.Subset(self.dataset, normal_indices)
        
        else:
            self.dataset = CustomCIFAR100(
                root = self.root, train=False, download=True, transform=self.transform_img, target_transform=self.target_transform
            )
            
            self.dataset = torch.utils.data.Subset(self.dataset, list(range(len(self.dataset)))) # Subset 객체로 맞춰주기 위한 과정
            
        self.imagesize = (3, imagesize, imagesize)
            
    def __len__(self):
        return len(self.dataset)
            
class CustomCIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, *args, **kwargs):
        super(CustomCIFAR100, self).__init__(*args, **kwargs)
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        
    def __getitem__(self, idx):
        image, is_anomaly = self.data[idx], self.targets[idx]
        
        # apply transforms
        if self.transform:
            image = self.transform(Image.fromarray(image))
                    
        else:
            image = torch.from_numpy(image).float().div(255).permute(2,0,1)
        
        if self.target_transform:
            is_anomaly = self.target_transform(is_anomaly)
        
        mask = torch.zeros([1, *image.size()[1:]])
        
        return {
            "image": image,
            "mask": mask,
            "classname": self.idx_to_class[self.targets[idx]],
            "anomaly": self.targets[idx],
            "is_anomaly": is_anomaly
        }

        
if __name__ == "__main__":
    dataset = CIFAR100Dataset("/home/data/anomaly_detection/semantic/cifar100", classname=1, split="test")
    print(len(dataset))
    
    testloader = torch.utils.data.DataLoader(
        dataset.dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(next(iter(testloader))['is_anomaly'].sum())
    