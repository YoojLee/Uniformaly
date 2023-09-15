import numpy as np
import PIL
from PIL import Image

import torch
from torchvision import transforms


class BaseTransform(object):
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, resize, imagesize):
        self.resize = resize
        self.imagesize = imagesize
        
        self.transform = [
            transforms.Resize(self.resize),
            transforms.CenterCrop(self.imagesize),
            transforms.ToTensor()
        ]

    def __call__(self, img):
        transform = transforms.Compose(self.transform)
        return transform(img)
    
    
class ImageTransform(BaseTransform):
    def __init__(self, resize, imagesize):
        super().__init__(resize, imagesize)
        self.transform.append(transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD))
        
    def __call__(self, img):
        transform = transforms.Compose(self.transform)
        return transform(img)
        

class PadTransform(ImageTransform):
    def __init__(self, resize, imagesize):
        super().__init__(resize, imagesize)

    def __call__(self, img): # img: PIL Image
        if isinstance(img, np.ndarray):
            img = PIL.Image.fromarray(img)        

        padding_w = (224 - self.resize) // 2
        padding_h = (224 - self.resize) // 2      
        
        transform = transforms.Compose(
            [   
                transforms.Resize((self.resize, self.resize), Image.ANTIALIAS),
                transforms.Pad((padding_w, padding_h)),
                transforms.CenterCrop((self.imagesize, self.imagesize)), # to ensure that transformed image has a given shape.
                transforms.ToTensor(),
                transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        return transform(img)

class TargetTransform(object):
    def __init__(self, normal_classes):
        """Transfrom class labels to binary labels (normal or not).

        Args:
            normal_classes (Union[List[int], List[str]]): indicates class labels belonging to normal classes.
        """
        if type(normal_classes) in [str, int]:
            normal_classes = list(normal_classes)
            
        self.normal_classes = normal_classes
        self.transform = transforms.Lambda(
            lambda x: 0 if x in self.normal_classes else 1
        )
        
    def __call__(self, label):
        return self.transform(label)

class CIFAR100TargetTransform(object):
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
    def __init__(self, normal_classes):
        
        if type(normal_classes) in [str, int]: # normal class는 superclass로 받는 형식
            normal_classes = list(normal_classes)
            
        self.normal_classes = normal_classes
        self.transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: self.superclass_dict[x]),
                transforms.Lambda(lambda x: 0 if x in self.normal_classes else 1)        
            ]
        )
        
    def __call__(self, label):
        return self.transform(label)