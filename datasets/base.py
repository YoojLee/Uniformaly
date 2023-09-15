import os
import PIL
import torch

try:
    from datasets.transforms import ImageTransform, BaseTransform
except ModuleNotFoundError:
    from transforms import ImageTransform, BaseTransform


class BaseDataset(torch.utils.data.Dataset):
    """
    BaseDataset for Anomaly Detection
    """
    _CLASSNAMES=[]
    
    def __init__(self, root:str, classname:str, resize:int, imagesize:int, split:str, train_val_split:float=1.0, use_multiclass:bool=False, low_shot:int=-1, **kwargs):
        super().__init__()
        self.root = root
        self.split = split
        self.classnames_to_use = [classname] if classname is not None else self._CLASSNAMES # 일단 이렇게 해보고 생각
        self.train_val_split = train_val_split
        self.use_multiclass = use_multiclass
        self.low_shot = low_shot
        
        if "mvtec" in self.__class__.__name__.lower():
            self.imgpaths_per_class, self.data_to_iterate = self.get_image_data()
        
        # transform
        self.transform_img = ImageTransform(resize, imagesize)
        self.transform_mask = BaseTransform(resize, imagesize)

        self.imagesize = (3, imagesize, imagesize)
    
    def __len__(self):
        return len(self.data_to_iterate)
    
    def __getitem__(self, idx):
        classname, anomaly, image_path, mask_path = self.data_to_iterate[idx]
        image = PIL.Image.open(image_path).convert("RGB")
        image = self.transform_img(image)

        if self.split == "test" and mask_path is not None:
            mask = PIL.Image.open(mask_path)
            mask = self.transform_mask(mask)
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": classname,
            "anomaly": anomaly,
            "is_anomaly": int("good" not in anomaly),
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
        }
        
    @classmethod        
    def get_classname(cls):
        return cls._CLASSNAMES
    
    def get_image_data(self):
        raise NotImplementedError