import os
from typing import Union, List

try:
    from base import BaseDataset
    from transforms import ImageTransform, TargetTransform
except ModuleNotFoundError:
    from datasets.base import BaseDataset
    from datasets.transforms import ImageTransform, TargetTransform

import numpy as np
import torch
import torchvision
from PIL import Image


class SpeciesDataset(BaseDataset):
    class_to_idx = {'abudefduf_vaigiensis': 0, 'acanthurus_coeruleus': 1, 'acarospora_socialis': 2, 'acris_gryllus': 3, 
                    'adolphia_californica': 4, 'agaricus_augustus': 5, 'amanita_parcivolvata': 6, 'ameiurus_natalis': 7, 
                    'anartia_jatrophae': 8, 'aplysia_californica': 9, 'arcyria_denudata': 10, 'arion_ater': 11, 
                    'aulostomus_chinensis': 12, 'bjerkandera_adusta': 13, 'bombus_pensylvanicus': 14, 'callianax_biplicata': 15, 
                    'calochortus_apiculatus': 16, 'campsis_radicans': 17, 'cepaea_nemoralis': 18, 'cladonia_chlorophaea': 19, 
                    'cochlodina_laminata': 20, 'crepidotus_applanatus': 21, 'cypripedium_macranthos': 22, 
                    'dendrobates_auratus': 23, 'diaulula_sandiegensis': 24, 'drosera_rotundifolia': 25, 
                    'echinocereus_pectinatus_pectinatus': 26, 'eremnophila_aureonotata': 27, 'etheostoma_caeruleum': 28, 
                    'fistularia_commersonii': 29, 'ganoderma_tsugae': 30, 'halysidota_tessellaris': 31, 
                    'herichthys_cyanoguttatus': 32, 'hippocampus_whitei': 33, 'hygrocybe_miniata': 34, 
                    'hypomyces_chrysospermus': 35, 'juniperus_turbinata': 36, 'kuehneromyces_mutabilis': 37, 
                    'laetiporus_gilbertsonii': 38, 'lepisosteus_platyrhincus': 39, 'leucocoprinus_cepistipes': 40, 
                    'lissachatina_fulica': 41, 'lycium_californicum': 42, 'megapallifera_mutabilis': 43, 
                    'mononeuria_groenlandica': 44, 'neverita_lewisii': 45, 'octopus_tetricus': 46, 'orienthella_trilineata': 47, 
                    'phyllotopsis_nidulans': 48, 'planorbarius_corneus': 49, 'protoparmeliopsis_muralis': 50, 
                    'quercus_dumosa': 51, 'ruditapes_philippinarum': 52, 'salamandra_lanzai': 53, 'swietenia_macrophylla': 54, 
                    'teloschistes_chrysophthalmus': 55, 'tramea_onusta': 56, 'umbra_limi': 57, 'vespula_squamosa': 58, 
                    'zelus_renardii': 59}
    def __init__(self, root:str, classname:Union[str, List[int], List[str]], resize=256, imagesize=224, split="train", train_val_split=1.0, **kwargs):
        super().__init__(root, classname, resize, imagesize, split, **kwargs)
        
        self.root = root
        self.split = split
        datapath = os.path.join(self.root, "one_class_{}".format(self.split))
        
        if type(classname) != list:
            classname=[classname]
        
        if not classname[0].isdigit():
            self.normal_classes=[self.class_to_idx[c] for c in classname]
        else:
            self.normal_classes = list(map(int, classname))
        
        assert train_val_split == 1.0 # deactivate validation mode
        
        self.transform_img = ImageTransform(resize, imagesize)
        self.target_transform = TargetTransform(self.normal_classes)
        
        if self.split == "train":
            self.dataset = SpeciesClass(
                root=datapath, target_transform=self.target_transform, transform=self.transform_img
            )
                
            # subset for normal
            normal_indices = np.argwhere(
                np.isin(np.asarray(self.dataset.targets), self.normal_classes)
            ).flatten().tolist()
            
            self.dataset = torch.utils.data.Subset(self.dataset, normal_indices)
        
        else:
            self.dataset = SpeciesClass(
                root=datapath, transform=self.transform_img, target_transform=self.target_transform
            )
            
            self.dataset = torch.utils.data.Subset(self.dataset, list(range(len(self.dataset)))) # Subset 객체로 맞춰주기 위한 과정
            
        self.imagesize = (3, imagesize, imagesize)
            
    def __len__(self):
        return len(self.dataset)
            
class SpeciesClass(torchvision.datasets.ImageFolder):
    def __init__(self, *args, **kwargs):
        super(SpeciesClass, self).__init__(*args, **kwargs)
        self.idx_to_class = {v:k for k,v in self.class_to_idx.items()}
        
    def __getitem__(self, idx):
        image, is_anomaly = self.imgs[idx]
        
        # apply transforms
        if self.transform:
            image = self.transform(Image.open(image).convert("RGB"))
                    
        else:
            image = torch.from_numpy(image).float().div(255).permute(2,0,1)
        
        if self.target_transform:
            is_anomaly = self.target_transform(is_anomaly)
        
        mask = torch.zeros([1, *image.size()[1:]])
        
        return {
            "image": image,
            "mask": mask,
            "is_anomaly": is_anomaly
        }

        
if __name__ == "__main__":
    dataset = SpeciesDataset("/home/data/anomaly_detection/semantic/species60", classname=[i for i in range(30)], split="train")
    print(len(dataset))
    
    testloader = torch.utils.data.DataLoader(
        dataset.dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    

    print(next(iter(testloader))['is_anomaly'])
