import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


"""FAISS Nearest Neighbourhood Class"""
class FaissNN(object):
    def __init__(self, on_gpu: bool = False, num_workers: int = 8, device: Union[int, torch.device]=0, prenorm:bool=True) -> None:
        """
        Args:
            on_gpu: If set true, nearest neighbour searches are done on GPU.
            num_workers: Number of workers to use with FAISS for similarity search.
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None
        if isinstance(device, torch.device):
            device = int(torch.cuda.current_device())
        self.device = device
        self.prenorm = prenorm

    def _gpu_cloner_options(self):
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        if self.on_gpu:
            # For the non-gpu faiss python package, there is no GpuClonerOptions
            # so we can not make a default in the function header.
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), self.device, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        if self.on_gpu:
            gpu_config = faiss.GpuIndexFlatConfig()
            gpu_config.device = self.device
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, gpu_config
            )
        return faiss.IndexFlatL2(dimension)

    
    # features: NxD Array
    def fit(self, features: np.ndarray) -> None:
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)

        # normalize
        if self.prenorm:
            faiss.normalize_L2(features)

        self.search_index.add(features.cpu())

    def _train(self, _index, _features):
        pass
    
    
    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        
        if self.prenorm:
            faiss.normalize_L2(query_features)

        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # Build a search index just for this search.
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        
        # normalize
        if self.prenorm:
            faiss.normalize_L2(index_features)

        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


"""FAISS NN Approximation Class"""
class ApproximateFaissNN(FaissNN):
    def __init__(self, on_gpu: bool = False, num_workers: int = 8, device: int=0, prenorm:bool=True):
        super().__init__(on_gpu, num_workers, device, prenorm)

    def _train(self, index, features):
        index.train(features)

    def _gpu_cloner_options(self):
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        cloner.usePrecomputed = False
        return cloner

    def _create_index(self, dimension):
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids
            64,  # sub-quantizers
            8,
        )  # nbits per code
        return self._index_to_gpu(index)

class _BaseMerger:
    def __init__(self):
        """Merges feature embedding by name."""

    def merge(self, features: Union[list, np.ndarray]):
        if type(features) == list:
            features = [self._reduce(feature) for feature in features]
            return torch.cat(features, dim=1)
        else:
            return self._reduce(features)



class ConcatMerger(_BaseMerger):
    @staticmethod
    def _reduce(features):
        # (B,N,D) -> (B*N,D) or (N,D) -> (N,D)
        D = features.shape[-1]
        return features.reshape(-1, D)


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim, patch_size):
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim, patch_size)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1) # [# patches, # layers, d]


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim, patch_size=3):
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim
        self.patch_size = patch_size
        
    def forward(self, features):
        device = torch.device('cuda')
            
        features = features.reshape(len(features), 1, -1)
            
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class SimMeanMapper(torch.nn.Module):
    def __init__(self):
        super(SimMeanMapper, self).__init__()

    def forward(self, features):
        n, _, p, _ = features.shape # features: [N,D,P,P]

        sim = torch.ones(n*p*p).cuda()
        sim = sim.reshape(n,1,p,p) # [N,1,P,P]

        sim_filtered = features * sim # [N,D,P,P]

        # aggregation
        sim_filtered = sim_filtered.mean(dim=[-2,-1]) # [N,D]

        return sim_filtered

    def similarity(self, patch, eps=1e-8):
        """
        Compute the similarity among each position of the patch representation.

        - Args
            patch (torch.Tensor): Patch representations of [D,P,P]
            eps (float): Small value to avoid division by zero. Default: 1e-8
        - Returns
            sim (torch.Tensor): Similarity between center of patch and other positions [P**2,]
        """
        norm = lambda v: torch.sqrt(torch.sum(v**2, dim=1)).unsqueeze(-1)

        # flatten into 2d
        d, p, _ = patch.shape
        patch = patch.reshape(d,-1).permute(1,0)
        p_norm = norm(patch)

        sim = torch.matmul(p_norm, p_norm.T)

        return sim[p//2]        


class LayerwiseAggregator(torch.nn.Module):
    """
    (B,L,N,D) -> (B,N,D)

    Averaging feature maps along layer-axis.
    """
    def __init__(self):
        super(LayerwiseAggregator, self).__init__()

    def forward(self, features):
        features = features.mean(axis=1)
        return features

class ChannelwiseAggregator(torch.nn.Module):
    """
    (B,L,N,D) -> reshape into (B,N,L*D) -> (B,N,D)
    """
    def __init__(self):
        super(ChannelwiseAggregator, self).__init__()
    
    def forward(self, features):
        d = features.shape[-1]
        features = features.permute(0,2,1,3).flatten(2)
        aggregated = F.adaptive_avg_pool1d(features, d)

        return aggregated

class PatchwiseAggregator(torch.nn.Module):
    """
    (B,L,N,D) -> stack features along layer-axis (B,L*N,D) 
    -> averaging L patches so that resulting tensors to be (B,N,D)
    """
    def __init__(self):
        super(PatchwiseAggregator, self).__init__()

    def forward(self, features):
        b,l,n,d = features.shape
        features = features.reshape(b,l*n,d).permute(0,2,1)
        aggregated = F.adaptive_avg_pool1d(features, n).permute(0,2,1)

        return aggregated

class ConcatAggregator(torch.nn.Module):
    def __init__(self):
        super(ConcatAggregator, self).__init__()
    
    def forward(self, features):
        return features.permute(0,2,1,3).flatten(2)


class RescaleSegmentor:
    def __init__(self, device, target_size=224):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):

        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


class NetworkFeatureAggregator(torch.nn.Module):

    def __init__(self, backbone, layers_to_extract_from, device):
        super(NetworkFeatureAggregator, self).__init__()
        """Extraction of network features.

        Runs a network only to the last layer of the list of layers where
        network features should be extracted from.

        Args:
            backbone: torchvision.model
            layers_to_extract_from: [list of str]
        """

        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}
        self.is_vit_backbone = "VisionTransformer" in self.backbone.__class__.__name__

        # register forward hook
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            
            if "swin" in backbone.name:
                network_layer = backbone.__dict__["_modules"]['layers'][int(extract_layer)] # swin-T
            else:
                network_layer = backbone.__dict__["_modules"]['blocks'][int(extract_layer)] # vit


            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass

        return self.outputs

    def feature_dimensions(self, input_shape):
        """Computes the feature dimensions for all layers given input_shape."""
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]

class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        return None


class LastLayerToExtractReachedException(Exception):
    pass


"""Nearest-Neighbourhood Anomaly Scorer Class"""
class NearestNeighbourScorer(object):
    def __init__(self, n_nearest_neighbours: int, local_nn_method=FaissNN(False, 4), temp=0.0) -> None:
        """
        Args:
            n_nearest_neighbours: [int] Number of nearest neighbours used to
                determine anomalous pixels.
            nn_method: Nearest neighbour search method.
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.local_nn_method = local_nn_method
        self.temp = temp
        self.patch_nn = lambda query: self.local_nn_method.run(
            n_nearest_neighbours, query
        )

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """Calls the fit function of the nearest neighbour method.

        Args:
            detection_features: [list of np.arrays]
                [[bs x d_i] for i in n] Contains a list of
                np.arrays for all training images corresponding to respective
                features VECTORS (or maps, but will be resized) produced by
                some backbone network which should be used for image-level
                anomaly detection.
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.local_nn_method.fit(self.detection_features)

    def predict(
        self, query_features: np.ndarray
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """Predicts anomaly score.

        Searches for nearest neighbours of test images in all
        support training images.

        Args:
             detection_query_features: np.array
                 corresponding to the test features generated by
                 some backbone network.
        """
        local_query_features = self.feature_merger.merge(
            query_features,
        )
        
        query_distances, query_nns = self.patch_nn(local_query_features)

        if self.temp:
            query_distances = np.exp((1/self.temp)*query_distances)

        local_anomaly_scores = np.mean(query_distances, axis=-1)

        return local_anomaly_scores, query_distances, query_nns

    @staticmethod
    def _detection_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        self.local_nn_method.save(self._index_file(save_folder, f"{prepend}"))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        self.save(save_folder)
        self.local_nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        self.local_nn_method.load(self._index_file(load_folder, f"{prepend}"))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )
