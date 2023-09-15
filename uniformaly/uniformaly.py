import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import time

import uniformaly
import uniformaly.encoder
import uniformaly.common
import uniformaly.sampler
import uniformaly.cluster

import matplotlib.pyplot as plt

LOGGER = logging.getLogger(__name__)


class Uniformaly(torch.nn.Module):
    def __init__(self, device, params):
        """Uniformaly anomaly detection class."""
        super(Uniformaly, self).__init__()
        self.device = device
        self.params = params

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        bpm=True,
        patchsize=7,
        anomaly_scorer_num_nn=1,
        featuresampler=uniformaly.sampler.IdentitySampler(),
        local_nn_method=uniformaly.common.FaissNN(False, 4),
        topk=0.01,
        lmda=0.0,
        thres=0.1,
        temp=0.0,
        return_topk_index=False,
        agg_type='layer',
        **kwargs,
    ):
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.bpm = bpm

        self.device = device
        
        assert topk > 0.0, "Top k-ratio should be greater than zero. Please retry."
        
        self.patch_scorer = PatchScorer(topk, return_topk_index)
        if self.bpm:
            self.attention_mask = AttentionMask(self.backbone, self.layers_to_extract_from[-1], patchsize, rollout='sup' in self.backbone.name)

        self.forward_modules = torch.nn.ModuleDict({})

        feature_aggregator = uniformaly.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )

        self.forward_modules["feature_aggregator"] = feature_aggregator

        self.agg_type = agg_type
        if agg_type == 'layer':
            layer_aggregator = uniformaly.common.LayerwiseAggregator()
            _ = layer_aggregator.to(self.device)
        elif agg_type == 'channel':
            layer_aggregator = uniformaly.common.ChannelwiseAggregator()
            _ = layer_aggregator.to(self.device)
        elif agg_type == 'patch':
            layer_aggregator = uniformaly.common.PatchwiseAggregator()
            _ = layer_aggregator.to(self.device)
        else:
            layer_aggregator = uniformaly.common.ConcatAggregator()
            _ = layer_aggregator.to(self.device)

        self.forward_modules["layer_aggregator"] = layer_aggregator

        self.anomaly_scorer = uniformaly.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_scorer_num_nn, local_nn_method=local_nn_method, temp=temp
        )

        self.anomaly_segmentor = uniformaly.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        self.featuresampler = featuresampler
        self.lmda = lmda
        self.thres = thres
        self.topk_index = return_topk_index

    def embed(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=False, provide_patch_shapes=False):
        """Returns feature embeddings for images."""

        def _detach(features):
            if detach:
                return features.detach().cpu().numpy()
            return features

        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)
        features = [features[layer] for layer in self.layers_to_extract_from] 

        if "vit" in self.backbone.name:
            patch_shapes = [(int(x.shape[1]**0.5), int(x.shape[1]**0.5)) for x in features]

            features = [x.unsqueeze(1) for x in features]
            
            features = torch.cat(features, dim=1)

            features = self.forward_modules["layer_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """Uniformaly training.

        This function computes the embeddings of the training data and fills the
        memory bank of Uniformaly.
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """Computes and sets the support features for Uniformaly."""
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                _features = _image_to_features(image)[:, 1:, :]
                
                # apply bpm
                if self.bpm:
                    _attn_mask = self.attention_mask.get_attention(image.cuda())
                    
                    # binarize
                    _attn_mask = _attn_mask > self.thres
                    
                    # apply masking
                    _features *= _attn_mask.detach().unsqueeze(-1)
                    
                    nonzero_idx = torch.unique(_features.nonzero(as_tuple=True)[1])
                    _features = _features[:, nonzero_idx, :].reshape(-1, _features.shape[-1])
                    
                features.append(_features)
                torch.cuda.empty_cache()
                

        features = torch.cat(features)
        features = features.reshape(-1, features.shape[-1])
        
        # subsampling
        features = self.featuresampler.run(features)

        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []

        if self.topk_index:
            topk_features = []

        # Measure FPS
        start_time = time.time()
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=True) as data_iterator:
            for i, image in enumerate(data_iterator):
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                                    
                if self.topk_index:
                    _scores, _masks, _topk_features = self._predict(image)
                    topk_features.append(_topk_features)
                else:
                    _scores, _masks = self._predict(image)

                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        end_time = time.time()
        print("FPS:", len(dataloader)/(end_time-start_time))
        
        if self.topk_index:
            return scores, masks, labels_gt, masks_gt, topk_features
        
        
        return scores, masks, labels_gt, masks_gt


    def _predict(self, images):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = features[:, 1:, :] # exclude global tokens
                
            pred = self.anomaly_scorer.predict(features.cpu().numpy())
            patch_scores = image_scores = pred[0] # [784,]

            image_scores = self.patch_scorer.unpatch_scores(
                image_scores, batchsize=batchsize
            ) 
            
            ######## Apply Pooled Attention Mask #########
            if self.bpm:
                attn_mask = self.attention_mask.get_attention(images)
                if self.thres > 0.0:
                    attn_mask = attn_mask > self.thres
                else:
                    pass
                
                image_scores *= attn_mask.detach().cpu().numpy() 

            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)

            
            image_scores = self.patch_scorer.score(image_scores)

            if self.topk_index:
                topk_index = image_scores[1]
                image_scores = image_scores[0]
                topk_features = np.take_along_axis(features, topk_index, axis=1)

            # Scale the score 
            scores = image_scores

            
            patch_scores = self.patch_scorer.unpatch_scores(
                patch_scores, batchsize=batchsize
            ) # patch_scores for segmentation
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            if self.topk_index:
               return [score for score in scores], [mask for mask in masks], topk_features 

        return [score for score in scores], [mask for mask in masks]
    
    @staticmethod
    def _params_file(filepath, prepend=""):
        return os.path.join(filepath, prepend + "uniformaly_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        LOGGER.info("Saving Uniformaly data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        if self.params:
            uniformaly_params = self.params
            uniformaly_params['input_shape'] = self.input_shape

            try:
                uniformaly_params['backbone.name'] = uniformaly_params['backbone_name']
                del uniformaly_params['backbone_name']
            except KeyError:
                pass

        else:
            uniformaly_params = {
                "backbone.name": self.backbone.name,
                "layers_to_extract_from": self.layers_to_extract_from,
                "input_shape": self.input_shape,
                "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
            }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(uniformaly_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        local_nn_method: uniformaly.common.FaissNN(False, 4),
        topk: float=0.01,
        bpm: bool=True,
        thres: float=0.1,
        anomaly_scorer_nn: int=1,
        patchsize: int=7,
        prepend: str = "",
    ) -> None:
        LOGGER.info("Loading and initializing Uniformaly.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            uniformaly_params = pickle.load(load_file)
        
        uniformaly_params["backbone"] = uniformaly.encoder.load(
            uniformaly_params["backbone.name"]
        )
        uniformaly_params["backbone"].name = uniformaly_params["backbone.name"]
        del uniformaly_params["backbone.name"]
        
        uniformaly_params["topk"] = topk
        uniformaly_params["bpm"] = bpm
        uniformaly_params["thres"] = thres
        uniformaly_params["anomaly_scorer_num_nn"] = anomaly_scorer_nn
        uniformaly_params["patchsize"] = patchsize
        
        self.load(**uniformaly_params, device=device, local_nn_method=local_nn_method)

        self.anomaly_scorer.load(load_path, prepend)




class PatchScorer:
    def __init__(self, k, return_index):
        self.k = k
        self.return_index = return_index
    
    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        
        k_num = int(round(self.k*x.shape[1]))
        topk = torch.topk(x, min(x.shape[1], k_num), dim=1)
        x = torch.mean(topk.values, dim=1).reshape(-1)

        if self.return_index:
            if was_numpy:
                return x.numpy(), topk.indices.numpy()
            return x, topk.indices

        if was_numpy:
            return x.numpy()
        return x


class AttentionMask:
    """
    1) 2-D realignment of attention vectors
    2) average pooling with fixed resolution
    3) masking to score
    """
    def __init__(self, model, layer, kernel_size=5, stride=1, rollout=False):
        self.model = model
        if isinstance(layer, str):
            self.layer = int(layer)
        elif isinstance(layer, list):
            self.layer = list(map(int, layer))
        else:
            raise TypeError("Layer should be a single scalar or a list of scalars.")

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2

        self.rollout=rollout

    def get_attention(self, images):
        b = images.shape[0]

        attentions = self.model.get_all_selfattention(images)
        attentions = torch.stack(attentions)
        
        # Averaging across heads
        attentions = torch.mean(attentions, dim=2)

        if self.rollout:
            res_attn = torch.eye(attentions.size(-1)).cuda()

            aug_att_mat = attentions + res_attn

            joint_attentions = torch.zeros(aug_att_mat.size()).cuda()
            joint_attentions[0] = aug_att_mat[0]

            for i in range(1, len(attentions)):
                joint_attentions[i] = torch.matmul(aug_att_mat[i],joint_attentions[i-1])
            
            attentions = joint_attentions

        if isinstance(self.layer, list):
            attentions = torch.mean(attentions[self.layer], dim=0)
        else:
            attentions = attentions[self.layer]
        
        # get an attention score of the global token
        if int(np.sqrt(attentions.shape[-1])) == np.sqrt(attentions.shape[-1]): # when global token not available
            attentions = torch.mean(attentions, dim=1)
        else:        
            attentions = attentions[:, 0, 1:]

        # realign attention score to a 2-D array
        grid_size = int(np.sqrt(attentions.shape[-1]))
        attentions = attentions.reshape(b, grid_size, grid_size)

        pooled = self.pool(attentions).flatten(start_dim=1)

        return pooled

    def pool(self, attn):
        """
        Apply pooling to 2d-attention map
        
        - Args
            attn (torch.Tensor): 2-d attention map

        - Returns
            pooled (torch.Tensor): 2-d normalized & pooled attention map
        """
        pooling = torch.nn.AvgPool2d(kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        pooled = pooling(attn)

        return pooled / pooled.max()
