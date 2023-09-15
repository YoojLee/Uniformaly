from abc import ABC, abstractmethod
import os

import numpy as np
import torch
from sklearn.cluster import KMeans, DBSCAN
from sklearn import metrics
from munkres import Munkres

import torch

# for visualization
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt 
import seaborn as sns

RESULTS_PATH="uniformaly_results/clustering/topk"
os.makedirs(RESULTS_PATH, exist_ok=True)

N_CLUSTER_DICT = {'wood': 6, 
                  'metal_nut': 5, 
                  'tile': 6, 
                  'pill': 8, 
                  'cable': 9, 
                  'toothbrush': 2, 
                  'carpet': 6, 
                  'transistor': 5, 
                  'bottle': 4, 
                  'hazelnut': 5, 
                  'capsule': 6, 
                  'leather': 6, 
                  'screw': 6, 
                  'grid': 6, 
                  'zipper': 8,
                  'mtd': 6}

class AnomalyCluster(ABC):
    def __init__(self, feature, agg_mode, classname):
        if isinstance(feature, list):
            self.feature = torch.cat(feature).squeeze()

        if agg_mode not in ['mean', 'cat']:
            raise ValueError("Invalid aggregation mode. It should be either 'mean' or 'cat'.")
        self.agg_mode = agg_mode
        self.fname = f"{RESULTS_PATH}/{classname}_clustering_results.png"
        self.classname=classname.replace("mvtec_","")

    @abstractmethod
    def get_cluster(self):
        raise NotImplementedError

    def visualize(self):
        agg_feat, cluster = self.get_cluster()
        scaler = StandardScaler()
        scaled = scaler.fit_transform(agg_feat)

        tsne = TSNE(n_components=2)
        reduced = tsne.fit_transform(scaled)
        
        sns.scatterplot(x=reduced[:,0], y=reduced[:,1], hue=cluster, style=cluster, palette="Set2", data=reduced, s=50)

        plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
        plt.title(self.classname, fontsize=16)
        plt.savefig(self.fname, bbox_inches='tight')
        plt.cla() # flush previous plots

    def aggregate_feature(self):

        if self.feature.ndim == 2:
            return self.feature
        
        if self.agg_mode == 'mean':
            agg = self.feature.mean(axis=1)
            
        elif self.agg_mode == 'cat':
            n,_,_ = self.feature.shape
            agg = self.feature.reshape(n,-1)
        
        return agg

    # Code for Clustering Evaluation is borrowed from https://github.com/bytedance/ibot/blob/main/evaluation/unsupervised/unsup_cls.py
    def evaluate(self, cluster_assignment, labels_true, calc_f1=True):
        # compute NMI and ARI
        nmi = metrics.normalized_mutual_info_score(labels_true, cluster_assignment)
        ari = metrics.adjusted_rand_score(labels_true, cluster_assignment)
        f1 = -1

        # get y predictions
        if calc_f1:
            y_pred = self.get_y_preds(cluster_assignment, labels_true, len(set(labels_true)))
            f1 = metrics.f1_score(labels_true, y_pred, average='micro')
        
        return {
            "nmi": nmi, "ari": ari, "f1":f1
        }

        
    def get_y_preds(self, cluster_assignment, labels_true, n_clusters):

        confusion_matrix = metrics.confusion_matrix(labels_true, cluster_assignment, labels=None)
        # compute accuracy based on optimal 1:1 assignment of clusters to labels
        cost_matrix = self.calculate_cost_matrix(confusion_matrix, n_clusters)
        indices = Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = self.get_cluster_labels_from_indices(indices)

        if np.min(cluster_assignment) != 0:
            cluster_assignment = cluster_assignment - np.min(cluster_assignment)
        y_pred = kmeans_to_true_cluster_labels[cluster_assignment]

        return y_pred
    
    def calculate_cost_matrix(self, C, n_clusters):
        cost_matrix = np.zeros((n_clusters, n_clusters))
        # cost_matrix[i,j] will be the cost of assigning cluster i to label j
        for j in range(n_clusters):
            s = np.sum(C[:, j])  # number of examples in cluster i
            for i in range(n_clusters):
                t = C[i, j]
                cost_matrix[j, i] = s - t

        return cost_matrix
    
    def get_cluster_labels_from_indices(self,indices):
        n_clusters = len(indices)
        cluster_labels = np.zeros(n_clusters)
        for i in range(n_clusters):
            cluster_labels[i] = indices[i][1]

        return cluster_labels
    
    def scaling(self, x):
        scaler = StandardScaler()
        scaled = scaler.fit_transform(x)
        
        return scaled

class KMeansCluster(AnomalyCluster):
    def __init__(self, feature, agg_mode, classname, k=-1):
        super(KMeansCluster, self).__init__(feature, agg_mode, classname)
        if k == -1:
            self.k = N_CLUSTER_DICT[classname.replace('mvtec_', '')]
        else:
            self.k = k

    def get_cluster(self):
        agg = self.aggregate_feature()
        if type(agg) == torch.Tensor:
            agg = agg.cpu().numpy()
        cluster = KMeans(n_clusters=self.k, n_init=10, random_state=42).fit_predict(agg)

        return agg, cluster

class DBSCANCluster(AnomalyCluster):
    def __init__(self, feature, agg_mode, classname):
        super(DBSCANCluster, self).__init__(feature, agg_mode, classname)

    def get_cluster(self):
        agg = self.aggregate_feature()
        cluster = DBSCAN().fit_predict(agg)

        return agg, cluster
