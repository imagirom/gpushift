import torch
from gpushift.meanshift import MeanShiftStep
from pykeops.torch import LazyTensor

from sklearn.base import BaseEstimator, ClusterMixin

import hdbscan
import numpy as np
from tqdm import tqdm
import sys

def obtain_hard_clusters(points, thresholds):
    r"""
    Copied from:
    https://github.com/ModShift/ModShift/blob/master/utils/obtain_hard_clusters.py
    """

    # expects points as B * ? * E, thresholds as list
    # returns single linkage labels of shape B len(thresholds) ?
    clusterer = hdbscan.HDBSCAN(min_samples=1, approx_min_span_tree=False)
    labels_list = []
    for batch_id in tqdm(range(points.shape[0]), desc="Processing batches", leave=False):
        clusterer.fit(points[batch_id].reshape(-1, points.shape[-1]))
        labels_batch = []
        for threshold in tqdm(thresholds, desc="Cutting at thresholds", leave=False):
            labels_batch_threshold = clusterer.single_linkage_tree_.get_clusters(cut_distance=threshold, min_cluster_size=1)
            labels_batch.append(labels_batch_threshold.reshape(*points.shape[1:-1]))
        labels_list.append(labels_batch)
        tqdm.write("Done with hard clustering batch {}".format(batch_id))
    return np.array(labels_list)

class ClusteringStep:
    r"""
    This class represents the clustering algorithm which finds the cluster
    centers after meanshift has moved its points sufficiently close together.
    """

    def __init__(self, bandwidth, kernel, distance_metric, max_clusters=500, use_keops=False,
            cluster_center_function=None):
        r"""
        Inputs:
            :param bandwidth: float
            :param kernel: unary operation
            :param distance_metric: str, identifier
            :param max_clusters: int
            :param use_keops: bool
            :param cluster_center_function: None or callable
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.distance_metric = distance_metric
        self.max_clusters = max_clusters
        self.use_keops = use_keops
        self.cluster_center_function = cluster_center_function

    def __call__(self, points):
        r"""
        Inputs:
            :param points: torch.Tensor (N,C) <-> (batch, number of points,
                dimensionality/channels)

        Outputs:
            torch.Tensor (K,C) float. K clusters, C features. Each row
            represents a cluster center location.
        """
        if self.use_keops:
            raise NotImplementedError()

        N,C = points.shape

        unlabelled_clusters = torch.arange(N)
        cluster_belonging = torch.zeros(N)

        counter = 1
        while counter < self.max_clusters+1:
            indices_unlabelled = torch.nonzero(unlabelled_clusters)

            if len(indices_unlabelled) == 0:
                # All points are labelled.
                break

            idx_unlabelled = indices_unlabelled[0,0].item()
            query_point = points[idx_unlabelled,:].unsqueeze(0)

            distances_from_point = self.distance_metric(points, query_point).squeeze(-1) # (N)
            neighbours_mask = distances_from_point < self.bandwidth
            unlabelled_clusters[neighbours_mask] = 0
            cluster_belonging[neighbours_mask] = counter # Counter also works as cluster id.

            counter += 1

        unique_cluster_ids = torch.unique(cluster_belonging)
        num_clusters = len(unique_cluster_ids)
        cluster_centers = torch.zeros(num_clusters,C)
        for new_idx, cluster_id in enumerate(unique_cluster_ids):
            if self.cluster_center_function is None:
                cluster_centers[new_idx, :] = points[ cluster_belonging == cluster_id ].mean(0)
            else:
                cluster_centers[new_idx, :] = self.cluster_center_function( 
                        points[ cluster_belonging == cluster_id ][None,...] )[0]

        return cluster_centers

class MeanShift(BaseEstimator, ClusterMixin):
    r"""
    This class implements the sklean interface for a gpu-accelerated mean 
    shift algrithm.
    """

    def __init__(self, bandwidth, distance_metric='composite', kernel='gaussian', 
                       n_iter=15, max_clusters=500, blurring=False, use_keops=True,
                       meanshift_step_transform=None, cluster_center_function=None):
        r"""
        Inputs:
            :param bandwidth: int, bandwidth of the kernel. Some kernels such
                as 'gaussian' have infinite support, so this parameter becomes
                the sigma of the curve. For other kernels such as 'flat' and 
                'epanechnikov', it sets the support of the kernel.

            :param distance_metric: str

            :param kernel: str, one of pre-defined values which controls which
                kernel the algorithm will use.

            :param n_iter: int

            :param max_clusters: int

            :param blurring: bool, whether to run the meanshift algorithm in a
                blurring way (each point moves and the KDE landscape changes 
                at each iteration), or non-blurring (keep the KDE landscape
                constant, and move a set of reference points up the gradient
                instead).

            :param use_keops: bool, whether to use the gpu-accelerated version
                of mean shift.

            :param meanshift_step_transform: callable or None. This function is
                called on the shifted points after each iteration of the shift
                step. Its input is (n_samples, n_features).

            :param cluster_center_function: callable or None. If None, the
                cluster centers will be calculated as the mean over all dims of
                the selected points (during the clustering step). Can pass a
                custom function, which preserves certain properties of the
                embeddings (such as spherical embeddings), or has any additional
                required properties.
        """
        self.bandwidth = bandwidth
        self.distance_metric = distance_metric
        self.kernel = kernel
        self.n_iter = n_iter
        self.max_clusters = max_clusters
        self.blurring = blurring
        self.use_keops = use_keops

        self.cluster_centers_ = None

        self.meanshift_step = MeanShiftStep(bandwidth=bandwidth, kernel='custom', 
                use_keops=use_keops, 
                distance_metric=self._get_distance_metric(distance_metric),
                custom_kernel=self._get_kernel(kernel))

        self.meanshift_step_transform = meanshift_step_transform

        self.clustering_step = ClusteringStep(bandwidth, self._get_kernel(kernel), 
                self._get_distance_metric(distance_metric),
                max_clusters=max_clusters, use_keops=False,
                cluster_center_function=cluster_center_function)

    def _get_kernel(self, kernel_name):
        r"""
        Dispatch kernel functions.

        Inputs:
            :param kernel_name: str

        Outputs:
            unary callable operation, is a kernel function.
        """
        _kernels = {
                'gaussian'  :       lambda x : ( -x / (2*self.bandwidth**2)).exp(),
                'epanechnikov'  :   lambda x : (1 - (x**2)/(self.bandwidth**2)).relu(),
                'flat'  :           lambda x : (x - self.bandwidth).step()
        }

        return _kernels[kernel_name]

    def _get_distance_metric(self, distance_metric_name):
        r"""
        Dispatch distance metric functions.

        Inputs:
            :param distance_metric_name: str, descriptive name of the kernel.

        Outputs:
            binary callable operation, its inputs are (B, N, 1, C), and
            and (B,1,M,C). The output shape is (B, N, M).
        """
        _distance_metrics = {
                'euclidean'  :  lambda x,y : (x-y).square().sum(-1).sqrt(),
                'spherical'  :  lambda x,y : 1 - (x*y).sum(-1)**2,
                'composite'  :  None
        }

        # Define separately to avoid huge lambda function
        def _composite(x,y):
            r"""
            Assumed that the last two dimensions are flat, and the rest
            are spherical.
            """
            euclid_ = _distance_metrics['euclidean']
            sphere_ = _distance_metrics['spherical']
            if isinstance(x, LazyTensor):
                _,_,_, C = x.shape
                _,_,_, Cy = y.shape
                assert C == Cy

                return euclid_(x[C-2:C],y[C-2:C])**2 + sphere_(x[0:C-2],y[0:C-2])**2
            else:
                return euclid_(x[...,-2:],y[...,-2:])**2 + sphere_(x[...,:-2],y[...,:-2])**2

        _distance_metrics['composite'] = _composite

        return _distance_metrics[distance_metric_name]

    def fit(self, X):
        r"""
        Find and save cluster centers.

        Inputs:
            :param X: torch.Tensor (N,C), N number of samples, C 
                dimensions.
        """
        #from sp_tracking.visuals import visualize_PCA
        #import matplotlib.pyplot as plt
        #import pdb; pdb.set_trace()
        N,_ = X.shape

        shifted = X[None,:,:] # (1,N,C)
        X_cp = X[None,:,:]
        for _ in range(self.n_iter):

            if N == 1:
                # Corner case of passing a single point. Cannot apply the mean
                # shift step in that case.
                print("Attempted meanshift on a single point.")
                break

            if self.blurring:
                shifted = self.meanshift_step(shifted)
            else:
                shifted = self.meanshift_step(shifted, X_cp)

            if self.meanshift_step_transform is not None:
                shifted = self.meanshift_step_transform(shifted)

        self.cluster_centers_ = self.clustering_step(shifted[0])

    def predict(self, X):
        r"""
        Predict cluster belonging based on which cluster center is the closest.

        Inputs:
            :param X: torch.Tensor, (n_samples, n_features)

        Outputs:
            torch.Tensor, (n_samples,) int, indices identifying for each
            datapoint which cluster center it belongs to.
        """
        metric_function = self._get_distance_metric(self.distance_metric)
        N_, _ = X.shape
        K_, _ = self.cluster_centers_.shape

        if K_ == 1:
            # All points will belong to cluster 0.
            return torch.zeros(N_)

        if self.use_keops:
            points_i = LazyTensor(X[None,:,None,:]) # (B=1, N, 1, C)
            points_j = LazyTensor(self.cluster_centers_[None,None,:,:]) # (B=1, 1, K, C)

            cluster_id_ = metric_function(points_i, points_j).argmin(dim=2)[0,:,0] # rm B,C dims
        else:
            # TODO Untested
            distances_to_centers = metric_function(X[:,None,:], self.cluster_centers_[None,:,:]) # (N,K)
            cluster_id_ = torch.argmin(distances_to_centers, dim=1)

        return cluster_id_

    def fit_predict(self, X):
        r"""
        """
        self.fit(X)

        return self.predict(X)
