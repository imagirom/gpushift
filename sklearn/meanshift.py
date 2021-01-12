import torch
from gpushift.meanshift import MeanShiftStep
from pykeops.torch import LazyTensor

from sklearn.base import BaseEstimator, ClusterMixin

class ClusteringStep:
    r"""
    This class represents the clustering algorithm which finds the cluster
    centers after meanshift has moved its points sufficiently close together.
    """

    def __init__(self, bandwidth, kernel, distance_metric, max_clusters=500, use_keops=True):
        r"""
        Inputs:
            :param bandwidth: float
            :param kernel: unary operation
            :param distance_metric: binary operation
            :param max_clusters: int
            :param use_keops: bool
        """
        self.bandwidth = bandwidth
        self.kernel = kernel
        self.max_clusters = max_clusters
        self.use_keops = use_keops

    def __call__(self, points):
        r"""
        Inputs:
            :param points: torch.Tensor (B,N,C) <-> (batch, number of points,
                dimensionality/channels)

        Outputs:
            torch.Tensor (B,N) bool. True values indicate the the corresponding
                point from ''points'' is a cluster center.
        """
        B,N,_ = points.shape
        assert max_clusters < N

        points_i = points[:,None,:,:] # (B,1,N,C)
        points_j = points[:,:,None,:] # (B,N,1,C)

        if self.use_keops:
            points_i = LazyTensor(points_i)
            points_j = LazyTensor(points_j)

        pairwise_distances_ij = self.distance_metric(points_i, points_j) # (B,N,N,1)
        kernel_ij = self.kernel(pairwise_distances_ij) # (B,N,N,1)
        neighbours_ij = (self.bandwidth**2 - pairwise_distances_ij) > 0 # (B,N,N,1) bool

        intensities = kernel_ij.sum(2) # (B,N,1,1)
        possible_clusters = (-intensities).argKmin(self.max_clusters, dim=1) # (B,K,1,1)

        are_cluster_centers_ = torch.ones(B,N).bool()
        for b in range(B):
            possible_cluster_indices = possible_clusters[b] # (K,)
            possible_cluster_intensities = intensities[b, possible_cluster_indices] # (K,)

            argsort_res = torch.argsort(possible_cluster_intensities) # Ascending
            sorted_cluster_indices = possible_cluster_indices[argsort_res] # (K,)

            for idx in sorted_cluster_indices:
                are_neighbours = neighbours_ij[b,idx] # (N,1)
                are_cluster_centers_[b,are_neighbours] = False
                are_cluster_centers_[b,idx] = True

        return are_cluster_centers_

class MeanShift(BaseEstimator, ClusterMixin):
    r"""
    This class implements the sklean interface for a gpu-accelerated mean 
    shift algrithm.
    """

    def __init__(self, bandwidth, distance_metric='composite', kernel='gaussian', 
                       n_iter=15, max_clusters=500, blurring=False, use_keops=True):
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
        """
        self.meanshift_step = MeanShiftStep(bandwidth=bandwidth, kernel=kernel, use_keops=use_keops)
        self.clustering_step = ClusteringStep(bandwidth, self._get_kernel(kernel), 
            self._get_distance_metric(distance_metric), max_clusters=max_clusters, use_keops=use_keops)

        self.bandwidth = bandwidth
        self.kernel = kernel
        self.use_keops = use_keops

        self.cluster_centers_ = None

    def _get_kernel(self, kernel_name):
        r"""
        Dispatch kernel functions.

        Inputs:
            :param kernel_name: str

        Outputs:
            unary callable operation, is a kernel function.
        """
        _kernels = {
                'gaussian'  :       lambda x : ( -x / (2*bandwidth**2)).exp(),
                'epanechnikov'  :   lambda x : 1 - (x**2)/(bandwidth**2),
                'flat'  :           lambda x : x < bandwidth
        }

        return _kernels[kernel_name]

    def _get_distance_metric(self, distance_metric_name):
        r"""
        Dispatch distance metric functions.

        Inputs:
            :param distance_metric_name: str, descriptive name of the kernel.

        Outputs:
            binary callable operation, its inputs must have dimensions of (B,C),
            where B is a batch dimension, and C are the features.
        """
        _distance_metrics = {
                'euclidean'  :  lambda x,y : (x-y).square().sum(1).sqrt(),
                'spherical'  :  lambda x,y : 1 - x.dot(y),
                'composite'  :  # TODO
        }

        return _distance_metrics[distance_metric_name]

    def get_sp_composite_distance_metric(self):
        r"""
        """
        # TODO
        pass

    def fit(self, X):
        r"""
        Find and save cluster centers.

        Inputs:
            :param X: torch.Tensor (N,C), N number of samples, C 
                dimensions.
        """
        shifted = X[None,:,:] # (1,N,C)
        for _ in range(self.n_iter):
            if self.blurring:
                shifted = self.meanshift_step(shifted)
            else:
                shifted = self.meanshift_step(shifted, X)

        are_cluster_centers = self.clustering_step(shifted) # (1,N)

        self.cluster_centers_ = shifted[0, are_cluster_centers] # (n_centers, n_features)

    def predict(self, X):
        r"""
        """
        return None

    def fit_predict(self, X):
        r"""
        """
        return None
