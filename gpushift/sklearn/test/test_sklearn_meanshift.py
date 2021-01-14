import pytest
import torch
from torch.distributions.normal import Normal
from gpushift.sklearn.meanshift import MeanShift
from sp_tracking.meanshift import MeanShiftForeground
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from sp_tracking.transforms import add_XY

N = 1000
C = 16

gaussian_scale = 0.2
atol = 3*gaussian_scale/N**2 # Tolerance within 3x error in the mean.

@pytest.fixture
def generated_data():
    r"""
    Simple generated Gaussian cluster data. There are 4 clusters located along
    different axes of the C-dimensional space.
    """
    sample = torch.zeros(4*N, C)

    # Locations of the cluster centers
    loc1 = torch.zeros(C); loc1[0] = 5*gaussian_scale
    loc2 = torch.zeros(C); loc2[1] = 5*gaussian_scale
    loc3 = torch.zeros(C); loc3[2] = 5*gaussian_scale
    loc4 = torch.zeros(C); loc4[3] = 5*gaussian_scale

    locs = [loc1, loc2, loc3, loc4]

    for idx, loc in enumerate(locs):
        sample[idx*N : (idx+1)*N] = Normal(loc, gaussian_scale).sample(sample_shape=torch.Size([N]))

    return sample, locs

@pytest.fixture
def self_embedding():
    r"""
    """
    return torch.load("tile_02000.tensor")

def test_MeanShift(generated_data):
    r"""
    Simple check that predicted cluster centers match the ground truth ones.
    """
    sample, locs = generated_data

    ms = MeanShift(0.7, distance_metric='euclidean', kernel='gaussian')
    ms.fit(sample)
    cluster_centers = ms.cluster_centers_

    belongings = ms.predict(sample)
    reducer = PCA().fit(sample)
    for idx in range(len(locs)):
        reduced = reducer.transform(sample[ belongings == idx ])
        plt.scatter( reduced[:,0], reduced[:,1], alpha=0.07)

    plt.show()

    assert len(cluster_centers) == len(locs)

    # Match each predicted cluster center with the ground truth locations.
    for center in cluster_centers:
        assert locs, "Locations list should not be empty"
        for idx,loc in enumerate(locs):
            if torch.isclose(center, loc, atol=atol): locs.pop(idx) 

    assert not locs, "Locations list should be empty"

def test_CompositeSpace(self_embedding):
    r"""
    Test the algorithm on a real dataset tile.

    Inputs:
        :fixture self_embedding: (1,C,H,W)
    """
    _,C,H,W = self_embedding.shape

    ms = MeanShiftForeground(0.32, distance_metric='euclidean', kernel='gaussian', use_keops=True)

    self_embedding = add_XY(self_embedding[0], scale_down=0.01)
    outp_ = ms.fit_predict(self_embedding[None,:,None,:,:])

    plt.imshow(outp_.reshape(H,W)); plt.show()
