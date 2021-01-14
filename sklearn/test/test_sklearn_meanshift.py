import pytest
import torch


N = 1000
C = 16
n_clusters = 4

@pytest.fixture
def generated_data():
    r"""
    """

    sample = torch.zeros(n_clusters*N, C)
