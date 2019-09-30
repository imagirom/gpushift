import unittest
from unittest import TestCase
import torch
import numpy as np
from matplotlib import pyplot as plt
from gpushift.meanshift import MeanShiftStep


def scatter(tensor):
    plt.figure()
    plt.scatter(*tensor[0, :100, :2].detach().cpu().numpy().T, marker='.')


class TestMeanShiftStep(TestCase):
    def setUp(self):
        self.B, self.E, self.N = 4, 2, 10
        self.bandwidth = 1
        points = torch.randn(self.B, self.N, self.E)
        points = points.contiguous().cuda()
        points.requires_grad_()
        self.points = points

    def test_forward(self):
        results = {True: [], False: []}
        for use_keops in [True, False]:
            for kernel in MeanShiftStep.KERNELS:
                with self.subTest(use_keops=use_keops, kernel=kernel):
                    step = MeanShiftStep(bandwidth=self.bandwidth, kernel=kernel, use_keops=use_keops)
                    result = step(self.points).detach().cpu()
                    print(result.shape)
                    results[use_keops].append(result)
        for a, b in zip(results[True], results[False]):
            assert a.shape == b.shape
            assert np.allclose(a.numpy(), b.numpy(), atol=1e-6)


if __name__ == '__main__':
    unittest.main()
