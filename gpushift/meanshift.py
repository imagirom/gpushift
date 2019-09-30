import torch
from pykeops.torch import LazyTensor

# Dimension specifications:
# B: Samples in Batch
# N: Number of points to cluster
# E: Dimensionality of the space the points are embedded in


class MeanShiftStep(torch.nn.Module):
    GAUSSIAN_KERNEL = 'gaussian'
    FLAT_KERNEL = 'flat'
    EPANECHNIKOV_KERNEL = 'epanechnikov'
    KERNELS = [GAUSSIAN_KERNEL, FLAT_KERNEL, EPANECHNIKOV_KERNEL]

    def __init__(self, bandwidth, kernel='gaussian', use_keops=True):
        """
        Module encapsulating a single mean shift step.
        :param bandwidth: float
        Bandwidth of the kernel to be used, i.e. standard deviation for gaussian, radius for flat kernel.
        :param kernel: str
        Which kernel to use. Has to be one of MeanShiftStep.KERNELS .
        :param use_keops: bool
        Whether to use the PyKeOps Library or only vanilla PyTorch.
        """
        super(MeanShiftStep, self).__init__()
        self.bandwidth = bandwidth
        self.use_keops = use_keops
        self.kernel = kernel
        assert self.kernel in self.KERNELS, f'Kernel {kernel} not supported. Choose one of {self.KERNELS}'

    @staticmethod
    def mean_shift_step(points, reference=None, bandwidth=1, kernel='gaussian', use_keops=True):
        """
        Perform one mean shift step: Move points towards maxima of Kernel density estimate using reference.
        :param points: torch.FloatTensor or torch.cuda.FloatTensor
        Points that will be shifted. Shape should be arbitrary barch dimensions (B) followed by number of points (N) and
        :param reference: torch.FloatTensor
        Points used for the kernel density estimate. Shape should be identical to points, except in (N) dimension.
        :param bandwidth: float
        Bandwidth of the kernel to be used, i.e. standard deviation for gaussian, radius for flat kernel.
        :param kernel: str
        Which kernel to use. Has to be one of MeanShiftStep.KERNELS .
        :param use_keops: bool
        Whether to use the PyKeOps Library or only vanilla PyTorch.
        :return: torch.FloatTensor
        Shifted points. Shape identical to points.
        """
        reference = points if reference is None else reference
        assert reference.shape[-1] == points.shape[-1], f'{reference.shape}, {points.shape}'
        points_i = points[:, :, None, :]     # B N1 1 E
        points_j = reference[:, None, :, :]  # B 1 N2 E

        if use_keops:
            points_i = LazyTensor(points_i)
            points_j = LazyTensor(points_j)

        # array of vector differences
        v_ij = points_i - points_j  # B N1 N2 E

        # array of squared distances
        s_ij = (v_ij ** 2).sum(-1)  # B N1 N2 E

        if kernel == MeanShiftStep.GAUSSIAN_KERNEL:
            factor = points.new([-1 / (2 * bandwidth ** 2)])
            k_ij = (s_ij * factor).exp()             # B N1 N2 (1)
        elif kernel == MeanShiftStep.FLAT_KERNEL:
            squared_radius = points.new([bandwidth ** 2])
            k_ij = (-s_ij + squared_radius).step()   # B N1 N2 (1)
        elif kernel == MeanShiftStep.EPANECHNIKOV_KERNEL:
            squared_radius = points.new([bandwidth ** 2])
            k_ij = (-s_ij + squared_radius).relu()   # B N1 N2 (1)
        else:
            assert False, f'Kernel {kernel} not supported. Choose one of {MeanShiftStep.KERNELS}'

        if not use_keops:
            k_ij = k_ij.unsqueeze(-1)  # KeOps never squeezes last dim

        nominator = k_ij * points_j  # B N1 N2 E

        return nominator.sum(2) / k_ij.sum(2)  # B N1 E

    def forward(self, points, reference=None):
        """
        Perform one mean shift step: Move points towards maxima of Kernel density estimate using reference.
        :param points: torch.FloatTensor or torch.cuda.FloatTensor
        Points that will be shifted. Shape should be arbitrary barch dimensions (B) followed by number of points (N) and
        :param reference: torch.FloatTensor
        Points used for the kernel density estimate. Shape should be identical to points, except in (N) dimension.
        :return: torch.FloatTensor
        Shifted points. Shape identical to points.
        """
        # points should have shape B N E
        return self.mean_shift_step(points, reference,
                                    bandwidth=self.bandwidth, kernel=self.kernel, use_keops=self.use_keops)


class MeanShift(torch.nn.Module):
    def __init__(self, n_iter, bandwidth, kernel='gaussian', blurring=True, use_keops=True):
        """
        Module encapsulating a number of mean shift iterations.
        :param n_iter: int
        Number of shifts to perform.
        :param bandwidth: float
        Bandwidth of the kernel to be used, i.e. standard deviation for gaussian, radius for flat kernel.
        :param kernel: str
        Which kernel to use. Has to be one of MeanShiftStep.KERNELS .
        :param blurring: bool
        Weather to use the 'blurring' version of the algorithm, where the kernel density estimate is updated after every
        iteration.
        :param use_keops: bool
        Whether to use the PyKeOps Library or only vanilla PyTorch.
        """
        super(MeanShift, self).__init__()
        self.n_iter = n_iter
        self.blurring = blurring
        self.step = MeanShiftStep(bandwidth=bandwidth, kernel=kernel, use_keops=use_keops)

    def forward(self, points):
        """
        Perform a number of mean shift steps: Move points towards maxima of Kernel density estimate using reference.
        :param points: torch.FloatTensor
        Points that will be shifted. Shape should be arbitrary barch dimensions (B) followed by number of points (N) and
        dimensionality of the space they lie in (E).
        :return: torch.FloatTensor
        Shifted points. Shape identical to points.
        """
        # points should have shape B N E
        current = points
        for _ in range(self.n_iter):
            if self.blurring:
                current = self.step(current)
            else:
                current = self.step(current, points)
        return current
