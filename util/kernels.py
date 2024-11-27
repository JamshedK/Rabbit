import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
import torch
import gpytorch
from gpytorch.kernels import Kernel
from torch import Tensor

from gpytorch.lazy import LazyTensor, LinearOperator
from typing import Union



class NewKernel(Kernel):
    """
    define the kernel structure for GP
    :arg
    structure_info: dict, specify the **additive** kernel components and their hyper-params.
    e.g.{'squared_exp':{'gamma':1},'gaussian_noise':{'std':1}}
    """

    def __init__(self, structure_info, **kwargs):
        super().__init__(**kwargs)
        self.structure_info = structure_info
    
    # def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
    #     # 实现核的前向传播逻辑
    #     cov, _ = self.covar(x1.numpy(), x2.numpy())
    #     return torch.tensor(cov, dtype=torch.float32)
    
    def forward(self, x1: Tensor, x2: Tensor = None, diag: bool = False, last_dim_is_batch: bool = False, **params) :

         # 确保输入是 PyTorch 张量
        if isinstance(x1, np.ndarray):
            x1 = torch.tensor(x1, dtype=torch.float)
        if x2 is not None and isinstance(x2, np.ndarray):
            x2 = torch.tensor(x1, dtype=torch.float)

        x1 = x1.to(dtype=torch.float)
        if x2 is not None:
            x2 = x1.to(dtype=torch.float)

        if x1.dim() > 2:
            x1 = x1.view(-1, x1.size(-1))  # Flatten batch dimensions for x1
        if x2 is not None and x2.dim() > 2:
            x2 = x2.view(-1, x2.size(-1))  # Flatten batch dimensions for x2

        # 将 x1 和 x2 转换为 numpy 数组，以便与现有的核逻辑兼容
        x1_array = x1.detach().cpu().numpy()
        x2_array = x2.detach().cpu().numpy() if x2 is not None else None


        if x1_array.ndim != 2:
            raise ValueError(f"x1_array must be 2D, but got {x1_array.ndim}D")

        if x2_array is not None and x2_array.ndim != 2:
            raise ValueError(f"x2_array must be 2D, but got {x2_array.ndim}D")

        # 调用现有的协方差计算函数 (covar)
        cov, cov_grad = self.covar(x1_array, x2_array)

        # 将结果转换回 PyTorch 张量
        cov_tensor = torch.tensor(cov, dtype=torch.float32)

        if diag:
            # Return only the diagonal if requested
            cov_tensor = cov_tensor.diagonal()

        if last_dim_is_batch:
            # Reshape tensor if the last dimension is treated as a batch dimension
            cov_tensor = cov_tensor.unsqueeze(-1)

        return  cov_tensor

    def covar(self, x1_array, x2_array=None):
        """
        generate covariance matrix between the arrays given
        :param x1_array: nested array like
        :param x2_array: nested array like, Optional

        :return cov: covariance matrix between x1 and x2
        :return cov_grad: covariance gradient
        """

        cov_list = []
        cov_grad_list = []

        for kernel_name, kernel_params in self.structure_info.items():
            cov_i, cov_grad_i = getattr(self, kernel_name)(x1_array, x2_array, params=kernel_params)
            cov_list.append(cov_i)
            if kernel_name != 'gaussian_noise':
                cov_grad_list.append(cov_grad_i)
        cov = sum(cov_list)
        cov_grad = np.dstack(cov_grad_list)

        return cov, cov_grad

    @staticmethod
    def matern(x1_array, x2_array, params):
        """
        Matérn kernel class
        :arg
        x1_array, x2_array: nested array-like, list of points in search space
        params: must contain filed 'rho' as the free parameter of matern kernel
        nu: now fixed in 1.5, 2.5 or 3.5, Matérn covariance is ⌈ν⌉−1 times differentiable in the mean-square sense

        :return
        K: array, covariance matrix
        K_grad: array, covariance gradient
        """
        rho = params.get('rho')
        anisotropic = True if len(rho) > 1 else False
        sigma2 = params.get('sigma2', 1.0)
        nu = params.get('nu', 2.5)

        n1 = len(x1_array)
        n2 = len(x2_array) if x2_array is not None else n1

        if x2_array is None:  # Kxx must be symmetric, so we can save some labor accordingly
            K1 = np.full((n1, n2), sigma2)
            dists = pdist(x1_array / rho, metric='euclidean')

            if nu == 0.5:
                K2 = np.exp(-dists)
            elif nu == 1.5:
                K2 = dists * np.sqrt(3)
                K2 = (1. + K2) * np.exp(-K2)
            elif nu == 2.5:
                K2 = dists * np.sqrt(5)
                K2 = (1. + K2 + K2 ** 2 / 3.0) * np.exp(-K2)

            K2 = squareform(K2)
            np.fill_diagonal(K2, 1)
            K = K1 * K2

            # compute grad
            K1_grad = np.full((n1, n2), sigma2)[:, :, np.newaxis]

            if anisotropic:
                D = (x1_array[:, np.newaxis, :] - x1_array[np.newaxis, :, :]) ** 2 / (rho ** 2)
            else:
                D = squareform(dists ** 2)[:, :, np.newaxis]

            if nu == 0.5:
                K2_grad = K[..., np.newaxis] * D / np.sqrt(D.sum(2))[:, :, np.newaxis]
                K2_grad[~np.isfinite(K2_grad)] = 0
            elif nu == 1.5:
                K2_grad = 3 * D * np.exp(-np.sqrt(3 * D.sum(-1)))[..., np.newaxis]
            elif nu == 2.5:
                tmp = np.sqrt(5 * D.sum(-1))[..., np.newaxis]
                K2_grad = 5.0 / 3.0 * D * (tmp + 1) * np.exp(-tmp)

            K_grad = np.dstack((K1_grad * K2[:, :, np.newaxis],
                                K2_grad * K1[:, :, np.newaxis]))

        else:  # Kxy, we calculate for every entry, only used for prediction, no grad needed
            dists = cdist(x1_array / rho, x2_array / rho, metric='euclidean')

            if nu == 0.5:
                K = np.exp(-dists)
            elif nu == 1.5:
                K = dists * np.sqrt(3)
                K = (1. + K) * np.exp(-K)
            elif nu == 2.5:
                K = dists * np.sqrt(5)
                K = (1. + K + K ** 2 / 3.0) * np.exp(-K)

            K1 = np.full((n1, n2), sigma2)
            K = K * K1
            K_grad = np.zeros([n1, n2, len(rho) + 1])

        return K, K_grad

    @staticmethod
    def _matern(nu, rho, sigma2, dist):
        """
        covariance for each entry (not used)
        """
        assert nu in [1.5, 2.5, 3.5]
        cov = np.nan
        cov_grad1 = np.nan
        cov_grad2 = np.nan

        if nu == 1.5:
            k1 = sigma2
            k2 = (1 + np.sqrt(3) * dist / rho) * np.exp(-np.sqrt(3) * dist / rho)
            k1_grad = sigma2
            k2_grad = 3 * (dist / rho) ** 2 * np.exp(-np.sqrt(3) * dist / rho)

            cov = k1 * k2
            cov_grad1 = k2 * k1_grad
            cov_grad2 = k1 * k2_grad

        elif nu == 2.5:
            k1 = sigma2
            k2 = (1 + np.sqrt(5) * dist / rho + 5 * dist ** 2 / (3 * rho ** 2.0)) * np.exp(
                -np.sqrt(5) * dist / rho)
            k1_grad = sigma2
            k2_grad = 5 / 3 * (dist / rho) ** 2 * (1 + np.sqrt(5) * dist / rho) * np.exp(- np.sqrt(5) * dist / rho)

            cov = k1 * k2
            cov_grad1 = k2 * k1_grad
            cov_grad2 = k1 * k2_grad

        elif nu == 3.5:
            cov = sigma2 * (1 + np.sqrt(7) * dist / rho + 14 * dist ** 2 / (5 * rho ** 2)
                            + (np.sqrt(7) * dist / rho) ** 3 / 15) * np.exp(-np.sqrt(7) * dist / rho)

        return cov, cov_grad1, cov_grad2

    @staticmethod
    def gaussian_noise(x1_array, x2_array, params):
        """
        Gaussian white noise
        :arg
        x1_array, x2_array: nested array-like, list of points in search space
        params: must contain filed 'std' as the std of Gaussian white noise
        :return
        array, covariance matrix
        """

        noise = np.clip(params.get('std2'), 1e-6, a_max=None)

        n1 = len(x1_array)
        n2 = len(x2_array) if x2_array is not None else n1

        white_noise = np.zeros([n1, n2])
        white_noise_grad = np.empty([n1, n2])

        if x2_array is None:
            # add noise variance to the diagonal
            np.fill_diagonal(white_noise, noise)
            # white_noise_grad = noise * np.eye(n1, n2)[:, :, np.newaxis]

        return white_noise, white_noise_grad # as we do not estimate std2 and using observations directly
