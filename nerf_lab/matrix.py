import torch
import numpy as np

def rotation_matrix_alpha(alpha: float = 0) -> torch.Tensor:
    matrix = [
        [1, 0, 0, 0],
        [0, np.cos(alpha), np.sin(alpha), 0],
        [0, -np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 0, 1],
    ]

    return torch.tensor(matrix).float()

def rotation_matrix_beta(beta: float = 0) -> torch.Tensor:
    matrix = [
        [ np.cos(beta), 0, -np.sin(beta), 0 ],
        [ 0, 1, 0, 0 ],
        [ np.sin(beta), 0, np.cos(beta), 0 ],
        [ 0, 0, 0, 1 ]
    ]

    return torch.tensor(matrix).float()

def rotation_matrix_gamma(gamma: float = 0) -> torch.Tensor:
    matrix = [
        [ np.cos(gamma), np.sin(gamma), 0, 0 ],
        [ -np.sin(gamma), np.cos(gamma), 0, 0 ],
        [ 0, 0, 1, 0 ],
        [ 0, 0, 0, 1 ]
    ]

    return torch.tensor(matrix).float()

def rotation_matrix(*, alpha: float = 0, beta: float = 0, gamma: float = 0) -> torch.Tensor:
    malpha = rotation_matrix_alpha(alpha)
    mbeta = rotation_matrix_beta(beta)
    mgamma = rotation_matrix_gamma(gamma)

    return torch.matmul(torch.matmul(malpha, mbeta), mgamma)

def translation_matrix(*, x: float = 0, y: float =  0, z: float = 0) -> torch.Tensor:
    matrix = [
        [ 1, 0, 0, x ],
        [ 0, 1, 0, y ],
        [ 0, 0, 1, z ],
        [ 0, 0, 0, 1 ]
    ]

    return torch.tensor(matrix).float()

def transform_matrix(*,
        alpha: float = 0, beta: float = 0, gamma: float = 0,
        x: float = 0, y: float = 0, z: float = 0) -> torch.Tensor:

    rotation = rotation_matrix(alpha=alpha, beta=beta, gamma=gamma)
    translation = translation_matrix(x=x, y=y, z=z)

    return torch.matmul(rotation, translation)
