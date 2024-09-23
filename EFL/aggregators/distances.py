import numpy as np
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm


def calculate_cosine_similarity(
    parameter_matrix: np.array
):
    cosmat = np.zeros(shape=(parameter_matrix.shape[0], parameter_matrix.shape[0]))
    for i in range(parameter_matrix.shape[0]):
        for j in range(parameter_matrix.shape[0]):
            if i == j:
                cosmat[i, j] = 1
            elif i > j:
                cosmat[i, j] = cosmat[j, i]
            else:
                cosmat[i, j] = dot(parameter_matrix[i, :], parameter_matrix[j, :]) / ((norm(parameter_matrix[i, :])) * norm(parameter_matrix[j, :]))
    return cosmat


def calculate_cosine_dist(
    parameter_matrix: np.array
):
    cosmat = np.zeros(shape=(parameter_matrix.shape[0], parameter_matrix.shape[0]))
    for i in range(parameter_matrix.shape[0]):
        for j in range(parameter_matrix.shape[0]):
            if i == j:
                cosmat[i, j] = 0
            elif i > j:
                cosmat[i, j] = cosmat[j, i]
            else:
                cosmat[i, j] = cosine(parameter_matrix[i, :], parameter_matrix[j, :])
    return cosmat


def compute_max_update_norm(
    parameter_matrix: np.array
):
    return np.max(np.linalg.norm(parameter_matrix, axis=1))


def compute_mean_update_norm(
    parameter_matrix: np.array
):
    return np.linalg.norm(np.mean(parameter_matrix, axis=0))