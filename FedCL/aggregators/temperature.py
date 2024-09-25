from math import sqrt

import numpy as np
from numpy.linalg import norm


def calculate_temperature(
    cosine_similarity_matrix:np.array,
    lamb:float = None
):
    if not lamb:
        lamb = sqrt((cosine_similarity_matrix.shape[0] * (cosine_similarity_matrix.shape[0] - 1) * 2 ** 2))
    return norm(cosine_similarity_matrix) / lamb