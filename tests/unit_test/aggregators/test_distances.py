import unittest

import numpy as np
from scipy.spatial.distance import cosine
from numpy import dot
from numpy.linalg import norm

from EFL.aggregators.distances import calculate_cosine_similarity


class Test_Distance(unittest.TestCase):
    
    def test_cosine_similarity(self):
        ## FIRST TEST
        # Generating random parameters
        parameter_matrix = np.zeros(shape=(5, 10))
        for i in range(5):
            parameter_matrix[i, :] = np.random.rand(1, 10)
        cosine_distance = calculate_cosine_similarity(parameter_matrix=parameter_matrix)
        
        # Primitive way of calculating cosine similarity
        cosine_distance_prim = np.zeros(shape=(5, 5))
        for i in range(5):
            for j in range(5):
                cosine_distance_prim[i, j] = dot(parameter_matrix[i, :], parameter_matrix[j, :]) / ((norm(parameter_matrix[i, :])) * norm(parameter_matrix[j, :]))
        self.assertTrue(np.allclose(cosine_distance, cosine_distance_prim))
        
        ## SECOND TEST
        # Generating random parameters
        parameter_matrix = np.zeros(shape=(20, 50))
        for i in range(20):
            parameter_matrix[i, :] = np.random.rand(1, 50)
        cosine_distance = calculate_cosine_similarity(parameter_matrix=parameter_matrix)
        
        # Primitive way of calculating cosine similarity
        cosine_distance_prim = np.zeros(shape=(20, 20))
        for i in range(20):
            for j in range(20):
                cosine_distance_prim[i, j] = dot(parameter_matrix[i, :], parameter_matrix[j, :]) / ((norm(parameter_matrix[i, :])) * norm(parameter_matrix[j, :]))
        self.assertTrue(np.allclose(cosine_distance, cosine_distance_prim))

if __name__ == "__main__":
    unittest.main()