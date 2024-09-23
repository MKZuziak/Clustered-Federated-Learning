import unittest
import copy

import torch

from EFL.utils.computations import average_of_weigts
from tests.test_props.nets import NeuralNetwork


class Test_Average_of_Weights(unittest.TestCase):
    
        
    def test_weights_average(self):
        model_1 = NeuralNetwork()
        model_2 = NeuralNetwork()
        weights_1 = model_1.state_dict()
        weights_2 = model_2.state_dict()
        weights_1_copy = copy.deepcopy(weights_1)
        weights_2_copy = copy.deepcopy(weights_2)
        
        # Checking if the random initialization has not resulted in the same weights
        for (layer_weights_1, layer_weights_2) in zip(weights_1.values(), weights_2.values()):
            self.assertFalse(torch.allclose(layer_weights_1, layer_weights_2))
        # Checking for the correct copy
        for (layer_weights_1, layer_weights_1_copy) in zip(weights_1.values(), weights_1_copy.values()):
            self.assertTrue(torch.allclose(layer_weights_1, layer_weights_1_copy))
        for (layer_weights_2, layer_weights_2_copy) in zip(weights_2.values(), weights_2_copy.values()):
            self.assertTrue(torch.allclose(layer_weights_2, layer_weights_2_copy))
        
        # Weights dict of the same structure as in original code
        weights_dict = {
            0: weights_1,
            1: weights_2
        }
        avg_of_weights = average_of_weigts(weights_dict)
        
        # Check if the weights are intact after the operation
        for (layer_weights_1, layer_weights_1_copy) in zip(weights_1.values(), weights_1_copy.values()):
            self.assertTrue(torch.allclose(layer_weights_1, layer_weights_1_copy))
        for (layer_weights_2, layer_weights_2_copy) in zip(weights_2.values(), weights_2_copy.values()):
            self.assertTrue(torch.allclose(layer_weights_2, layer_weights_2_copy))
        
        # Check if the aggregated version is different
        for (layer_weights_1, layer_avg_weights) in zip(weights_1.values(), avg_of_weights.values()):
            self.assertFalse(torch.allclose(layer_weights_1, layer_avg_weights))
        for (layer_weights_2, layer_avg_weights) in zip(weights_2.values(), avg_of_weights.values()):
            self.assertFalse(torch.allclose(layer_weights_2, layer_avg_weights))


if __name__ == "__main__":
    unittest.main()