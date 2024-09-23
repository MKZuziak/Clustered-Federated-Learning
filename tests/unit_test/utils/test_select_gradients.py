import unittest
import copy

import torch

from EFL.utils.select_gradients import select_gradients
from tests.test_props.nets import NeuralNetwork


class Test_Select_Gradients(unittest.TestCase):

        
    def test_gradient_selection(self):
        model_0 = NeuralNetwork()
        model_1 = NeuralNetwork()
        model_5 = NeuralNetwork()
        weights_0 = model_0.state_dict()
        weights_1 = model_1.state_dict()
        weights_5 = model_5.state_dict()
                
        # Weights dict of the same structure as in original code
        weights_dict = {
            0: weights_0,
            1: weights_1,
            5: weights_5
        }
        
        # Select client no. 1
        select_0 = select_gradients(
            nodes_id_list = [0],
            gradients = weights_dict
            )
        self.assertEqual(len(select_0), 1)
        self.assertTrue(select_0[0])
        for (layer_weights_0, layer_weights_select) in zip(weights_0.values(), select_0[0].values()):
            self.assertTrue(torch.allclose(layer_weights_0, layer_weights_select))
        
        # Select client no. 5
        select_5 = select_gradients(
            nodes_id_list = [5],
            gradients = weights_dict
            )
        self.assertEqual(len(select_5), 1)
        self.assertTrue(select_5[5])
        for (layer_weights_5, layer_weights_select) in zip(weights_5.values(), select_5[5].values()):
            self.assertTrue(torch.allclose(layer_weights_5, layer_weights_select))
        
        # Select client no. 0, 5
        select_0_5 = select_gradients(
            nodes_id_list = [0, 5],
            gradients = weights_dict
            )
        self.assertEqual(len(select_0_5), 2)
        self.assertTrue(select_0_5[0])
        self.assertTrue(select_0_5[5])
        for (layer_weights_0, layer_weights_select) in zip(weights_0.values(), select_0_5[0].values()):
            self.assertTrue(torch.allclose(layer_weights_0, layer_weights_select))
        for (layer_weights_5, layer_weights_select) in zip(weights_5.values(), select_0_5[5].values()):
            self.assertTrue(torch.allclose(layer_weights_5, layer_weights_select))
        
        # Select all -> client no. 0, 1, 5
        select_0_1_5 = select_gradients(
            nodes_id_list = [0, 1, 5],
            gradients = weights_dict
            )
        self.assertEqual(len(select_0_1_5), 3)
        self.assertTrue(select_0_1_5[0])
        self.assertTrue(select_0_1_5[1])
        self.assertTrue(select_0_1_5[5])
        for (layer_weights_0, layer_weights_select) in zip(weights_0.values(), select_0_1_5[0].values()):
            self.assertTrue(torch.allclose(layer_weights_0, layer_weights_select))
        for (layer_weights_1, layer_weights_select) in zip(weights_1.values(), select_0_1_5[1].values()):
            self.assertTrue(torch.allclose(layer_weights_1, layer_weights_select))
        for (layer_weights_5, layer_weights_select) in zip(weights_5.values(), select_0_1_5[5].values()):
            self.assertTrue(torch.allclose(layer_weights_5, layer_weights_select))
        
        # Do not select any client -> client no. 10, 12
        select_none = select_gradients(
            nodes_id_list = [10, 12],
            gradients = weights_dict
        )
        self.assertEqual(len(select_none), 0)


if __name__ == "__main__":
    unittest.main()