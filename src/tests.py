import unittest
from classes import Layer, Weights, NetWork
from consts import *


class TestNN(unittest.TestCase):
    print("setup test case")
    val = 5
    actual_network = NetWork(val)
    actual_Layer = Layer(val)


    def test_Layer_constructor(self):
        self.assertEqual(self.val, self.actual_Layer.node_amount)

    def test_NetWork_constructor(self):
        ex = self.val
        self.assertEqual(ex, self.actual_network.hidden_layer)
        self.assertEqual(ex + 2, len(self.actual_network.layers))
        self.assertEqual(None, self.actual_network.layers[0].lead_weights)
        self.assertEqual((DIM + 1) * MD1, self.actual_network.layers[0].lag_weights.weight.size)
        self.assertEqual((MD2 + 1) * CLASS_NUM, self.actual_network.layers[-1].lead_weights.weight.size)
        self.assertEqual(None, self.actual_network.layers[-1].lag_weights)
        self.assertEqual(self.actual_network.layers[1].lag_weights, self.actual_network.layers[2].lead_weights)
        self.assertEqual(self.actual_network.layers[2].lead_weights.lead_layer, self.actual_network.layers[1])

    def test_Layer_norm(self):
        pass

    def test_Layer_activate(self):
        pass

    def test_SoftMaxLayer_activate(self):
        pass

    def test_Layer_calc_net_values(self):
        pass

    def test_Layer_lag_prop(self):
        pass

    def test_Layer_calc_delta(self):
        pass

    def test_SoftMaxLayer_calc_delta(self):
        pass

