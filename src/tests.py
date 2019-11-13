import unittest
import numpy as np
from classes import *
from consts import *
# BATCH_SIZE = 4


class TestNN(unittest.TestCase):
    print("setup test case")
    val = 5
    actual_network = NetWork(val)
    actual_Layer = Layer(val)
    actual_Layer.net_values = np.arange(0, BATCH_SIZE*val).reshape(BATCH_SIZE, val)
    s_val = 10
    actual_S_Layer = SoftMaxLayer(s_val)
    actual_weight = Weights(np.ones((val + 1, s_val)), actual_Layer, actual_S_Layer)
    actual_Layer.lag_weights = actual_weight
    actual_S_Layer.lead_weights = actual_weight

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
        self.actual_Layer.normalize()
        self.assertEqual(0, np.average(self.actual_Layer.net_values))
        self.assertEqual(1, np.var(self.actual_Layer.net_values))

    def test_Layer_activate(self):
        self.actual_Layer.normalize()
        self.actual_Layer.activate()
        self.assertEqual(0, self.actual_Layer.activated_values.min())
        self.assertNotEqual(0, np.var(self.actual_Layer.activated_values))
        self.assertNotEqual(0, np.average(self.actual_Layer.activated_values))

    def test_Layer_calc_net_values(self):
        self.actual_Layer.normalize()
        self.actual_Layer.activate()
        self.actual_Layer.calc_net_values()
        expect = np.dot(self.actual_Layer.activated_values[0], self.actual_weight.weight.T[0])
        self.assertEqual(self.actual_S_Layer.net_values.size, BATCH_SIZE * self.s_val)
        self.assertEqual(self.actual_S_Layer.net_values[0][0], expect)

    def test_SoftMaxLayer_activate(self):
        self.actual_Layer.normalize()
        self.actual_Layer.activate()
        self.actual_Layer.calc_net_values()
        self.actual_S_Layer.normalize()
        self.actual_S_Layer.activate()
        self.assertLessEqual(self.actual_S_Layer.activated_values.max(), 1)
        self.assertGreaterEqual(self.actual_S_Layer.activated_values.min(), 0)

    def test_Layer_calc_delta(self):
        self.actual_Layer.normalize()
        self.actual_Layer.activate()
        self.actual_Layer.calc_net_values()
        self.actual_S_Layer.normalize()
        self.actual_S_Layer.activate()
        s_norm_diff = self.actual_S_Layer.calc_delta([3, 2, 1, 4])
        # norm_diff = self.actual_Layer.calc_delta([])
        expect_s00 = self.actual_S_Layer.activated_values[0][0] * s_norm_diff[0][0]
        expect_s12 = (self.actual_S_Layer.activated_values[1][2] - 1) * s_norm_diff[0][0]
        self.assertEqual(self.actual_S_Layer.delta[0][0], expect_s00)
        self.assertEqual(self.actual_S_Layer.delta[1][2], expect_s12)

    def test_Weight_calc_grad(self):
        self.actual_Layer.normalize()
        self.actual_Layer.activate()
        self.actual_Layer.calc_net_values()
        self.actual_S_Layer.normalize()
        self.actual_S_Layer.activate()
        self.actual_S_Layer.calc_delta([3, 2, 1, 4])
        self.actual_weight.calc_grad()
        lead = self.actual_Layer.activated_values.T[0]
        lag = self.actual_S_Layer.delta.T[0]
        round_w_00 = np.dot(lead, lag.T)/BATCH_SIZE
        self.assertEqual(self.actual_weight.gradients[0][0], round_w_00)
        #  self.actual_weight.sgd()
