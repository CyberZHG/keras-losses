import unittest
import math
import tensorflow as tf
from keras_losses import get_ranking_loss


class TestRankingLoss(unittest.TestCase):

    def test_eval(self):
        y_pred = tf.convert_to_tensor([
            [0.1, 0.2, 0.5, 2.0, 1.8],
            [0.2, 0.3, 0.15, 2.2, 1.1],
        ])
        y_true = tf.convert_to_tensor([[3], [4]])
        ranking_loss = get_ranking_loss(gamma=2.0, mp=2.5, mn=0.5)
        loss = ranking_loss(y_true, y_pred)
        self.assertAlmostEqual(
            math.log(1.0 + math.exp(2.0 * (2.5 - 2.0))) + math.log(1.0 + math.exp(2.0 * (0.5 + 1.8))),
            loss[0].numpy(),
            places=6,
        )
        self.assertAlmostEqual(
            math.log(1.0 + math.exp(2.0 * (2.5 - 1.1))) + math.log(1.0 + math.exp(2.0 * (0.5 + 2.2))),
            loss[1].numpy(),
            places=6,
        )
