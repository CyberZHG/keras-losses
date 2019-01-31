import unittest
import tensorflow as tf
from keras_losses import coral_loss


class TestCoralLoss(unittest.TestCase):

    def test_eval(self):
        y_pred = tf.convert_to_tensor([
            [0.1, 0.2, 0.5, 2.0, 1.8],
            [0.2, 0.3, 0.15, 2.2, 1.1],
        ])
        y_true = tf.convert_to_tensor([
            [0.2, 0.1, 0.51, 2.1, 2.8],
            [0.5, 0.2, 0.25, 2.1, 0.1],
        ])
        loss = coral_loss(y_true, y_pred)
        self.assertAlmostEqual(0.11974798887968063, loss.numpy(), places=6)
