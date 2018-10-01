import unittest
import math
import tensorflow as tf
from keras_losses import get_weighted_categorical_crossentropy, get_weighted_sparse_categorical_crossentropy


class TestLoss(unittest.TestCase):

    def test_eval_sparse_categorical_crossentropy(self):
        y_pred = tf.convert_to_tensor([
            [0.1, 0.2, 0.5, 0.1, 0.1],
            [0.25, 0.1, 0.15, 0.2, 0.3],
        ])
        y_true = tf.convert_to_tensor([[3], [4]])
        weights = [0.8, 0.1, 0.2, 0.3, 0.4]
        weighted_loss = get_weighted_sparse_categorical_crossentropy(weights=weights)
        loss = weighted_loss(y_true, y_pred)
        with tf.Session() as sess:
            loss = sess.run(loss)
            self.assertAlmostEqual(
                - 0.3 * math.log(0.1),
                loss[0],
                places=6,
            )
            self.assertAlmostEqual(
                - 0.4 * math.log(0.3),
                loss[1],
                places=6,
            )

    def test_eval_categorical_crossentropy(self):
        y_pred = tf.convert_to_tensor([
            [0.1, 0.2, 0.5, 0.1, 0.1],
            [0.25, 0.1, 0.15, 0.2, 0.3],
        ])
        y_true = tf.convert_to_tensor([
            [0, 0, 1.0, 0, 0],
            [0, 1.0, 0, 0, 0],
        ])
        weights = [0.8, 0.1, 0.2, 0.3, 0.4]
        weighted_loss = get_weighted_categorical_crossentropy(weights=weights)
        loss = weighted_loss(y_true, y_pred)
        with tf.Session() as sess:
            loss = sess.run(loss)
            self.assertAlmostEqual(
                - 0.2 * math.log(0.5),
                loss[0],
                places=6,
            )
            self.assertAlmostEqual(
                - 0.1 * math.log(0.1),
                loss[1],
                places=6,
            )
