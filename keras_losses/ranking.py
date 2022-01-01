import tensorflow as tf
from tensorflow.keras import backend as K


__all__ = ['get_ranking_loss']


def get_ranking_loss(gamma=2.0, mp=2.5, mn=0.5):
    """L = log(1 + exp(gamma * (mp - y_pred[true_label]))) +
           log(1 + exp(gamma * (mn + y_pred[false_label_with_max_score])))

    :param gamma: scaling factor.
    :param mp: positive margin.
    :param mn: negative margin.

    :return: loss function.
    """

    def _loss(y_true, y_pred):
        def _loss_elem(i):
            scores, pos_label = y_pred[i], tf.cast(y_true[i][0], dtype='int32')
            pos_score = scores[pos_label]
            top_values, top_indices = tf.nn.top_k(scores, k=2)
            neg_score = tf.cond(tf.equal(top_indices[0], pos_label), lambda: top_values[1], lambda: top_values[0])
            return K.log(1.0 + tf.exp(gamma * (mp - pos_score))) + K.log(1.0 + tf.exp(gamma * (mn + neg_score)))
        return tf.map_fn(_loss_elem, tf.range(tf.shape(y_true)[0]), dtype=K.floatx())
    return _loss
