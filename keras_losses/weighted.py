from tensorflow import keras
from tensorflow.keras import backend as K


__all__ = ['get_weighted_categorical_crossentropy', 'get_weighted_sparse_categorical_crossentropy']


def get_weighted_categorical_crossentropy(weights):
    r"""L = - \sum_i weights[i] y_true[i] \log(y_pred[i])

    :param weights: a list of weights for each class.

    :return: loss function.
    """
    weights = K.constant(weights, dtype=K.floatx())

    def _loss(y_true, y_pred):
        return K.squeeze(K.dot(K.cast(y_true, dtype=K.floatx()), K.expand_dims(weights)), axis=-1) \
            * keras.losses.categorical_crossentropy(y_true, y_pred)
    return _loss


def get_weighted_sparse_categorical_crossentropy(weights):
    r"""L = - \sum_i weights[i] y_true[i] \log(y_pred[i])

    :param weights: a list of weights for each class.

    :return: loss function.
    """
    weights = K.constant(weights, dtype=K.floatx())

    def _loss(y_true, y_pred):
        return K.squeeze(K.gather(weights, K.cast(y_true, 'int32')), axis=-1) \
            * keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
    return _loss
