from tensorflow.keras import backend as K


__all__ = ['coral_loss']


def covariance(x):
    n = K.cast(K.shape(x)[0], x.dtype)
    r = K.dot(K.transpose(x), x)
    c = K.sum(x, axis=0, keepdims=True)
    c = K.dot(K.transpose(c), c)
    return (r - c / n) / (n - 1 + K.epsilon())


def coral_loss(y_true, y_pred):
    r"""Deep CORAL (Correlation Alignment) loss.
    See: https://arxiv.org/abs/1607.01719

    loss = \frac{1}{4d^2} \left \| C_S - C_T \right \| _F^2

    C_S = \frac{1}{n_S - 1} (D_S^\top D_S - \frac{1}{n_S} (\mathbf{1}^\top D_S)^\top (\mathbf{1}^\top D_S))
    C_T = \frac{1}{n_T - 1} (D_T^\top D_T - \frac{1}{n_T} (\mathbf{1}^\top D_T)^\top (\mathbf{1}^\top D_T))
    """
    d = K.cast(K.shape(y_true)[-1], y_true.dtype)
    diff = covariance(y_true) - covariance(y_pred)
    return K.sum(diff * diff) / (4 * d * d)
