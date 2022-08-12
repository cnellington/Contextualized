import tensorflow as tf


# tf loss
def _DAG_loss(w_pred, alpha, rho, _from=""):
    d = tf.cast(tf.shape(w_pred)[1], w_pred.dtype)
    h = tf.linalg.trace(tf.linalg.expm(w_pred * w_pred)) - d
    return alpha * h + 0.5 * rho * h * h


def _mse_loss(x_true, w_pred):
    x_prime = tf.matmul(x_true, w_pred)
    return (0.5 / tf.cast(tf.shape(x_true)[0], w_pred.dtype)) * tf.square(
        tf.linalg.norm(x_true - x_prime)
    )


def _NOTEARS_loss(x_true, w_pred, l1_lambda, alpha, rho):
    mse = _mse_loss(x_true, w_pred)
    return mse + l1_lambda * tf.norm(w_pred, ord=1) + _DAG_loss(w_pred, alpha, rho)
