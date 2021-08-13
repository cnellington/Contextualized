import numpy as np
import tensorflow as tf


class TaskSwitcher(tf.keras.layers.Layer):

    def __init__(self):
        self.archetypes = None



class ContextualCorrelator:
    """
    rho(c) = beta(c) * beta'(c)
    beta(c) = sigma(A @ f(c) + b)
    
    beta_{a_i, b_j} = sigma(A(t_i, t_j) @ f(c) + b)
    f(c) = sigma(dense(c))
    A(t_i, t_j) = <g(t_i, t_j), A_{1..K}>
    g(t_i, t_j) = softmax(dense(t_i, t_j))
    """

    def __init__(self):
        self.context_encoder = None
        self.task_switcher = TaskSwitcher()

