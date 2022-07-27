import numpy as np
import tensorflow as tf

class _NGAM(tf.keras.layers.Layer):
    """
    Neural Generalized Additive Model
    One multi-layer perceptron per input dimension for an interpretable non-linear model
    """
    def __init__(self, input_width, output_width, depth, width, activation='swish',
                final_activation='linear', boolean_feats=None):
        super(_NGAM, self).__init__()
        self.models = []
        self.input_width = input_width
        self.output_width = output_width
        for i in range(input_width):
            my_layers = [tf.keras.layers.Flatten()]
            if boolean_feats is None or boolean_feats[i] == False: # if boolean feature, don't need hidden layers
                my_layers.extend([tf.keras.layers.Dense(
                    width, activation=activation,
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)) for _ in range(depth)])
            my_layers.append(tf.keras.layers.Dense(
                output_width, activation=final_activation,
                kernel_initializer=tf.keras.initializers.RandomUniform(minval=-0.01, maxval=-0.01)))
            self.models.append(tf.keras.models.Sequential(my_layers))
        self._name = "NGAM"
            
    def build(self, input_shapes):
        pass
    
    def call(self, my_input):
        outputs = [self.models[i](
            tf.keras.layers.Lambda(
                lambda x: x[:, i],
                output_shape=(None, self.output_width))(my_input))
                for i in range(self.input_width)]
        return outputs

class _Explainer(tf.keras.layers.Layer):
    """
    NOTMAD dictionary of archetypes
    Forward pass takes a weight vector and returns a weighted combination of archetypes
    """
    def __init__(self, archetype_shape, tf_dtype, init_mat=None, init_archs=None,
                 freeze_archs=False, init_compat=None, freeze_compat=False,
                 use_compat=False, update_compat_by_grad=True):
        # archetype_shape is (number of entries, number of features, rank)
        super(_Explainer, self).__init__()
        self.k, self.d1, self.d2 = archetype_shape
        self.mask = tf.Variable(1 - np.eye(self.d1).astype(np.float32), trainable=False)
        self.use_compat = use_compat
        self.update_compat_by_grad = False

        #set the archetypes
        if self.d2 == self.d1: # Full-Rank
            if init_mat is None:
                init_mat = np.random.uniform(-0.01, 0.01, size=(self.k, self.d1,self.d1)) 
                # init_mat = np.zeros(archetype_shape)
            self.archs = self.add_weight("archs", shape=archetype_shape,
                                              initializer=tf.keras.initializers.Constant(init_mat),
                                              trainable=True)

        else:
            print("Error. Explainer rank cannot be greater than data dimension. Was provided rank {} for data dimension {}.".format(self.d2, self.d1))
            return None

    def build(self, input_shapes):
        pass

    def archetypes(self):
        archs = self.archs
        return tf.multiply(archs, self.mask)

    def call(self, subtype): # Subtype is of shape (None x k) = weights
        mult = tf.tensordot(subtype, self.archetypes(), axes=1)
        return mult