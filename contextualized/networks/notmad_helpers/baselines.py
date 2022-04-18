import copy
import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
import tensorflow_addons as tfa
from tqdm import tqdm
from tensorflow.keras.callbacks import Callback

from contextualized.networks.notmad_helpers import graph_utils
from contextualized.networks.notmad_helpers.graph_utils import project_to_dag, trim_params
from contextualized.networks.notmad_helpers.tf_utils import NOTEARS_loss, DAG_loss

# Keras NOTEARS version which takes in context and returns a single population model.

class DummyWeight(tf.keras.layers.Layer):
    """
    Placeholder for NOTEARS
    """
    def __init__(self, W_shape):
        super(DummyWeight, self).__init__()

        # Define the trainable networks
        self.W = self.add_weight("W", shape=W_shape,
            initializer=tf.keras.initializers.Constant(np.zeros(W_shape)))

    def build(self, input_shapes):
        pass

    def call(self, _):
        return self.W

# TODO: Sync this callback with the callback for CNOTEARS
class DynamicAlphaRho(Callback):
    """
    Dual-step DAG optimization, required for NO-TEARS structure learning
    """
    def __init__(self, C_train):
        super(DynamicAlphaRho, self).__init__()
        self.C_train = C_train
        self.h_old = 0.
        
    def on_epoch_begin(self, epoch, logs=None):
        pred = np.squeeze(self.model.predict(np.expand_dims(
            self.C_train[np.random.choice(self.C_train.shape[0])], 0)))
        #pred = trim_params(pred, thresh=0.1)
        my_dag_loss = DAG_loss(pred, self.model.alpha.numpy(), self.model.rho.numpy()) # TODO: should be measured over batch
        self.model.W.W.assign(self.model.W.W*(1-np.eye(self.model.W.W.shape[0]))) # set the diagonal to 0
        if my_dag_loss > 0.25*self.h_old:
            self.model.alpha.assign(self.model.alpha+self.model.rho*my_dag_loss)
            self.model.rho.assign(self.model.rho*10)
            #self.model.rho.assign(self.model.rho*1.1)
            # print(self.model.alpha.numpy(), self.model.rho.numpy())
        self.h_old = my_dag_loss

        
class NOTEARS:
    """
    Learn a DAG by minimizing NO-TEARS loss over a training set
    """
    def __init__(self, loss_params, context_shape, W_shape,
                 learning_rate=1e-3,
                 tf_dtype=tf.dtypes.float32):
#         super(NOTEARS, self).__init__()
        encoder_input_shape = (context_shape[1], 1)
        self.context = tf.keras.layers.Input(
            shape=encoder_input_shape, dtype=tf_dtype, name="C")
        self.W = DummyWeight(W_shape)
        #self.W = tf.Variable(initial_value=np.zeros(W_shape), trainable=True)
        self.outputs = self.W(self.context)
        self.model = tf.keras.models.Model(inputs=self.context, outputs=self.outputs)

        # Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = []
        try:
            self.model.alpha = tf.Variable(loss_params['init_alpha'], trainable=False)
            self.model.rho   = tf.Variable(loss_params['init_rho'], trainable=False)
            self.use_dynamic_alpha_rho = True
        except:
            self.model.alpha = loss_params['alpha']
            self.model.rho   = loss_params['rho']
            self.use_dynamic_alpha_rho = False
        self.model.W = self.W
        my_loss = lambda x,y: NOTEARS_loss(x, y,
                                           loss_params['l1'],
                                           self.model.alpha,
                                           self.model.rho)
        self.model.compile(loss=my_loss,
                     optimizer=self.opt,
                     metrics = self.metrics)

    def fit(self, C, X, epochs, batch_size, es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        callbacks = copy.deepcopy(callbacks)
        if es_patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience))
        if self.use_dynamic_alpha_rho:
            callbacks.append(DynamicAlphaRho(C))
        if verbose:
            bar = 'NOTEARS {l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}' 
            callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False, overall_bar_format=bar))
        self.model.fit(C, X, batch_size=batch_size, epochs=epochs,
            callbacks=callbacks, validation_split=val_split, verbose=0)

    def predict_w(self, C, project_to_dag=False):
        if project_to_dag:
            my_w = graph_utils.project_to_dag(self.get_w())[0]
            return np.array([my_w for _ in range(len(C))])
        else:
            return np.array([self.get_w() for _ in range(len(C))])

    def set_w(self, W):
        self.W.W.assign(W)

    def get_w(self):
        return self.W.W.numpy()


class ClusteredNOTEARS:
    """
    Learn several NO-TEARS optimized DAGs based on a clustering function
    Clusters can be pre-defined or learned here using context data
    """
    def __init__(self, n_clusters, loss_params, context_shape, W_shape,
                 learning_rate=1e-3, clusterer=None, clusterer_fitted=False,
                 tf_dtype=tf.dtypes.float32):
#         super(ClusteredNOTEARS, self).__init__()
        if clusterer is None:
            self.clusterer = KMeans(n_clusters=n_clusters)
        else:
            self.clusterer = clusterer # Must have a predict() function
        self.clusterer_fitted = clusterer_fitted
        self.notears_models = [NOTEARS(loss_params, context_shape, W_shape, learning_rate) for i in range(n_clusters)]

    def fit(self, C, X, epochs, batch_size, es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        if len(C.shape) > 2:
            C = C.squeeze()
        if not self.clusterer_fitted:
            self.clusterer.fit(C)
            self.clusterer_fitted = True
        train_labels = self.clusterer.predict(C)
        loop = list(set(train_labels))
        if verbose:
            loop = tqdm(loop, desc='Clustered NOTEARS Training')
        for clust in loop:
            ct_idxs = train_labels == clust
            if np.sum(ct_idxs) < 2:
                self.notears_models[clust].set_w(np.zeros_like(self.notears_models[clust].get_w()))
            else:
                C_ct = C[ct_idxs]
                X_ct = X[ct_idxs]
                self.notears_models[clust].fit(C_ct, X_ct, epochs, batch_size, es_patience=es_patience, val_split=val_split, callbacks=callbacks, verbose=0)
                self.notears_models[clust].set_w(project_to_dag(self.notears_models[clust].get_w())[0])
            
    def predict_w(self, C, project_to_dag=False):
        # Already projected to DAG space, nothing to do here.
        if len(C.shape) > 2:
            C = C.squeeze()
        test_labels  = self.clusterer.predict(C)
        return np.array([self.notears_models[label].get_w() for label in test_labels])
    
    def get_ws(self, project_to_dag=False):
        # Already projected to DAG space, nothing to do here.
        return np.array([model.get_w() for model in self.notears_models])
