import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import Callback
import igraph as ig

from contextualized.dags.notmad_helpers.tf_utils import NOTEARS_loss, DAG_loss
from contextualized.dags.notmad_helpers import graph_utils


class NGAM(tf.keras.layers.Layer):
    """
    Neural Generalized Additive Model
    One multi-layer perceptron per input dimension for an interpretable non-linear model
    """
    def __init__(self, input_width, output_width, depth, width, activation='swish',
                final_activation='linear', boolean_feats=None):
        super(NGAM, self).__init__()
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
        #y = Lambda(lambda x: x[:,0,:,:], output_shape=(1,) + input_shape[2:])(x)
        outputs = [self.models[i](
            tf.keras.layers.Lambda(
                lambda x: x[:, i],
                output_shape=(None, self.output_width))(my_input))
                   for i in range(self.input_width)]
        return outputs
    

class DynamicAlphaRho(Callback):
    """
    Dual-step DAG optimization parameter update, required for NO-TEARS structure learning
    """
    def __init__(self, C_train, base_predictor=None, tol=0.25):
        super(DynamicAlphaRho, self).__init__()
        self.C_train = C_train
        self.h_old = 0.
        self.tol = tol
        self.base_predictor = base_predictor

    def on_epoch_begin(self, epoch, logs=None):
        if self.base_predictor is not None:
            base_W = self.base_predictor.predict_w(self.C_train, project_to_dag=False)
        else:
            base_W = np.zeros((len(self.C_train), 1))
        pred = np.squeeze(self.model.predict(
                {"C": self.C_train,
                 "base_W": base_W}))
#         if self.base_predictor is not None:
#             pred = np.squeeze(self.model.predict(
#                 {"C": self.C_train,
#                  "base_W": self.base_predictor.predict_w(self.C_train, project_to_dag=False)}))
#         else:
#             pred = np.squeeze(self.model.predict(
#                 {"C": self.C_train,
#                  "base_W": np.zeros((len(self.C_train), 1))}))
        #pred = np.squeeze(self.model.predict(np.expand_dims(
        #    self.C_train[np.random.choice(self.C_train.shape[0])], 0)))
        #pred = trim_params(pred, thresh=0.1)
        #if not is_dag(pred):
        my_dag_loss = tf.reduce_mean(DAG_loss(pred, self.model.alpha.numpy(), self.model.rho.numpy()))
        if my_dag_loss > self.tol*self.h_old:
            self.model.alpha.assign(self.model.alpha+self.model.rho*my_dag_loss)
            self.model.rho.assign(self.model.rho*1.1)
            # print(self.model.alpha.numpy(), self.model.rho.numpy())
        self.h_old = my_dag_loss

class ProjectToDAG(Callback):
    """
    Project low and rull-rank archetypes in NOTMAD to DAG
    """
    def __init__(self, distance=0.1):
        super(ProjectToDAG, self).__init__()
        self.distance = distance

    def project_low_rank(self, A, B, mask):
        A_new = np.zeros_like(A)
        B_new = np.zeros_like(B)
        pca = PCA(n_components=A.shape[-1])
        # print(A.shape, B.shape)
        arches = np.vstack(([graph_utils.project_to_dag(a.dot(np.squeeze(B)))[0] for a in A]))
        a = pca.fit_transform(arches)
        b = pca.components_
        A_new = A + self.distance(a - A)
        B_new = B + self.distance(b - B)
        """
        for i in range(len(A)):
            arch = np.multiply(A[i].dot(B[i]), mask)
            arch_dag, thresh = graph_utils.project_to_dag(arch)
            a = pca.fit_transform(arch_dag)
            b = pca.components_
            A_new[i] = A[i] + self.distance*(a - A[i])
            B_new[i] = B[i] + self.distance*(b - B[i])
        """
        return A_new, B_new
    
    def project_full_rank(self, archs):
        archs_new = np.zeros_like(archs)
        for i in range(len(archs)):
            arch_dag, thresh = graph_utils.project_to_dag(archs[i])
            archs_new[i] = archs[i] + self.distance*(arch_dag - archs[i])
        return archs_new
    
    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'archetypes'):
                explainer = layer
        try:
            A_new, B_new = self.project_low_rank(explainer.A.numpy(), explainer.B.numpy(), explainer.mask.numpy())
            explainer.A.assign(A_new)
            explainer.B.assign(B_new)
        except: # A, B do not exist for full-rank version
            explainer.archs.assign(self.project_full_rank(explainer.archs.numpy()))


class UpdateCompatMat(Callback):
    """
    Binary compatibility matrix to enforce DAG-only archetype mixtures in NOTMAD
    """
    def __init__(self):
        super(UpdateCompatMat, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, 'archetypes'):
                explainer = layer
        #explainer = self.model.layers[-4] # TODO: How to automate this count?
        archs = explainer.archetypes().numpy()
        compat_mat = np.eye(explainer.k)
        for i in range(explainer.k):
            for j in range(i):
                #for k in range(j):
                if graph_utils.is_dag(archs[i] + archs[j]): #graph_utils.is_dag(graph_utils.trim_params(archs[i] + archs[j], thresh=0.01)):
                    compat_mat[i,j] = 1    
                    compat_mat[j,i] = 1
        explainer.compatibility_mat.assign(compat_mat)

            
class BatchDot(tf.keras.layers.Layer):
    """
    Matrix multiplication over a batch
    """
    def __init__(self):
        super(BatchDot, self).__init__()

    def build(self, input_shapes):
        pass

    def call(self, A, B):
        return tf.keras.backend.batch_dot(A, B)
        
class Explainer(tf.keras.layers.Layer):
    """
    NOTMAD dictionary of archetypes
    Forward pass takes a weight vector and returns a weighted combination of archetypes
    """
    def __init__(self, archetype_shape, tf_dtype, init_mat=None, init_archs=None,
                 freeze_archs=False, init_compat=None, freeze_compat=False,
                 use_compat=False, update_compat_by_grad=True):
        # archetype_shape is (number of entries, number of features, rank)
        super(Explainer, self).__init__()
        self.k, self.d1, self.d2 = archetype_shape
        self.mask = tf.Variable(1 - np.eye(self.d1).astype(np.float32), trainable=False)
        self.use_compat = use_compat
        self.update_compat_by_grad = update_compat_by_grad
        if self.use_compat:
            if init_compat is None:
                init_compat = np.eye(self.k), # initialize to only picking 1 archetype
            self.compatibility_mat = self.add_weight("compat", shape=(self.k, self.k),
                                                     initializer=tf.keras.initializers.Constant(init_compat), 
                                                     trainable=update_compat_by_grad and not freeze_compat)
        trainable = not freeze_archs
        if self.d2 == self.d1: # Full-Rank
            if init_mat is None:
                init_mat = np.zeros(archetype_shape)
            self.archs = self.add_weight("archs", shape=archetype_shape,
                                              initializer=tf.keras.initializers.Constant(init_mat),
                                              trainable=trainable)
        elif self.d2 < self.d1: # Low-Rank
            A_shape = (self.k, self.d1, self.d2)
            B_shape = (1, self.d2, self.d1)#(self.k, self.d2, self.d1)
            if init_archs is not None:
                init_A = init_archs["A"]
                init_B = init_archs["B"]
            elif init_mat is not None:
                pca = PCA(n_components=self.d2)
                init_A = pca.fit_transform(init_mat)
                init_B = np.array(pca.components_)
                init_A = np.array([init_A + np.random.binomial(0, 0.1, size=init_A.shape)*np.random.normal(0, 0.1, size=init_A.shape)
                    for _ in range(self.k)])
                #init_B = np.array([init_B + np.random.binomial(0, 0.1, size=init_B.shape)*np.random.normal(0, 0.1, size=init_B.shape)
                #    for _ in range(self.k)])
            else:
                # Default init, set A to all 0s and and B to all 1s.
                init_A = np.zeros(A_shape)
                init_B = np.ones(B_shape)
            self.A = self.add_weight("A", shape=A_shape,
                                     initializer=tf.keras.initializers.Constant(init_A),
                                     trainable=trainable)
            self.B = self.add_weight("B", shape=B_shape,
                                     initializer=tf.keras.initializers.Constant(init_B),
                                     trainable=trainable)
        else:
            print("Error. Explainer rank cannot be greater than data dimension. Was provided rank {} for data dimension {}.".format(self.d2, self.d1))
            return None

    def build(self, input_shapes):
        pass

    def archetypes(self):
        try:
            archs = tf.tensordot(self.A, tf.squeeze(self.B), axes=1)#tf.keras.backend.batch_dot(self.A, self.B)
        except:
            archs = self.archs
        return tf.multiply(archs, self.mask)

    def call(self, subtype): # Subtype is of shape (None x k)
        if self.use_compat:
            max_subtype = 1 + tf.math.sign(subtype - tf.reduce_max(subtype, axis=0))
            subtype = tf.multiply(tf.tensordot(max_subtype, self.compatibility_mat, axes=1), subtype)
        return tf.tensordot(subtype, self.archetypes(), axes=1)


class NOTMAD: # TODO: Only difference between low-rank and full-rank version should be the Explainer.
    """
    NO-TEARS Optimized Mixtures of Archetypal DAGs
    The sample-specific DAG estimation model
    """
    def __init__(self, context_shape, data_shape, n_archetypes,
                 sample_specific_loss_params, archetype_loss_params,
                 n_encoder_layers=1, encoder_width=16,
                 context_activity_regularizer=tf.keras.regularizers.l1(1e-3),
                 activation='relu', rank=None,
                 init_mat=None, init_archs=None, freeze_archs=False,
                 learning_rate=1e-3, project_archs_to_dag=True,
                 project_distance=0.1,
                 tf_dtype=tf.dtypes.float32,
                 use_compatibility=False,
                 update_compat_by_grad=True,
                 pop_model=None,
                 base_predictor=None,
                 encoder_type='gam',
                 init_compat=None,
                 freeze_compat=False
                ):
        super(NOTMAD, self).__init__()
        encoder_input_shape = (context_shape[1], 1)
        encoder_output_shape = (n_archetypes, )
        if rank is None:
            rank = data_shape[-1]
        dict_shape = (n_archetypes, data_shape[-1], rank)
        self.use_compatibility = use_compatibility
        self.update_compat_by_grad = update_compat_by_grad
        self.pop_model = pop_model
        self.base_predictor = base_predictor
        self.init_compat = init_compat
        self.freeze_compat = freeze_compat
    
        # Encoder: context --> subtype
        self.context = tf.keras.layers.Input(
            shape=encoder_input_shape, dtype=tf_dtype, name="C")
        
        C_flat = tf.keras.layers.Flatten()(self.context)
        if encoder_type == 'gam':
            self.encoder_gam = NGAM(encoder_input_shape[0], encoder_output_shape[0],
                                   depth=n_encoder_layers, width=encoder_width,
                                   activation=activation)(C_flat)
            self.encoder_out = tf.nn.softmax(tf.reduce_sum(self.encoder_gam, axis=0), axis=-1)
            self.encoder = tf.keras.models.Model(inputs=self.context, outputs=self.encoder_out)
        else:
            self.encoder = self._build_encoder(n_encoder_layers, encoder_width, 
                                          encoder_output_shape,
                                          context_activity_regularizer,
                                          activation)
        self.encodings = self.encoder(self.context)
        self.project_archs_to_dag = project_archs_to_dag
        self.project_distance = project_distance
        # TODO: Pre-train encoder?

        self.explainer = Explainer(dict_shape, tf_dtype, init_mat, init_archs, freeze_archs,
                                   init_compat=self.init_compat, freeze_compat=self.freeze_compat,
                                   use_compat=use_compatibility, 
                                   update_compat_by_grad=update_compat_by_grad)
        self.outputs = self.explainer(self.encodings)
        if self.pop_model is not None:
            self.outputs += self.pop_model
        self.base_W  = tf.keras.layers.Input(
                shape=(data_shape[-1], data_shape[-1]), dtype=tf_dtype, name="base_W")
        if self.base_predictor is not None:
            self.outputs += self.base_W
        self.model = tf.keras.models.Model(inputs=(self.context, self.base_W),
                                           outputs=self.outputs)
        self.model.explainer = self.explainer

        # Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = []
        self.model.alpha, self.model.rho, self.use_dynamic_alpha_rho = self._parse_alpha_rho(sample_specific_loss_params)
        
        self.my_loss = lambda x,y: NOTEARS_loss(x, y,
                                           sample_specific_loss_params['l1'],
                                           self.model.alpha,
                                           self.model.rho) \
            + self._build_arch_loss(archetype_loss_params) # TODO: Uses static alpha, rho for archetypes.
        self.model.compile(loss=self.my_loss,
                     optimizer=self.opt,
                     metrics = self.metrics)

    def _parse_alpha_rho(self, params):
        try:
            alpha = tf.Variable(params['init_alpha'], trainable=False)
            rho = tf.Variable(params['init_rho'], trainable=False)
            dynamic = True
        except:
            alpha = params['alpha']
            rho = params['rho']
            dynamic = False
        return alpha, rho, dynamic
    
    
    def _build_encoder(self, n_layers, width, output_shape,
                       context_activity_regularizer, activation='relu'):
        encoder_layers = [tf.keras.layers.Flatten()]
        for i in range(n_layers-1):
            encoder_layers.append(
                tf.keras.layers.Dense(width,
                                      activation=activation,
                                      trainable=True))
            encoder_layers.append(
                tf.keras.layers.BatchNormalization())
        encoder_layers.append(
            tf.keras.layers.Dense(np.prod(output_shape),
                activation='softmax',
                activity_regularizer=context_activity_regularizer,
                trainable=True)
        )
        return tf.keras.models.Sequential(encoder_layers)
        
        
    def _build_arch_loss(self, params):
        try: # low-rank archetype loss, no good way to penalize dag-ness
            return params['l1']*tf.norm(self.explainer.A, ord=1) \
                 + params['l1']*tf.norm(self.explainer.B, ord=1)
        except: # full-rank, directly penalize dag-ness of archetypes
            return tf.reduce_sum([
                params['l1']*tf.norm(self.explainer.archs[i], ord=1) + \
                DAG_loss(self.explainer.archs[i], alpha=params['alpha'],
                     rho=params['rho'])
                for i in range(self.explainer.k)
            ])
    
    def transform_to_low_rank(self, W):
        # W is N x P x P
        return np.tensordot(W, self.explainer.B.T, axes=1)
                     
    def fit(self, C, X, epochs, batch_size,
            es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        callbacks = copy.deepcopy(callbacks)
        if es_patience is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience))
        if self.use_dynamic_alpha_rho:
            callbacks.append(DynamicAlphaRho(C, self.base_predictor))
        if self.project_archs_to_dag:
            callbacks.append(ProjectToDAG(distance=self.project_distance))
        if self.use_compatibility and not self.update_compat_by_grad and not self.freeze_compat:
            callbacks.append(UpdateCompatMat())
        if verbose:
            bar = 'NOTMAD {l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}' 
            callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False, overall_bar_format=bar))
        if self.base_predictor is not None:
            base_W = self.base_predictor.predict_w(C, project_to_dag=False)
            #if self.explainer.d2 < self.explainer.d1: # low-rank
            #    if base_W.shape[-1] == base_W.shape[-2]: # not low-rank
            #        base_W = self.transform_to_low_rank(base_W)
        else:
            base_W = np.zeros((len(C), X.shape[-1], X.shape[-1])) # TODO: this is expensive.
        self.model.fit({"C":C, "base_W": base_W},
                       y=X, batch_size=batch_size, epochs=epochs,
            callbacks=callbacks, validation_split=val_split, verbose=0)

    def progressive_fit(self, C, X, epochs_per_grouping, groupings, max_samples=100, verbose=1):
        """
        Learn personalized networks by learning group networks with increasing
        specificity. This encourages the framework to learn population networks
        before personalizing these networks (e.g. tissue --> cell-type --> cell)

        C: Context variables (n, c)
        X: Attribute variables (n, m)
        epochs_per_grouping: Number of epochs to spend learning group networks for
                              individuals
        groupings: list of population groupings (g, n) for C, X where the total
                    number of labels per grouping can vary

        TODO: Gradient computation is unreasonably slow for large groups (or
        conversely, small # of groups e.g. male/female) because individual
        gradients are computed using the all group samples.
        """
        for i, grouping in enumerate(groupings):
            if verbose: print(f"Grouping {i+1}/{len(groupings)}")
            labels = np.unique(grouping)

            if len(labels) == len(grouping):  # Normal fit for individual grouping
                self.model.fit(C, X, batch_size=1, epochs=epochs_per_grouping, verbose=verbose)
                continue

            for epoch in range(epochs_per_grouping):  # Group fit otherwise
                for j, label in enumerate(labels):
                    if verbose:
                        print(f"Epoch {epoch+1} Group {j+1}/{len(labels)}")
                    group_idx = grouping == label
                    C_group = C[group_idx]
                    X_group = X[group_idx]
                    sample_idx = np.random.choice(np.arange(len(C_group)), size=max_samples)
                    C_sample = C_group[sample_idx]
                    X_sample = X_group[sample_idx]
                    # Training individuals in the same group to produce
                    # identical DAGs for group expression data
                    with tf.GradientTape(persistent=True) as tape:
                        w_preds = self.model(C_sample)
                        losses = [self.my_loss(X_sample, w_pred) for w_pred in w_preds]
                    grads = tape.gradient(losses, self.model.trainable_variables)
                    self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict_w(self, C, project_to_dag=False):
        if self.base_predictor is not None:
            base_W = self.base_predictor.predict_w(C, project_to_dag=False)
            #if self.explainer.d2 < self.explainer.d1: # low-rank
            #    if base_W.shape[-1] == base_W.shape[-2]: # not low-rank
            #        base_W = self.transform_to_low_rank(base_W)
        else:
            base_W = np.zeros((len(C), 1))
        preds = self.model.predict({"C": C, "base_W": base_W})
        if project_to_dag:
            return np.array([graph_utils.project_to_dag(w)[0] for w in preds])
        else:
            return preds
