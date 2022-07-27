#tf notmad class
import tensorflow as tf
import tensorflow_addons as tfa
import copy

from test_notmad.tf_layers import _NGAM, _Explainer
from test_notmad.tf_loss import _NOTEARS_loss, _DAG_loss
from test_notmad.tf_callbacks import _DynamicAlphaRho

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
        self.use_compatibility = False
        self.update_compat_by_grad = update_compat_by_grad
        self.pop_model = pop_model
        self.init_compat = init_compat
        self.freeze_compat = freeze_compat
    
        # Encoder: context --> subtype
        self.context = tf.keras.layers.Input(
            shape=encoder_input_shape, dtype=tf_dtype, name="C")
        
        C_flat = tf.keras.layers.Flatten()(self.context)

        if encoder_type == 'gam':
            self.encoder_gam = _NGAM(encoder_input_shape[0], encoder_output_shape[0],
                                   depth=n_encoder_layers, width=encoder_width,
                                   activation=activation)(C_flat)
            self.encoder_out = tf.nn.softmax(tf.reduce_sum(self.encoder_gam, axis=0), axis=-1)
            self.encoder = tf.keras.models.Model(inputs=self.context, outputs=self.encoder_out)
        else:
            self.encoder = self._build_encoder(n_encoder_layers, encoder_width, 
                                          encoder_output_shape,
                                          context_activity_regularizer,
                                          activation)
        
        # #replace with linear layer
        # self.encoder = layers.Dense(encoder_output_shape[0], activation='relu')
        self.encodings = self.encoder(self.context)
        
        
        self.project_archs_to_dag = project_archs_to_dag
        self.project_distance = project_distance
        
        # TODO: Pre-train encoder?
        self.explainer = _Explainer(dict_shape, tf_dtype, init_mat, init_archs, freeze_archs,
                                   init_compat=self.init_compat, freeze_compat=self.freeze_compat,
                                   use_compat=use_compatibility, 
                                   update_compat_by_grad=update_compat_by_grad)
        self.outputs = self.explainer(self.encodings)
        
        if self.pop_model is not None:
            self.outputs += self.pop_model
        
        
        self.model = tf.keras.models.Model(inputs=(self.context),#, self.base_W),
                                           outputs=self.outputs)
        self.model.explainer = self.explainer



        # Compile the model
        self.opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.metrics = []
        self.model.alpha, self.model.rho, self.use_dynamic_alpha_rho = self._parse_alpha_rho(sample_specific_loss_params)
        
        self.my_loss = lambda x,y: _NOTEARS_loss(x, y,
                                           sample_specific_loss_params['l1'],
                                           self.model.alpha,
                                           self.model.rho) \
            + self._build_arch_loss(archetype_loss_params) # TODO: Uses static alpha, rho for archetypes.
        # self.my_loss = lambda x,y: self._build_arch_loss(archetype_loss_params)
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
                _DAG_loss(self.explainer.archs[i], alpha=params['alpha'],
                     rho=params['rho'])
                for i in range(self.explainer.k)
            ])
        
    def fit(self, C, X, epochs, batch_size,
            es_patience=None, val_split=0.25, callbacks=[], verbose=1):
        callbacks = copy.deepcopy(callbacks)
        # callbacks.append(PlotLearning())
        # if es_patience is not None:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=es_patience))
        # if self.use_dynamic_alpha_rho:
        callbacks.append(_DynamicAlphaRho(C))
        
        if verbose:
            bar = 'NOTMAD {l_bar}{bar} {n_fmt}/{total_fmt} ETA: {remaining}s,  {rate_fmt}{postfix}' 
            callbacks.append(tfa.callbacks.TQDMProgressBar(show_epoch_progress=False, overall_bar_format=bar))

        self.model.fit(C,#, "base_W": base_W},
                       y=X, batch_size=batch_size, epochs=epochs,
            callbacks=callbacks, validation_split=val_split, verbose=0)


    def predict_w(self, C, project_to_dag=False):
        preds = self.model.predict(C)
        return preds
