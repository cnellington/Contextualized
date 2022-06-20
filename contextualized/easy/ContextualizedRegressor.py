from contextualized.regression import NaiveContextualizedRegression, ContextualizedRegression
from contextualized.regression import REGULARIZERS, LINK_FUNCTIONS, LOSSES

from contextualized.easy.wrappers import SKLearnInterface

# TODO: Multitask metamodels
# TODO: Task-specific link functions.
# TODO: Easier early stopping (right now, have to pass in 'callback_constructors' kwarg.


class ContextualizedRegressor(SKLearnInterface):
    def __init__(self, **kwargs):
        self.num_archetypes = kwargs.get('num_archetypes', 0)
        if self.num_archetypes == 0:
            self.constructor = NaiveContextualizedRegression
        elif self.num_archetypes > 0:
            self.constructor = ContextualizedRegression
        else:
            print("Was told to construct a ContextualizedRegressor with {} archetypes, but this should be a non-negative integer.".format(
                self.num_archetypes))
        self.constructor_kwargs = kwargs
        self.constructor_kwargs['link_fn'] = kwargs.get('link_fn', LINK_FUNCTIONS['identity'])
        self.constructor_kwargs['univariate'] = kwargs.get('univariate', False)
        self.constructor_kwargs['encoder_type'] = kwargs.get('encoder_type', 'mlp')
        self.constructor_kwargs['loss_fn'] = kwargs.get('loss_fn', LOSSES['mse'])
        self.constructor_kwargs['encoder_kwargs'] = kwargs.get(
            'encoder_kwargs', {'width': 25, 'layers': 2, 'link_fn': LINK_FUNCTIONS['identity']}
        )
        if kwargs.get('subtype_probabilities', False):
            self.constructor_kwargs['encoder_kwargs']['link_fn'] = LINK_FUNCTIONS['softmax']

        # Make regularizer
        if 'alpha' in kwargs and kwargs['alpha'] > 0:
            print(kwargs['alpha'], kwargs.get('l1_ratio', 1.0), kwargs.get('mu_ratio', 0.5))
            self.constructor_kwargs['model_regularizer'] = REGULARIZERS['l1_l2'](
                kwargs['alpha'], kwargs.get('l1_ratio', 1.0), kwargs.get('mu_ratio', 0.5))
        else:
            self.constructor_kwargs['model_regularizer'] = kwargs.get('model_regularizer', REGULARIZERS['none'])

        super().__init__(self.constructor)

    def fit(self, C, X, Y, **kwargs):
        # Merge kwargs and self.constructor_kwargs, prioritizing more recent kwargs.
        for k, v in self.constructor_kwargs.items():
            if k not in kwargs:
                kwargs[k] = v
        return super().fit(C, X, Y, **kwargs)

    def predict(self, C, X, **kwargs):
        return super().predict(C, X, **kwargs)

    def predict_params(self, C, **kwargs):
        return super().predict_params(C, **kwargs)
