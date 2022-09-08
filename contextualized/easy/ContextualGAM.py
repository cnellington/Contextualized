from contextualized.easy import ContextualizedClassifier, ContextualizedRegressor


class ContextualGAMClassifier(ContextualizedClassifier):
    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)


class ContextualGAMRegressor(ContextualizedRegressor):
    def __init__(self, **kwargs):
        kwargs["encoder_type"] = "ngam"
        super().__init__(**kwargs)
