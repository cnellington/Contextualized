import numpy as np
import torch

from contextualized.easy import ContextualizedClassifier, ContextualizedRegressor


def quicktest(model, C, X, Y, **kwargs):
    print(f'{type(model)} quicktest')
    model.fit(C, X, Y, max_epochs=0)
    err_init = np.linalg.norm(Y - model.predict(C, X), ord=2)
    model.fit(C, X, Y, **kwargs)
    beta_preds, mu_preds = model.predict_params(C)
    assert beta_preds.shape == (X.shape[0], Y.shape[1], X.shape[1])
    assert mu_preds.shape == (X.shape[0], Y.shape[1])
    y_preds = model.predict(C, X)
    assert y_preds.shape == Y.shape
    err_trained = np.linalg.norm(Y - y_preds, ord=2)
    assert err_trained < err_init
    print()


def test_regressor():
    n = 100
    c_dim = 4
    x_dim = 5
    y_dim = 3
    C = torch.rand((n, c_dim)) - .5
    W_1 = C.sum(axis=1).unsqueeze(-1) ** 2
    W_2 = - C.sum(axis=1).unsqueeze(-1)
    b_1 = C[:, 0].unsqueeze(-1)
    b_2 = C[:, 1].unsqueeze(-1)
    W_full = torch.cat((W_1, W_2), axis=1)
    b_full = b_1 + b_2
    X = torch.rand((n, x_dim)) - .5
    Y_1 = X[:, 0].unsqueeze(-1) * W_1 + b_1
    Y_2 = X[:, 1].unsqueeze(-1) * W_2 + b_2
    Y_3 = X.sum(axis=1).unsqueeze(-1)
    Y = torch.cat((Y_1, Y_2, Y_3), axis=1)

    k = 10
    epochs = 2
    batch_size = 1
    C, X, Y = C.numpy(), X.numpy(), Y.numpy()

    # Naive Multivariate
    model = ContextualizedRegressor()
    quicktest(model, C, X, Y, max_epochs=1)

    model = ContextualizedRegressor(num_archetypes=0)
    quicktest(model, C, X, Y, max_epochs=1)

    model = ContextualizedRegressor(num_archetypes=4)
    quicktest(model, C, X, Y, max_epochs=5)

    # With regularization
    model = ContextualizedRegressor(num_archetypes=4, alpha=0.,
        l1_ratio=0.5, mu_ratio=0.9)
    quicktest(model, C, X, Y, max_epochs=1)

    # With bootstrap
    model = ContextualizedRegressor(num_archetypes=4, alpha=0.1,
        l1_ratio=0.5, mu_ratio=0.9)
    quicktest(model, C, X, Y, max_epochs=1, n_bootstraps=2,
        learning_rate=1e-3)


if __name__ == "__main__":
    test_regressor()
