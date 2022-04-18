import numpy as np
import torch

from contextualized.networks import CorrelationTrainer, ContextualizedCorrelation, MultitaskContextualizedCorrelation


if __name__ == '__main__':
    n = 100
    c_dim = 4
    x_dim = 2
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

    def correlation_quicktest(model):
        print(f'{type(model)} quicktest')
        dataloader = model.dataloader(C, X, batch_size=32)
        trainer = CorrelationTrainer(max_epochs=1)
        y_preds = trainer.predict_y(model, dataloader)
        y_true = np.tile(X[:,:,np.newaxis], (1, 1, X.shape[-1]))
        err_init = ((y_true - y_preds)**2).mean()
        trainer.fit(model, dataloader)
        trainer.validate(model, dataloader)
        trainer.test(model, dataloader)
        beta_preds, mu_preds = trainer.predict_params(model, dataloader)
        network_preds = trainer.predict_network(model, dataloader)
        y_preds = trainer.predict_y(model, dataloader)
        err_trained = ((y_true - y_preds)**2).mean()
        assert err_trained < err_init
        print()

    # Correlation
    model = ContextualizedCorrelation(c_dim, x_dim)
    correlation_quicktest(model)
    
    # Tasksplit Correlation
    model = MultitaskContextualizedCorrelation(c_dim, x_dim)
    correlation_quicktest(model)
