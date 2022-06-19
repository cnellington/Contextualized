import torch
import dill


def save(model, path):
    torch.save(model, open(path, 'wb'), pickle_module=dill)


def load(path):
    model = torch.load(open(path, 'rb'), pickle_module=dill)
    return model

