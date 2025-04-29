"""
    Some handy functions for pytroch model training ...
"""
import os
import torch


# Checkpoints
def save_checkpoint(model, model_dir):
    dir = os.path.dirname(model_dir)
    if not os.path.exists(dir):
        os.makedirs(dir)
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device):
    state_dict = torch.load(model_dir, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded model from {model_dir} to {device}")


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            device = torch.device(f'cuda:{device_id}')
            print(f"Using CUDA device: {device}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple Metal (MPS) device.")
        else:
            device = torch.device("cpu")
            print("CUDA and MPS not available. Using CPU.")
    else:
        device = torch.device("cpu")
        print("CUDA/MPS disabled. Using CPU.")
    return device


def use_optimizer(network, params):
    if params['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(network.parameters(),
                                    lr=params['sgd_lr'],
                                    momentum=params['sgd_momentum'],
                                    weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(network.parameters(), 
                                                          lr=params['adam_lr'],
                                                          weight_decay=params['l2_regularization'])
    elif params['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(network.parameters(),
                                        lr=params['rmsprop_lr'],
                                        alpha=params['rmsprop_alpha'],
                                        momentum=params['rmsprop_momentum'])
    return optimizer