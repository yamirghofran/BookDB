import os
import torch
import glob
import re


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

def get_best_checkpoint(model_prefix):
    """Return the checkpoint file path with the best HR (and NDCG as tiebreaker) for a given model prefix."""
    checkpoint_dir = 'checkpoints'
    pattern = os.path.join(checkpoint_dir, f"{model_prefix}*_Epoch*_HR*_NDCG*.model")
    checkpoint_files = glob.glob(pattern)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found for pattern: {pattern}")

    best_file = None
    best_hr = -1
    best_ndcg = -1

    # Regex to extract HR and NDCG from filename
    regex = re.compile(r'_HR([0-9.]+)_NDCG([0-9.]+)\.model')

    for file in checkpoint_files:
        match = regex.search(file)
        if match:
            hr = float(match.group(1))
            ndcg = float(match.group(2))
            if (hr > best_hr) or (hr == best_hr and ndcg > best_ndcg):
                best_hr = hr
                best_ndcg = ndcg
                best_file = file

    if best_file is None:
        raise ValueError("No valid checkpoint files found with HR and NDCG in the filename.")
    return best_file


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