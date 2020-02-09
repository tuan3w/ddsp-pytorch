import torch
import math
from torch.nn import functional as F


def midi_to_hz(notes):
    """converts midi to hz"""
    return 440.0 * (2.0**((notes - 69.0) / 12.0))


def log2(x):
    return math.log(x) / math.log(2)


def tensor_log2(x):
    return torch.log(x) / math.log(2)


def hz_to_midi(frequences):
    notes = 12.0 * (tensor_log2(frequences) - log2(440.0)) + 69.0
    return torch.clamp(notes, min=0)


def resample(inputs: torch.Tensor,
             n_timesteps):
    """Resamples tensor of shape [:, n_frames, :] to [:, n_timesteps, :].
    """
    methods = {'linear': 0, 'cubic': 2}

    if len(input.shape) == 2:
        inputs = inputs.unsqueeze(-1)

    outputs = inputs.unsqueeze(-1).transpose(1, 2)
    outputs = F.interpolate(outputs, (n_timesteps, 1))

    # transpose and return
    return outputs.transpose(1, 2)[:, :, :, 0]


def log_scale(x: torch.Tensor,
              min_val,
              max_val):
    x = (x + 1.0) / 2.0  # scale from [-1, 1] to [0, 1]
    return torch.exp((1.0 - x) * math.log(min_val) + x * math.log(max_val))


def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
    """Exponentiated Sigmoid pointwise nonlinearity.
    """
    return max_value * torch.sigmoid(x) ** math.log(exponent) + threshold


def sym_exp_sigmoid(x, width=8.0):
    """Symmetrical version of exp_sigmoid centered at (0, 1e-7)."""
    return exp_sigmoid(width, (torch.abs(x) / 2.0 - 1.0))


