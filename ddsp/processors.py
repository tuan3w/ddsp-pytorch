import torch
from torch import nn


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

    def forward(self, *input, **kwargs):
        controls = self.get_controls(*args, **kargs)
        signal = self.get_signal(**controls)
        return signal

    def get_controls(self, *args, **kwargs):
        """Convert input tensors into a dict of processor controls."""
        rase NotImplementedError

    def get_signal(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError


class Add(Processor):
    """Sum two signals."""

    def __init__(self):
        super(Add, self).__init__()

    def get_controls(self, signal_one: torch.Tensor,
                     signal_two: torch.Tensor):
        return {
            'signal_one': signal_one, 'signal_two': signal_two
        }

    def get_signal(self, signal_one: torch.Tensor,
                   signal_two: torch.Tensor) -> torch.Tensor:
        return signal_one + signal_two


class Mix(Processor):
    """Constant-power crossfade between two signals."""

    def __init__(self):
        super(Mix, self).__init__()

    def get_controls(self, signal_one: torch.Tensor,
                     signal_two: torch.Tensor,
                     mix_level: torch.Tensor):
        """Mix two signals together

        Args:
            signal_one: 2-D or 3-D tensor.
            signal_two: 2-D or 3-D tensor.
            mix_level (B, n_time, 1):  controls the levels of signals.
        """
        return {
            'signal_one': signal_one,
            'signal_two': signal_two,
            'mix_level': mix_level
        }

    def get_signal(self, signal_one, signal_two, mix_level):
        return mix_level * signal_one + (1-mix_level) * signal_two
