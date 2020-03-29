import torch
from torch import nn


class Processor(nn.Module):
    def __init__(self):
        super(Processor, self).__init__()

    def forward(self, *args, **kwargs):
        controls = self.get_controls(*args, **kwargs)
        signal = self.get_signal(**controls)
        return signal

    def get_controls(self, *args, **kwargs):
        """Convert input tensors into a dict of processor controls."""
        raise NotImplementedError

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


class FilteredNoise(Processor):
    """Synthesize audio by filtering white noise."""

    def __init__(self,
                 n_samples=64000,
                 window_size=257,
                 scale_fn=torch.sigmoid,
                 initial_bias=-5.0
                 ):
        self.n_samples= n_samples
        self.window_size = window_size
        self.scale_fn = scale_fn
        self.initial_bias = initial_bias
    
    def get_controls(self, magnitudes):
        """Converts network outputs into a dictionary of synthesizer controls.

        Args:
            magnitude (B x T x n_filter_banks): input tensor
        
        Returns:
            controls: Dictionary of tensors of synthesizer controls.
        """
        if self.scale_fn is not None:
            magnitudes = self.scale_fn(magnitudes + self.initial_bias, axis=-1)
        
        return {'magnitudes': magnitudes}
    
    def get_signal(self, magnitudes: torch.Tensor):
        """Synthesizes audio with filtered white noise.

        Args:
            magnitudes (B x T x n_filter_banks): input tensor
        
        Returns:
            signal: A tensor of hamonic waves of shape (B x n_samples x 1)
        """

        batch_size = magnitudes.shape[0]
        signal = torch.rand(B, self.n_samples)
        core.
