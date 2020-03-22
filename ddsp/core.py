import math

import numpy as np
import torch
from torch.nn import functional as F

LOG2 = math.log(2)

def midi_to_hz(notes):
    """converts midi to hz"""
    return 440.0 * (2.0**((notes - 69.0) / 12.0))


def log2(x):
    return math.log(x) / LOG2


def tensor_log2(x):
    return torch.log(x) / LOG2


def hz_to_midi(frequences):
    notes = 12.0 * (tensor_log2(frequences) - log2(440.0)) + 69.0
    return torch.clamp(notes, min=0)


def resample(inputs: torch.Tensor,
             n_timesteps):
    """Resamples tensor of shape [:, n_frames, :] to [:, n_timesteps, :].
    """
    if len(inputs.shape) == 2:
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


def remove_above_nyquist(frequency_evelopes: torch.Tensor,
                         amplitude_evelopes: torch.Tensor,
                         sample_rate: int = 16000
                         ) -> torch.Tensor:
    """ Set amplitudes for oscillators above nyquist to 0."""
    # TODO: should we change this to max_freq ?
    amplitude_evelopes[frequency_evelopes > sample_rate / 2] = 0.0
    return amplitude_evelopes


def oscillator_bank(frequency_evelopes: torch.Tensor,
                    amplitude_envelopes: torch.Tensor,
                    sample_rate: int = 16000) -> torch.Tensor:
    """Generates audio from sample-wise frequecies for a bank of oscillators."""
    amplitude_envelopes = remove_above_nyquist(frequency_evelopes,
                                               amplitude_envelopes,
                                               sample_rate)
    # Change Hz to radians per sample.
    omegas = frequency_evelopes * (2.0 * np.pi)

    # Accumulate phrase and synthesize.
    phases = torch.cumsum(omegas, axis=1)
    wavs = torch.sin(phases)
    harmonic_audio = amplitude_envelopes * wavs  # [B, n_samples, n_sinusoids]
    audio = torch.sum(harmonic_audio, axis=-1)
    return audio

def get_harmonic_frequencies(frequecies: torch.Tensor,
                             n_harmonics: int) -> torch.Tensor:
    """Create integer multiples of the fundamental frequency."""
    f_ratios = torch.linspace(1.0, n_harmonics, n_harmonics)
    f_ratios = f_ratios.reshape(1, 1, -1)
    harmonic_frequencies = frequecies * f_ratios
    return harmonic_frequencies
