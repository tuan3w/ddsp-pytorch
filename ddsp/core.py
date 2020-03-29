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


def unit_to_midi(unit,
                 midi_min=20.0,
                 midi_max=90.0,
                 clip=False):
    """Map the unit interval [0, 1] to MIDI notes."""
    unit = torch.clamp(unit, 0.0, 1.0) if clip else unit
    return midi_min + (midi_max - midi_min) * unit


def midi_to_unit(midi, midi_min=20.0, midi_max=90.0, clip=False):
    """Map MIDI notes to the unit interval [0, 1]."""
    unit = (midi - midi_min) / (midi_max - midi_min)
    return torch.clamp(unit, min=0, max=1.0) if clip else unit


def resample(inputs: torch.Tensor,
             n_steps: int,
             mode='linear',
             add_endpoint=True) -> torch.Tensor:
    """Interpolates a tensor from n_frames to n_timesteps.

    Args:
        inputs: Framewise 1-D, 1-D, 3-D, 4-D tensor. Shape [n_frames],
            [batch_size, n_frames], [batch_size, channels, n_frames], 
            or [batch_size, n_freq, channesl, n_frames].
        n_timesteps: Time resolution of the output signal.
        mode: Type of resampling, must be in ['linear', 'cubic', 'window']. Linear overlapping windows
            (only for upsampling) which is smoother for amplitudes envelopes.
        add_endpoint: Hold the last timestep for an additional step as the endpoint.
            Then, n_timesteps is divided evenly to n_frames segments. If false, use
            the last timestep as the endpoint, procuding (n_frames - 1) segments with each having a length 
            of n_timesteps / (n_frames -1).

    Returns:
        Intepolated 1-D, 2-D, 3-D, 4-D Tensor. Shape [n_timesteps],
            [batch_size, n_timesteps], [batch_size, channels, n_timesteps], or
            [batch_size, n_freqs, channels, n_timesteps].

    """
    is_1d = len(input.shape) == 1
    is_2d = len(input.shape) == 2

    # Ensure inputs are at least 3d.
    if is_1d:
        inputs = inputs.reshape(1, 1,  -1j)
    elif is_2d:
        inputs = inputs.unsqueeze(0)

    outputs = F.interpolate(inputs, size=(
        n_steps), mode=mode, align_corners=not add_endpoint)

    if is_1d:
        outputs = outputs[0, 0]
    elif is_2d:
        outputs = outputs[0]

    return outputs


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


def haromic_synthesis(frequencies: torch.Tensor,
                      amplitudes: torch.Tensor,
                      harmonic_shifts: torch.Tensor,
                      harmonic_distribution: torch.Tensor = None,
                      n_samples: int = 64000,
                      sample_rate: int = 16000,
                      amp_resample_method='window') -> torch.Tensor:
    """Generates audio from frame-wise monophonic harmonic oscillator bank.

    Args:
        frequencies (B, n_frames, 1): Frame-wise fundamental frequency in Hz.
        amplitudes: (B, n_frames, 1): Frame-wise oscilator peak amplitude.
        harmonic_shifts (B, n_frames, n_harmonics): Hamonic frequency variations (Hz), zero-centered. Total 
            frequency of a harmonic is equal to (frequencies * harmonic_number * (1 + harmonic_shifts)).
        harmonic_distribution: (B, n_frames, n_harmonics): Harmonic amplitude variations, ranged zero to one.
            Total amplitude of a harmonic is equal to (amplitudes * harmonic_distribution).
        n_samples: Total length of output audio, interpolates and crops to this.
        sample_rate: Sample rate,
        amp_resample_mthod: Mode with which to resample amplitude envelopes.

    Returns:
        audio: Output audio, shape [B, n_samples, 1]
    """

    if harmonic_distribution is not None:
        n_harmonics = harmonic_distribution.shape[-1]
    elif harmonic_shifts is not None:
        n_harmonics = harmonic_shifts.shape[-1]
    else:
        n_harmonics = 1

    # Create harmonic frequencies [batch_size, n_frames, n_harmonics].
    harmonic_frequencies = get_harmonic_frequencies(frequencies, n_harmonics)
    if harmonic_shifts is not None:
        harmonic_frequencies *= (1.0 + harmonic_shifts)

    # Create harmonic amplitudes [batch_size, n_frames, n_harmonics].
    if harmonic_distribution is not None:
        harmonic_amplitudes = amplitudes * harmonic_distribution
    else:
        harmonic_amplitudes = amplitudes

    # Create sample-wise envelopes.
    frequency_envelopes = resample(
        harmonic_frequencies, n_samples)  # cycles/sec
    amplitude_envelopes = resample(
        harmonic_amplitudes, n_samples, method=amp_resample_method)

    # Synthesize from harmonics [batch_size, n_samples].
    audio = oscillator_bank(frequency_envelopes,
                            amplitude_envelopes, sample_rate=sample_rate)

    return audio


def linear_lookup(phase: torch.Tensor,
                  wavetables: torch.Tensor) -> torch.Tensor:
    """Lookups from wavetables with linear interpolation.
    
    Args:
        phase: The instantaneous phase of the base oscillator, ranging from 0 to 1.0.
            This gives the position to lookup in the wavetable.
        wavetables (batch_size, n_samples, n_wavetables) or (batch_size, n_wavetable): Wavetables to be read from on lookup.
    
    Returns:
        The resulting audio from linearly interpolated lookup of the wavetables at 
            each point in time. Shape [batch_size, n_samples].
    """

    # Add a time dimension if not present.
    if len(wavetables.shape) == 2:
        wavetables = wavetables.unsqueeze(1)

    # Add a wavetable dimension if not present.
    if len(phase.shape) == 2:
        phase = phase.unsqueeze(-1)
    
    n_wavetable = wavetables.shape[-1]

    # Get a phase value for each point on the wavetable
    phase_wavetables = torch.linspace(0, 1.0, n_wavetable)

    # Get pair-wise distances from the oscillator phase to eachwavetable point.
    # Axes are [batch, time, n_wavetable].
    # phase_distance = torch.abs((phase - phase_wavetables))

    # TODO:fix this
    


def wavetable_synthesis(frequencies: torch.Tensor,
                        amplitudes: torch.Tensor,
                        wavetables: torch.Tensor,
                        n_samples=64000,
                        sample_rate=16000) -> torch.Tensor:
    """Monophonic wavetable synthesizer.

    Args:
        frequencies (batch_size, n_frames, 1): Frame-wise frequency in Hertz of the fundamental oscillator.
        amplitudes (batch_size, n_frames, 1): Frame-wise amplitude envelop to apply to the oscillator.
        wavetables (batch_size, n_wavetable): Frame-wise wavetables which to lookup.
        n_samples: Total length of output audio. Intepolates and crops to this.
        sample_rate: Number  of samples per a second.

    Returns:
        audio (batch_size, n_samples): Audio at frequency and amplitude of the inputs,with 
            harmonics given by the wavetable.
    """
    # Create sample-wise envelopes.
    amplitude_envelope = resample(amplitudes, n_samples)
    frequency_envelope = resample(frequencies, n_samples)

    # Create intermediaate wavetables.
    if len(wavetables.shape) == 3 and wavetables[1] > 1:
        wavetables = resample(wavetables, n_samples)

    # Accumulate phase (in cycles which range from 0.0 to 1.0).
    phase_velocity = frequency_envelope / float(sample_rate)

    phase = torch.cumsum(phase_velocity, axis=1) % 1.0

    # Synthesize with linear lookup.
    audio = linear_lookup(pharse, wavetables)

    # Modulate with amplitude envelope.
    audio *= amplitude_envelope
    return audio

    pass
