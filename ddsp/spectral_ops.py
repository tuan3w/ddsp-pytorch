import torch
from torch.functional import F
import torchaudio
from librosa.filters import mel as librosa_mel_fn
from torchaudio import transforms
import librosa
import math

_LOG10 = math.log(10.0)

LD_RANGE = 120.0  # db
F0_RANGE = 127.0  # MIDI


def _log10(x):
    return torch.log(x) / _LOG10


def compute_stft(audio, n_fft, hop_length=None, win_length=None, window=None, center=False) -> torch.Tensor:
    """Computes stft features from given audio.

    Args:
        audio (B, T): audio batch
        n_fft (int); frame size
        hop_length (int): hop length
        win_length (int)
        overlap (float): determines how much audio frame will be overlap

    Returns:
        Tensor: tensor contains stft result, of shape (B,frame_size, num_frames, 2)
    """
    s = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center)
    return s


def compute_stft_mag(audio, n_fft=2048, win_length=None, hop_length=None, window=None, center=False):
    stft = compute_stft(audio, n_fft, hop_length, win_length, window, center)
    mag = torch.sqrt(stft[:, :, :, 0] ** 2 + stft[:, :, :, 1] ** 2)
    return mag


def compute_mel(audio,
                f_min=0.0,
                f_max=8000.0,
                n_mels=64,
                n_fft=2048,
                sample_rate=16000,
                overlap=0.75,
                center=True) -> torch.Tensor:
    """Calculates Mel Spectrogram.

    Returns:
        Tensor of shape (B, bins, T)
    """
    hop_length = int(n_fft * (1 - overlap))

    # this would be slow because we init mel converter everytime
    # you want to compute mel
    # In practice, you should init Melspectrogram transform and
    # use this object instead
    melconverter = transforms.MelSpectrogram(
        sample_rate,
        n_fft,
        hop_length=hop_length,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels
    )

    return melconverter(audio)


def diff(x: torch.Tensor, axis=-1, pad=False):
    """Take the finite difference of a tensor along an axis.

    Args:
        x: Input tensor of any dimension.
        axis: Axis on which to take the finite difference.

    Returns:
        d: Tensor with size less than x by 1 along the difference dimension.
    """

    # TODO: implement this
    raise NotImplementedError


def compute_mfcc(audio: torch.Tensor,
                 f_min=0,
                 f_max=8000.0,
                 n_fft=2048,
                 win_length=None,
                 hop_length=None,
                 sample_rate=16000,
                 n_mels=128,
                 n_mfcc=13,
                 window=None,
                 center=None
                 ) -> torch.Tensor:
    """Calculate Mel-frequency Cepstral Coefficients.
    Args:
        audio (B, T): audio batch 
        n_mfcc: number of mel-frequency Cepstral coefficients

    Returns:
        T: tensor of size (B, n_mfcc, T)
    """
    mfcc_transformer = transforms.MFCC(
        sample_rate,
        n_mfcc=n_mfcc,
        f_min=f_min,
        f_max=f_max,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        n_mels=n_mels,
        center=center
    ).to(audio.device)

    return mfcc_transformer(audio)


def compute_loudness(audio,
                     sample_rate=16000,
                     n_fft=2048,
                     hop_length=None,
                     top_db=LD_RANGE,
                     ref_db=20.7,
                     center=True
                     ) -> torch.Tensor:
    """Compute perceptual loudness in dB, related to white noise, amplitude =1

    Args:
        audio (B, T): input audio batch
        sample_rate: audio sample rate in Hz.
        frame_rate: Rate of loudness in Hz.
        n_fft: fft window size.
        top_db:Sets the dyamic range of loudness in decibles. The minimum 
            loudness (per  a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by 
            (A_weighting +10 * log10(abs(sttf(audio))**2.0)).The default value
            corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a 
            slight dependence on n_fft due to different graluariy of perceptual weighting.

    Returns:
        Loudness in decibles, shape (B, n_frames)
    """
    # import pdb; pdb.set_trace()
    window = torch.hann_window(n_fft).to(audio.device)
    stft = compute_stft(audio, n_fft, window=window,
                        hop_length=hop_length, center=center)

    amplitude = torch.sqrt(stft[:, :, :, 0] ** 2 + stft[:, :, :, 1] ** 2)

    # perceptual weighting.
    # copy directly from https://github.com/pytorch/audio/blob/933f6037e0b9bbe51c7720c5d3ba4174dc4f67f9/torchaudio/transforms.py#L140

    db_multiplier = math.log10(max(1e-10, ref_db))
    power_db = torchaudio.functional.amplitude_to_DB(
        amplitude, 10, 1e-10, db_multiplier, top_db=top_db)
    return power_db


def compute_loudness_from_spec(amplitude,
                               sample_rate=16000,
                               n_fft=2048,
                               hop_length=None,
                               top_db=LD_RANGE,
                               ref_db=20.7,
                               center=True
                               ) -> torch.Tensor:
    """Compute perceptual loudness in dB, related to white noise, amplitude =1

    Args:
        spec (B, T): input audio spectrogram
        n_fft: fft window size.
        top_db:Sets the dyamic range of loudness in decibles. The minimum 
            loudness (per  a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by 
            (A_weighting +10 * log10(abs(sttf(audio))**2.0)).The default value
            corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a 
            slight dependence on n_fft due to different graluariy of perceptual weighting.

    Returns:
        Loudness in decibles, shape (B, n_frames)
    """

    # perceptual weighting.
    # copy directly from https://github.com/pytorch/audio/blob/933f6037e0b9bbe51c7720c5d3ba4174dc4f67f9/torchaudio/transforms.py#L140
    db_multiplier = math.log10(max(1e-10, ref_db))
    power_db = torchaudio.functional.amplitude_to_DB(
        amplitude, 10, 1e-10, db_multiplier, top_db=top_db)
    return power_db


def compute_f0(audio,
               sample_rate,
               frame_rate,
               viterbi=True,
               model=None,
               model_capacity='full',
               center=True,
               step_size=10,
               batch_size=128
               ):
    """Estimates fundamental frequency (f0) using CREPE

    Args:
        audio (B, T): audio batch
        sample_rate: Sample rate in Hz.
        viterbi: Whether use vitebi decoding to estimate f0 in CREPE.
        model: Crepe model
        model_capacity (string): model type, valid values are 'full', 'small', 'medium','large','tiny'

    Returns:
        f0_hz (B, n_frame): Fundamental frequency in Hz
        f0_confidence (B, n_frame): Confidencee in Hz estimation
    """
    if model is None:
        model = torch.hub.load('tuan3w/crepe-pytorch',
                               'load_crepe', model_capacity)
        model.eval()
        model = model.to(audio.device)

    time, f0_hz, f0_confidence, _ = model.predict(
        audio,
        sample_rate,
        center=True,
        step_size=step_size,
        batch_size=batch_size)

    return f0_hz, f0_confidence
