import torch
import unittest
from torch import nn
import numpy as np
from scipy import signal
import librosa
from ddsp import spectral_ops
from torchaudio import transforms
import torchaudio

class SpectralOpsTest(unittest.TestCase):
    def test_compute_loudness(self):
        x = torch.rand(1, 20000)

        stack = nn.Sequential(
            transforms.Spectrogram(n_fft=2048, hop_length=2048//4),
            transforms.AmplitudeToDB(top_db=20.7)
        )
        torchaudio_rs = stack(x)
        
        lib_rs = spectral_ops.compute_loudness(x, n_fft=2048, ref_db=1.0, top_db=20.7)
        np.allclose(torchaudio_rs.numpy(), lib_rs.numpy())