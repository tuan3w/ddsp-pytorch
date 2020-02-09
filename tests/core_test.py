import torch
import unittest
import numpy as np
from scipy import signal
import librosa
from ddsp import core


class CoreTest(unittest.TestCase):
    def test_midi_to_hz(self):
        print('core')
        midi = np.arange(128)
        librosa_hz = librosa.midi_to_hz(midi)
        tensor_hz = core.midi_to_hz(torch.FloatTensor(midi)).data.numpy()

        assert np.allclose(librosa_hz, tensor_hz)

