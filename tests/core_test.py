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

    def test_hz_to_midi(self):
        hz = np.linspace(20.0, 20000.0, 128)
        librosa_midi = librosa.hz_to_midi(hz)
        tensor_midi = core.hz_to_midi(torch.FloatTensor(hz))
        assert np.allclose(librosa_midi, tensor_midi.data.numpy())
    
    def test_resample(self):
        samples = 1.0 - np.sin(np.linspace(0, np.pi, 1000))[np.newaxis,:, np.newaxis]
        samples = torch.FloatTensor(samples)

        x = np.random.rand(2,3)
        output = core.resample(samples, 100)
        self.assertEqual(output.shape[1], 100)

        # 2d signal
        output = core.resample(samples.squeeze(-1), 100)
        self.assertEqual(output.shape[1], 100)




