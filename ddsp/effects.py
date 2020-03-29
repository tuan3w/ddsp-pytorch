

from ddsp.processors import Processor
import torch


class Reverb(Processor):
    """Convolutional (FIR) reverb."""

    def __init__(self, trainable=False,
                 reverb_length=48000,
                 add_dry=True):
        """Takes neural network outputs directly as the impulse response.

        Args:
            trainable: learn the impulse_response as a single variable for the
                entire dataset.
            reverb_length: length of the impulse reponse. Only used if
                trainable=True.
            add_dry: Add dry signal to reverberated signal on output.
        """
        super(Reverb, self).__init__()
        self._reverbe_length = reverb_length
        self._add_dry = add_dry

    def _get_ir(self, gain, decay):
        """Simple exponential decay of white noise."""
        decay_exponent = 2.0 + torch.exp(decay)
        time = torch.linspace(0.0, 1.0, self._reverbe_length).unsqueeze(0)
        noise = (torch.rand(1, self._reverbe_length) - 0.5) * 2
        ir = gain * torch.exp(-decay_exponent * time) * noise

    def get_controsl(self, audio, gain=None, decay=None):
        """Converts network outputs into ir response.

        Args:
            audio: Dry audio. 2-D Tensor of shape [batch, n_samples].
            gain: Linear gain of impulse response. Scaled by self._scale_fn.
                2D tensor of shape [batch, 1]. Not used if trainable=True.
            decay: Exponential decay coefficient. The final impulse response is
                exp(-(2 + exp(decay)) * time) where time goes from 0 to 1 over the
                reverb_length samples. 2D Tensor of shape [batch, 1]

        Returns:
            controls: Dictionary of effect controls.
        """

        ir = self._get_ir(gain, decay)

        return {'audio': audio, 'ir': ir}
