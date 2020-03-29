import torch
from torch import nn
import torchaudio
from torch.functional import F
from ddsp.spectral_ops import compute_stft_mag, compute_loudness_from_spec


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)


def mean_difference(target, value, loss_type='L1'):
    difference = target - value
    loss_type = loss_type.upper()
    if loss_type == 'L1':
        return torch.mean(torch.abs(difference))
    elif loss_type == 'L2':
        return torch.mean(difference ** 2)
    elif loss_type == 'COSINE':
        return cosine_distance(target, value)
    else:
        raise ValueError('Loss type ({}), must be '
                         '"L1", "L2", or "COSINE"'.format(loss_type))


class SpectralLoss(nn.Module):
    """Multi-scale spectrogram loss."""

    def __init__(self,
                 fft_sizes=[2048, 1024, 512, 256, 128, 64],
                 loss_type='L1',
                 mag_weight=1.0,
                 delta_time_weight=0.0,
                 delta_delta_time_weight=0.0,
                 delta_freq_weight=0.0,
                 delta_delta_freq_weight=0.0,
                 logmag_weight=0.0,
                 loudness_weight=0.0):
        super(SpectralLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.loss_type = loss_type
        self.mag_weight = mag_weight
        self.delta_time_weight = delta_time_weight
        self.delta_delta_time_weight = delta_delta_time_weight
        self.delta_freq_weight = delta_freq_weight
        self.delta_delta_freq_weight = delta_delta_freq_weight
        self.logmag_weight = logmag_weight
        self.loudness_weight = loudness_weight

    def forward(self,  predict, target):
        loss = 0.0
        loss_ops = []
        for n_fft in self.fft_sizes:
            pred_mag = compute_stft_mag(predict, n_fft)
            target_mag = compute_stft_mag(target, n_fft)

            if self.mag_weight > 0:
                loss += self.mag_weight * \
                    mean_difference(target_mag, pred_mag, self.loss_type)

            if self.delta_time_weight > 0:
                target_diff = target_mag[:, :, 1:] - target_mag[:, :, :-1]
                pred_diff = pred_mag[:, :, 1:] - pred_mag[:, :, :-1]
                loss += self.delta_time_weight * \
                    mean_difference(target_diff, pred_diff, self.loss_type)

            if self.delta_freq_weight > 0:
                target_diff = target_mag[:, 1:, :] - target_mag[:, :-1, :]
                pred_diff = pred_mag[:, 1:, :] - pred_mag[:, :-1, :]
                loss += self.delta_freq_weight * \
                    mean_difference(target_diff, pred_diff, self.loss_type)

            if self.delta_delta_freq_weight > 0:
                target_diff = target_mag[:, 1:, :] - target_mag[:, :-1, :]
                pred_diff = pred_mag[:, 1:, :] - pred_mag[:, :-1, :]
                target_diff2 = target_diff[:,
                                           1:, :] - target_diff[:, :-1, :]
                pred_diff2 = pred_diff[:, 1:, :] - pred_diff[:, :-1, :]

                loss += self.delta_delta_freq_weight * \
                    mean_difference(target_diff2,
                                    pred_diff2, self.loss_type)

            if self.delta_delta_time_weight:
                target_diff = target_mag[:, :, 1:] - target_mag[:, :, :-1]
                target_diff2 = target_diff[:, :, 1:] - target_diff[:, :, :-1]
                pred_diff = pred_mag[:, :, 1:] - pred_mag[:, :, :-1]
                pred_diff2 = pred_diff[:, :, 1:] - pred_diff[:, :, :-1]
                loss += self.delta_freq_weight * \
                    mean_difference(target_diff2, pred_diff2, self.loss_type)

            if self.logmag_weight > 0:
                # log_target_mag = torch.log(torch.clamp(target_mag, 1e-5))
                # log_pred_mag = torch.log(torch.clamp(pred_mag, 1e-5))
                log_target_mag = torch.log(target_mag + 1e-5)
                log_pred_mag = torch.log(pred_mag + 1e-5)
                loss += self.logmag_weight * \
                    mean_difference(
                        log_target_mag, log_pred_mag, self.loss_type)

            if self.loudness_weight > 0:
                loudness_target = compute_loudness_from_spec(target)
                loudness_pred = compute_loudness_from_spec(predict)
                loss += self.loudness_weight * \
                    mean_difference(loudness_target,
                                    loudness_pred, self.loss_type)


class CrepeF0Loss(nn.Module):
    """F0 Loss based on Crepe model."""

    def __init__(self, loss_type='L0'):
        super(CrepeF0Loss, self).__init__()

        self.loss_type = loss_type
        
        # crepe model
        self.crepe = torch.hub.load(
            'tuan3w/crepe-pytorch', 'load_crepe', 'full')
        self.crepe.eval()

    def get_frames(self, audio: torch.Tensor, sr=16000, hop_length=1024, center=True) -> torch.Tensor:
        """Splits audio into frames of size 1024.
        Args:
            audio (B x T): audio batch

        Returns:
            audio frames of size (B x n_frames x 1024)

        """
        if sr != 16000:
            # resample audio
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio = resampler(audio)

        if center:
            audio = F.pad(audio, (512, 512))

        # change audio to 2D tensor
        if len(audio.shape) == 1:
            audio = audio.unsqueeze(0)

        # create strided frames
        audio_len = audio.shape[-1]
        B = audio.shape[0]

        # calculate number of frames in each stream to return
        n_frames = (audio_len - hop_length)//hop_length + 1
        frames = torch.as_strided(audio,
                                  size=(B, n_frames, 1024),
                                  stride=(audio_len, hop_length, 1))
        std, mean = torch.std_mean(frames, dim=-1, keepdim=True)
        return (frames - mean)/(std + 1e-5)

    def forward(self, predict, target, sample_rate=16000, step_size=10, batch_size=128):
        """Returns pitch loss  based on crepe picth estimations.

        Args:
            predict (B X T): predicted audio, length must be divisible by 1024.
            target (B X T): target audio batch, length must be divisible by 1024.

        Returns:
            pitch loss
        """
        pred_frames = self.get_frames(predict).reshape(-1, 1024)
        target_frames = self.get_frames(target).reshape(-1, 1024)

        pred_freq = self.crepe(pred_frames)
        target_freq = self.crepe(target_frames)

        #TODO: should we calculate softmax values ?

        return mean_difference(target_freq, pred_freq, loss_type=self.loss_type)

