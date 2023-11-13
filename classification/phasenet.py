import json

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import seisbench.util as sbu



class PhaseNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        in_samples=6000,
        classes=3,
        labels="NPS",
        sampling_rate=100,
        norm="std",
        output_type="array",	
        pred_sample=(0, 6000),
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu
        self.in_samples = in_samples
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.output_type = output_type
        self.pred_sample = pred_sample

        self.inc = nn.Conv1d(
            self.in_channels, self.filters_root, self.kernel_size, padding="same"
        )
        self.in_bn = nn.BatchNorm1d(8, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root
        for i in range(self.depth):
            filters = int(2**i * self.filters_root)
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root)
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    def annotate_window_pre(self, window, argdict):
        # Add a demean and normalize step to the preprocessing
        window = window - np.mean(window, axis=-1, keepdims=True)
        if self.norm_detrend:
            detrended = np.zeros(window.shape)
            for i, a in enumerate(window):
                detrended[i, :] = scipy.signal.detrend(a)
            window = detrended
        if self.norm_amp_per_comp:
            amp_normed = np.zeros(window.shape)
            for i, a in enumerate(window):
                amp = a / (np.max(np.abs(a)) + 1e-10)
                amp_normed[i, :] = amp
            window = amp_normed
        else:
            if self.norm == "std":
                std = np.std(window, axis=-1, keepdims=True)
                std[std == 0] = 1  # Avoid NaN errors
                window = window / std
            elif self.norm == "peak":
                peak = np.max(np.abs(window), axis=-1, keepdims=True) + 1e-10
                window = window / peak

        return window

    def annotate_window_post(self, pred, piggyback=None, argdict=None):
        # Transpose predictions to correct shape
        pred = pred.T
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            pred[:prenan] = np.nan
        if postnan > 0:
            pred[-postnan:] = np.nan
        return pred

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def get_model_args(self):
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
            "sampling_rate",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate

        return model_args


if __name__ == "__main__":
    x = torch.randn(32, 1, 6000)
    model = PhaseNet()
    y = model(x)
    print(y.shape)