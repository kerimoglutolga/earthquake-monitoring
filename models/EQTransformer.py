import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLSTM(nn.Module):
    """
    LSTM to be used with custom cells
    """

    def __init__(self, cell, *cell_args, bidirectional=True, **cell_kwargs):
        super(CustomLSTM, self).__init__()
        self.cell_f = cell(*cell_args, **cell_kwargs)
        self.bidirectional = bidirectional
        if self.bidirectional:
            self.cell_b = cell(*cell_args, **cell_kwargs)

    def forward(self, input, state=None):
        # Forward
        state_f = state
        outputs_f = []
        for i in range(len(input)):
            out, state_f = self.cell_f(input[i], state_f)
            outputs_f += [out]

        outputs_f = torch.stack(outputs_f)

        if not self.bidirectional:
            return outputs_f, None

        # Backward
        state_b = state
        outputs_b = []
        l = input.shape[0] - 1
        for i in range(len(input)):
            out, state_b = self.cell_b(input[l - i], state_b)
            outputs_b += [out]

        outputs_b = torch.flip(torch.stack(outputs_b), dims=[0])

        output = torch.cat([outputs_f, outputs_b], dim=-1)

        # Keep second argument for consistency with PyTorch LSTM
        return output, None


def hard_sigmoid(x):
    return torch.clip(0.2 * x + 0.5, 0, 1)


class ActivationLSTMCell(nn.Module):
    """
    LSTM Cell using variable gating activation, by default hard sigmoid

    If gate_activation=torch.sigmoid this is the standard LSTM cell

    Uses recurrent dropout strategy from https://arxiv.org/abs/1603.05118 to match Keras implementation.
    """

    def __init__(
            self, input_size, hidden_size, gate_activation=hard_sigmoid, recurrent_dropout=0
    ):
        super(ActivationLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_activation = gate_activation
        self.recurrent_dropout = recurrent_dropout

        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.randn(4 * hidden_size))
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            for param in [self.weight_hh, self.weight_ih]:
                for idx in range(4):
                    mul = param.shape[0] // 4
                    torch.nn.init.xavier_uniform_(param[idx * mul: (idx + 1) * mul])

    def forward(self, input, state):
        if state is None:
            hx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
            cx = torch.zeros(
                input.shape[0], self.hidden_size, device=input.device, dtype=input.dtype
            )
        else:
            hx, cx = state
        gates = (
                torch.mm(input, self.weight_ih.t())
                + self.bias_ih
                + torch.mm(hx, self.weight_hh.t())
                + self.bias_hh
        )
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = self.gate_activation(ingate)
        forgetgate = self.gate_activation(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = self.gate_activation(outgate)

        if self.recurrent_dropout > 0:
            cellgate = F.dropout(cellgate, p=self.recurrent_dropout)

        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


class EQTransformer(nn.Module):
    """
    The EQTransformer from Mousavi et al. (2020)

    Implementation adapted from the Github repository https://github.com/smousavi05/EQTransformer
    Assumes padding="same" and activation="relu" as in the pretrained EQTransformer models

    By instantiating the model with `from_pretrained("original")` a binary compatible version of the original
    EQTransformer with the original weights from Mousavi et al. (2020) can be loaded.

    .. document_args:: seisbench.models EQTransformer

    :param in_channels: Number of input channels, by default 3.
    :param in_samples: Number of input samples per channel, by default 6000.
                       The model expects input shape (in_channels, in_samples)
    :param classes: Number of output classes, by default 2. The detection channel is not counted.
    :param phases: Phase hints for the classes, by default "PS". Can be None.
    :param res_cnn_blocks: Number of residual convolutional blocks
    :param lstm_blocks: Number of LSTM blocks
    :param drop_rate: Dropout rate
    :param original_compatible: If True, uses a few custom layers for binary compatibility with original model
                                from Mousavi et al. (2020).
                                This option defaults to False.
                                It is usually recommended to stick to the default value, as the custom layers show
                                slightly worse performance than the PyTorch builtins.
                                The exception is when loading the original weights using :py:func:`from_pretrained`.
    :param norm: Data normalization strategy, either "peak" or "std".
    :param kwargs: Keyword arguments passed to the constructor of :py:class:`WaveformModel`.
    """

    def __init__(
            self,
            in_channels=3,
            in_samples=6000,
            classes=2,
            phases="PS",
            lstm_blocks=3,
            drop_rate=0.1,
            original_compatible=False,
            sampling_rate=100,
            norm="std",
            **kwargs,
    ):
        super().__init__()
        self.in_samples = in_samples
        self.in_channels = in_channels
        self.classes = classes
        self.lstm_blocks = lstm_blocks
        self.drop_rate = drop_rate
        self.norm = norm

        # Add options for conservative and the true original - see https://github.com/seisbench/seisbench/issues/96#issuecomment-1155158224
        if original_compatible == True:
            warnings.warn(
                "Using the non-conservative 'original' model, set `original_compatible='conservative' to use the more conservative model"
            )
            original_compatible = "non-conservative"

        if original_compatible:
            eps = 1e-7  # See Issue #96 - original models use tensorflow default epsilon of 1e-7
        else:
            eps = 1e-5
        self.original_compatible = original_compatible

        if original_compatible and in_samples != 6000:
            raise ValueError("original_compatible=True requires in_samples=6000.")

        self._phases = phases
        if phases is not None and len(phases) != classes:
            raise ValueError(
                f"Number of classes ({classes}) does not match number of phases ({len(phases)})."
            )

        # Parameters from EQTransformer repository
        self.filters = [
            8,
            16,
            16,
            32,
            32,
            64,
            64,
        ]  # Number of filters for the convolutions
        self.kernel_sizes = [11, 9, 7, 7, 5, 5, 3]  # Kernel sizes for the convolutions
        self.res_cnn_kernels = [3, 3, 3, 3, 2, 3, 2]

        # TODO: Add regularizers when training model
        # kernel_regularizer=keras.regularizers.l2(1e-6),
        # bias_regularizer=keras.regularizers.l1(1e-4),

        # Encoder stack
        self.encoder = Encoder(
            input_channels=self.in_channels,
            filters=self.filters,
            kernel_sizes=self.kernel_sizes,
            in_samples=self.in_samples,
        )

        # Res CNN Stack
        self.res_cnn_stack = ResCNNStack(
            kernel_sizes=self.res_cnn_kernels,
            filters=self.filters[-1],
            drop_rate=self.drop_rate,
        )

        # BiLSTM stack
        self.bi_lstm_stack = BiLSTMStack(
            blocks=self.lstm_blocks,
            input_size=self.filters[-1],
            drop_rate=self.drop_rate,
            original_compatible=original_compatible,
        )

        # Global attention - two transformers
        self.transformer_d0 = Transformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )
        self.transformer_d = Transformer(
            input_size=16, drop_rate=self.drop_rate, eps=eps
        )

        # Detection decoder and final Conv
        self.decoder_d = Decoder(
            input_channels=16,
            filters=self.filters[::-1],
            kernel_sizes=self.kernel_sizes[::-1],
            out_samples=in_samples,
            original_compatible=original_compatible,
        )
        self.conv_d = nn.Conv1d(
            in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
        )

        # Picking branches
        self.pick_lstms = []
        self.pick_attentions = []
        self.pick_decoders = []
        self.pick_convs = []
        self.dropout = nn.Dropout(drop_rate)

        for _ in range(self.classes):
            if original_compatible == "conservative":
                # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
                lstm = CustomLSTM(ActivationLSTMCell, 16, 16, bidirectional=False)
            else:
                lstm = nn.LSTM(16, 16, bidirectional=False)
            self.pick_lstms.append(lstm)

            attention = SeqSelfAttention(input_size=16, attention_width=3, eps=eps)
            self.pick_attentions.append(attention)

            decoder = Decoder(
                input_channels=16,
                filters=self.filters[::-1],
                kernel_sizes=self.kernel_sizes[::-1],
                out_samples=in_samples,
                original_compatible=original_compatible,
            )
            self.pick_decoders.append(decoder)

            conv = nn.Conv1d(
                in_channels=self.filters[0], out_channels=1, kernel_size=11, padding=5
            )
            self.pick_convs.append(conv)

        self.pick_lstms = nn.ModuleList(self.pick_lstms)
        self.pick_attentions = nn.ModuleList(self.pick_attentions)
        self.pick_decoders = nn.ModuleList(self.pick_decoders)
        self.pick_convs = nn.ModuleList(self.pick_convs)

    def forward(self, x, logits=False):
        assert x.ndim == 3
        assert x.shape[1:] == (self.in_channels, self.in_samples)

        # Shared encoder part
        x = self.encoder(x)
        x = self.res_cnn_stack(x)
        x = self.bi_lstm_stack(x)
        x, _ = self.transformer_d0(x)
        x, _ = self.transformer_d(x)

        # Detection part
        detection = self.decoder_d(x)
        if logits:
            detection = self.conv_d(detection)
        else:
            detection = torch.sigmoid(self.conv_d(detection))
        detection = torch.squeeze(detection, dim=1)  # Remove channel dimension

        outputs = [detection]

        # Pick parts
        for lstm, attention, decoder, conv in zip(
                self.pick_lstms, self.pick_attentions, self.pick_decoders, self.pick_convs
        ):
            px = x.permute(
                2, 0, 1
            )  # From batch, channels, sequence to sequence, batch, channels
            px = lstm(px)[0]
            px = self.dropout(px)
            px = px.permute(
                1, 2, 0
            )  # From sequence, batch, channels to batch, channels, sequence
            px, _ = attention(px)
            px = decoder(px)
            if logits:
                pred = conv(px)
            else:
                pred = torch.sigmoid(conv(px))
            pred = torch.squeeze(pred, dim=1)  # Remove channel dimension

            outputs.append(pred)

        return tuple(outputs)

        # Cosine taper (very short, i.e., only six samples on each side)
        tap = 0.5 * (1 + np.cos(np.linspace(np.pi, 2 * np.pi, 6)))
        window[:, :6] *= tap
        window[:, -6:] *= tap[::-1]

        return window


class Encoder(nn.Module):
    """
    Encoder stack
    """

    def __init__(self, input_channels, filters, kernel_sizes, in_samples):
        super().__init__()

        convs = []
        pools = []
        self.paddings = []
        for in_channels, out_channels, kernel_size in zip(
                [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

            # To be consistent with the behaviour in tensorflow,
            # padding needs to be added for odd numbers of input_samples
            padding = in_samples % 2

            # Padding for MaxPool1d needs to be handled manually to conform with tf padding
            self.paddings.append(padding)
            pools.append(nn.MaxPool1d(2, padding=0))
            in_samples = (in_samples + padding) // 2

        self.convs = nn.ModuleList(convs)
        self.pools = nn.ModuleList(pools)

    def forward(self, x):
        for conv, pool, padding in zip(self.convs, self.pools, self.paddings):
            x = torch.relu(conv(x))
            if padding != 0:
                # Only pad right, use -1e10 as negative infinity
                x = F.pad(x, (0, padding), "constant", -1e10)
            x = pool(x)

        return x


class Decoder(nn.Module):
    def __init__(
            self,
            input_channels,
            filters,
            kernel_sizes,
            out_samples,
            original_compatible=False,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.original_compatible = original_compatible

        # We need to trim off the final sample sometimes to get to the right number of output samples
        self.crops = []
        current_samples = out_samples
        for i, _ in enumerate(filters):
            padding = current_samples % 2
            current_samples = (current_samples + padding) // 2
            if padding == 1:
                self.crops.append(len(filters) - 1 - i)

        convs = []
        for in_channels, out_channels, kernel_size in zip(
                [input_channels] + filters[:-1], filters, kernel_sizes
        ):
            convs.append(
                nn.Conv1d(
                    in_channels, out_channels, kernel_size, padding=kernel_size // 2
                )
            )

        self.convs = nn.ModuleList(convs)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            x = self.upsample(x)

            if self.original_compatible:
                if i == 3:
                    x = x[:, :, 1:-1]
            else:
                if i in self.crops:
                    x = x[:, :, :-1]

            x = F.relu(conv(x))

        return x


class ResCNNStack(nn.Module):
    def __init__(self, kernel_sizes, filters, drop_rate):
        super().__init__()

        members = []
        for ker in kernel_sizes:
            members.append(ResCNNBlock(filters, ker, drop_rate))

        self.members = nn.ModuleList(members)

    def forward(self, x):
        for member in self.members:
            x = member(x)

        return x


class ResCNNBlock(nn.Module):
    def __init__(self, filters, ker, drop_rate):
        super().__init__()

        self.manual_padding = False
        if ker == 3:
            padding = 1
        else:
            # ker == 2
            # Manual padding emulate the padding in tensorflow
            self.manual_padding = True
            padding = 0

        self.dropout = SpatialDropout1d(drop_rate)

        self.norm1 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv1 = nn.Conv1d(filters, filters, ker, padding=padding)

        self.norm2 = nn.BatchNorm1d(filters, eps=1e-3)
        self.conv2 = nn.Conv1d(filters, filters, ker, padding=padding)

    def forward(self, x):
        y = self.norm1(x)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv1(y)

        y = self.norm2(y)
        y = F.relu(y)
        y = self.dropout(y)
        if self.manual_padding:
            y = F.pad(y, (0, 1), "constant", 0)
        y = self.conv2(y)

        return x + y


class BiLSTMStack(nn.Module):
    def __init__(
            self, blocks, input_size, drop_rate, hidden_size=16, original_compatible=False
    ):
        super().__init__()

        # First LSTM has a different input size as the subsequent ones
        self.members = nn.ModuleList(
            [
                BiLSTMBlock(
                    input_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
            ]
            + [
                BiLSTMBlock(
                    hidden_size,
                    hidden_size,
                    drop_rate,
                    original_compatible=original_compatible,
                )
                for _ in range(blocks - 1)
            ]
        )

    def forward(self, x):
        for member in self.members:
            x = member(x)
        return x


class BiLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, drop_rate, original_compatible=False):
        super().__init__()

        if original_compatible == "conservative":
            # The non-conservative model uses a sigmoid activiation as handled by the base nn.LSTM
            self.lstm = CustomLSTM(ActivationLSTMCell, input_size, hidden_size)
        elif original_compatible == "non-conservative":
            self.lstm = CustomLSTM(
                ActivationLSTMCell,
                input_size,
                hidden_size,
                gate_activation=torch.sigmoid,
            )
        else:
            self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv = nn.Conv1d(2 * hidden_size, hidden_size, 1)
        self.norm = nn.BatchNorm1d(hidden_size, eps=1e-3)

    def forward(self, x):
        x = x.permute(
            2, 0, 1
        )  # From batch, channels, sequence to sequence, batch, channels
        x = self.lstm(x)[0]
        x = self.dropout(x)
        x = x.permute(
            1, 2, 0
        )  # From sequence, batch, channels to batch, channels, sequence
        x = self.conv(x)
        x = self.norm(x)
        return x


class Transformer(nn.Module):
    def __init__(self, input_size, drop_rate, attention_width=None, eps=1e-5):
        super().__init__()

        self.attention = SeqSelfAttention(
            input_size, attention_width=attention_width, eps=eps
        )
        self.norm1 = LayerNormalization(input_size)
        self.ff = FeedForward(input_size, drop_rate)
        self.norm2 = LayerNormalization(input_size)

    def forward(self, x):
        y, weight = self.attention(x)
        y = x + y
        y = self.norm1(y)
        y2 = self.ff(y)
        y2 = y + y2
        y2 = self.norm2(y2)

        return y2, weight


class SeqSelfAttention(nn.Module):
    """
    Additive self attention
    """

    def __init__(self, input_size, units=32, attention_width=None, eps=1e-5):
        super().__init__()
        self.attention_width = attention_width

        self.Wx = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.Wt = nn.Parameter(uniform(-0.02, 0.02, input_size, units))
        self.bh = nn.Parameter(torch.zeros(units))

        self.Wa = nn.Parameter(uniform(-0.02, 0.02, units, 1))
        self.ba = nn.Parameter(torch.zeros(1))

        self.eps = eps

    def forward(self, x):
        # x.shape == (batch, channels, time)

        x = x.permute(0, 2, 1)  # to (batch, time, channels)

        q = torch.unsqueeze(
            torch.matmul(x, self.Wt), 2
        )  # Shape (batch, time, 1, channels)
        k = torch.unsqueeze(
            torch.matmul(x, self.Wx), 1
        )  # Shape (batch, 1, time, channels)

        h = torch.tanh(q + k + self.bh)

        # Emissions
        e = torch.squeeze(
            torch.matmul(h, self.Wa) + self.ba, -1
        )  # Shape (batch, time, time)

        # This is essentially softmax with an additional attention component.
        e = (
                e - torch.max(e, dim=-1, keepdim=True).values
        )  # In versions <= 0.2.1 e was incorrectly normalized by max(x)
        e = torch.exp(e)
        if self.attention_width is not None:
            lower = (
                    torch.arange(0, e.shape[1], device=e.device) - self.attention_width // 2
            )
            upper = lower + self.attention_width
            indices = torch.unsqueeze(torch.arange(0, e.shape[1], device=e.device), 1)
            mask = torch.logical_and(lower <= indices, indices < upper)
            e = torch.where(mask, e, torch.zeros_like(e))

        a = e / (torch.sum(e, dim=-1, keepdim=True) + self.eps)

        v = torch.matmul(a, x)

        v = v.permute(0, 2, 1)  # to (batch, channels, time)

        return v, a


def uniform(a, b, *args):
    return a + (b - a) * torch.rand(*args)


class LayerNormalization(nn.Module):
    def __init__(self, filters, eps=1e-14):
        super().__init__()

        gamma = torch.ones(filters, 1)
        self.gamma = nn.Parameter(gamma)
        beta = torch.zeros(filters, 1)
        self.beta = nn.Parameter(beta)
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        var = torch.mean((x - mean) ** 2, 1, keepdim=True) + self.eps
        std = torch.sqrt(var)
        outputs = (x - mean) / std

        outputs = outputs * self.gamma
        outputs = outputs + self.beta

        return outputs


class FeedForward(nn.Module):
    def __init__(self, io_size, drop_rate, hidden_size=128):
        super().__init__()

        self.lin1 = nn.Linear(io_size, hidden_size)
        self.lin2 = nn.Linear(hidden_size, io_size)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # To (batch, time, channel)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        x = self.lin2(x)
        x = x.permute(0, 2, 1)  # To (batch, channel, time)

        return x


class SpatialDropout1d(nn.Module):
    def __init__(self, drop_rate):
        super().__init__()

        self.drop_rate = drop_rate
        self.dropout = nn.Dropout2d(drop_rate)

    def forward(self, x):
        x = x.unsqueeze(dim=-1)  # Add fake dimension
        x = self.dropout(x)
        x = x.squeeze(dim=-1)  # Remove fake dimension
        return x
