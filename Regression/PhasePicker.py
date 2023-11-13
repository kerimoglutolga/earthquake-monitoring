from torch.nn import MaxPool1d
from torch.nn import Conv1d
from torch.nn import Linear
from torch.nn import ReLU
from torch.nn import Flatten
from torch.nn import BatchNorm1d
from torch.nn import Sequential
import torch


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,\
        padding: int, bn_num_features: int, eps: float = 1e-5, momentum: float = 0.1,\
        mp_kernel_size: int = 10, stride: int = 3) -> None:

        self.out_channels = out_channels
        self

        super().__init__()
        self.convSequential = Sequential(
            Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding),
            ReLU(),
            BatchNorm1d(num_features=bn_num_features, eps=eps, momentum=momentum),
            MaxPool1d(kernel_size=mp_kernel_size, stride=stride)
        )
    
    def forward(self, x):
        self.inDimension = x.shape
        self.outDimension = []
        return self.convSequential(x)

class LinearBlock(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, eps: float = 1e-5, momentum : float = 0.1):
        super().__init__()
        self.linearSequential = Sequential(
            Linear(in_features=in_features, out_features=out_features),
            ReLU(),
            BatchNorm1d(num_features=out_features, eps=eps, momentum=momentum)
        )
    def forward(self, x):
        return self.linearSequential(x)


class CNNNet(torch.nn.Module):
    def __init__(self, num_channels=1):
        super(CNNNet, self).__init__()
        filter1 = 21
        filter2 = 15
        filter3 = 9
        filter4 = 5
        flattened = 
        linear1 = 256
        linear2 = 128
        linearOut = 2

        self.convBlock1 = ConvBlock(in_channels=1, out_channels=32,\
                                    kernel_size=filter1, padding=filter1//2,bn_num_features=32)
        self.convBlock2 = ConvBlock(in_channels=32, out_channels=32,\
                                    kernel_size=filter2, padding=filter2//2,bn_num_features=32)
        self.convBlock3 = ConvBlock(in_channels=32, out_channels=32,\
                                    kernel_size=filter3, padding=filter3//2,bn_num_features=32)
        self.convBlock3 = ConvBlock(in_channels=32, out_channels=32,\
                                    kernel_size=filter3, padding=filter3//2,bn_num_features=32)
        self.fcBlock1 = LinearBlock(in_features=flattened, out_features=linear1)
        self.fcBlock2 = LinearBlock(in_features=linear1, out_features=linear2)
        self.regression = Linear(in_features=linear2, out_features=linearOut)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)
        x = self.fcBlock1(x)
        x = self.fcBlock2(x)
        return self.regression(x)

net = CNNNet()

dataloader = DataLoader(concatenated_dataset, batch_size=64, shuffle=True, num_workers=4)

for signals, labels, snrs in dataloader: break

signals = signals.reshape(64,1,-1)

net(signals)

