from torch import nn


class ConvBlock(nn.module):

    def __init__(self, in_dim=3, out_dim=3, kernel=5):
        super(ConvBlock, self).__init__()

        block = [nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=int((kernel - 1) / 2))]
        block += [nn.ReLU()]
        block += [nn.BatchNorm2d(out_dim)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class ToyNet(nn.Module):

    def __init__(self):
        super(ToyNet, self).__init__()

        input_channels = 3
        hidden_channels = 20
        output_channels = 3

        layers = [
            ConvBlock(input_channels, hidden_channels, 7),
            ConvBlock(hidden_channels, hidden_channels, 5),
            ConvBlock(hidden_channels, output_channels, 3),
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
