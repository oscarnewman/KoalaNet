import torch
from torch import nn
from PIL import Image
import numpy as np
from torchvision import transforms


class ConvBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, kernel=5):
        super(ConvBlock, self).__init__()

        block = [nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=int((kernel - 1) / 2))]
        block += [nn.LeakyReLU()]
        block += [nn.MaxPool2d(2)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, kernel=5):
        super(DeconvBlock, self).__init__()

        block = [nn.ConvTranspose2d(in_dim, out_dim, kernel, stride=1, padding=int((kernel - 1) / 2))]
        block += [nn.Conv2d(out_dim, out_dim, kernel)]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        kernel_size = 3

        layers = [
            ConvBlock(in_dim=3, out_dim=16, kernel=kernel_size),
            # ConvBlock(in_dim=16, out_dim=1, kernel=kernel_size),
            # DeconvBlock(in_dim=1, out_dim=16, kernel=kernel_size),
            DeconvBlock(in_dim=16, out_dim=3, kernel=kernel_size),
            nn.ReLU()
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, images):
        return self.net(images)


def main():
    image: torch.Tensor = transforms.ToTensor()(Image.open('mtn.jpg'))

    model = Network()

    output = model(image.reshape(1, *image.shape))[0]

    output = output / torch.max(output)

    image_out: Image = transforms.ToPILImage()(output)
    image_out.show()


if __name__ == '__main__':
    main()
