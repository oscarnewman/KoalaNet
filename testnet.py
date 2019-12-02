import torch
from torch import nn
from PIL import Image
from torchsummary import summary
from torchvision import transforms
import math


class ConvBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, kernel=4, pool=True):
        super(ConvBlock, self).__init__()

        block = [
            nn.Conv2d(in_dim, out_dim, kernel, stride=1, padding=kernel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(out_dim, out_dim, kernel, stride=2 if pool else 1, padding=kernel // 2),
            nn.LeakyReLU(),
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class DeconvBlock(nn.Module):

    def __init__(self, in_dim=3, out_dim=3, kernel=4):
        super(DeconvBlock, self).__init__()

        self.transpose = nn.ConvTranspose2d(in_dim, out_dim, kernel, stride=2, padding=kernel // 2)
        block = [
            nn.Conv2d(out_dim, out_dim, kernel, stride=1, padding=kernel // 2),
            nn.LeakyReLU(),
            nn.Conv2d(out_dim, out_dim, kernel, stride=1, padding=kernel // 2),
            nn.LeakyReLU(),
        ]

        self.block = nn.Sequential(*block)

    def forward(self, x, add=None):
        x = self.transpose(x, output_size=add.shape if add is not None else None)
        out = self.block(x)
        return out if add is None else torch.add(out, add)


class DownsampleBlock(nn.Module):
    def __init__(self):
        super(DownsampleBlock, self).__init__()

        self.conv32 = ConvBlock(in_dim=3, out_dim=32)
        self.conv64 = ConvBlock(in_dim=32, out_dim=64)
        self.conv128 = ConvBlock(in_dim=64, out_dim=128)
        self.conv256 = ConvBlock(in_dim=128, out_dim=256)
        self.conv512 = ConvBlock(in_dim=256, out_dim=512)

    def forward(self, x) -> [torch.Tensor]:
        conv32 = self.conv32(x)
        print(conv32.shape)
        conv64 = self.conv64(conv32)
        print(conv64.shape)
        conv128 = self.conv128(conv64)
        print(conv128.shape)
        conv256 = self.conv256(conv128)
        print(conv256.shape)
        conv512 = self.conv512(conv256)
        print(conv512.shape)

        return [conv512, conv256, conv128, conv64, conv32]


class UpsampleBlock(nn.Module):
    def __init__(self):
        super(UpsampleBlock, self).__init__()

        self.conv512 = DeconvBlock(in_dim=512, out_dim=256)
        self.conv256 = DeconvBlock(in_dim=256, out_dim=128)
        self.conv128 = DeconvBlock(in_dim=128, out_dim=64)
        self.conv64 = DeconvBlock(in_dim=64, out_dim=32)

    def forward(self, samples):
        u512, u256, u128, u64, u32 = samples

        print("UPSAMPLING")

        x = self.conv512(u512, add=u256)
        print(x.shape)
        x = self.conv256(x, add=u128)
        print(x.shape)
        x = self.conv128(x, add=u64)
        print(x.shape)
        x = self.conv64(x, add=u32)
        print(x.shape)

        return x


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()

        layers = [
            DownsampleBlock(),
            UpsampleBlock(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
        ]

        self.net = nn.Sequential(*layers)

    def forward(self, images):
        return self.net(images)


def main():
    image: torch.Tensor = transforms.ToTensor()(Image.open('mtn.jpg'))

    model = Network()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.cuda()
        image = image.cuda()

    test = torch.rand((1, 3, 512, 512))
    transforms.ToPILImage()(test[0]).show()

    with torch.no_grad():
        output = model(image.reshape(1, *image.shape))[0]

    output = output / torch.max(output)

    image_out: Image = transforms.ToPILImage()(output)
    image_out.show()


if __name__ == '__main__':
    main()
