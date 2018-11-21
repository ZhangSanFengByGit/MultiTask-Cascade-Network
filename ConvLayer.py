import torch

class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1):
        super(ConvLayer, self).__init__()
        self.reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(self.reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        o = self.reflection_pad(x)
        o = self.conv2d(o)

        return o