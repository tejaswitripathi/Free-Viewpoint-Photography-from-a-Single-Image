import torch
import torch.nn as nn
import torch.nn.functional as F


def double_convolution(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    Gaussian-parameter U-Net for Milestone 5A.

    Input channels:
        RGB        3
        depth      1
        target dx  1
        target dy  1
        target dz  1
        target yaw 1
        target pitch 1

    Suggested in_channels = 9

    Output channels:
        radius_delta   1
        opacity        1
        color_residual 3

    Total out_channels = 5

    Interpretation:
        radius_multiplier = exp(clamped radius_delta)
        opacity = sigmoid(opacity_logit)
        color = clamp(input_rgb + tanh(color_residual) * 0.1, 0, 1)
    """

    def __init__(self, in_channels=9, base_channels=32):
        super().__init__()

        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_convolution_1 = double_convolution(in_channels, base_channels)
        self.down_convolution_2 = double_convolution(base_channels, base_channels * 2)
        self.down_convolution_3 = double_convolution(base_channels * 2, base_channels * 4)
        self.down_convolution_4 = double_convolution(base_channels * 4, base_channels * 8)
        self.down_convolution_5 = double_convolution(base_channels * 8, base_channels * 16)

        self.up_transpose_1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        self.up_convolution_1 = double_convolution(base_channels * 16, base_channels * 8)

        self.up_transpose_2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.up_convolution_2 = double_convolution(base_channels * 8, base_channels * 4)

        self.up_transpose_3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.up_convolution_3 = double_convolution(base_channels * 4, base_channels * 2)

        self.up_transpose_4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.up_convolution_4 = double_convolution(base_channels * 2, base_channels)

        self.out = nn.Conv2d(base_channels, 5, kernel_size=1)

    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)

        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)

        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)

        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)

        down_9 = self.down_convolution_5(down_8)

        up_1 = self.up_transpose_1(down_9)
        x = self.up_convolution_1(torch.cat([down_7, up_1], dim=1))

        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], dim=1))

        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], dim=1))

        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], dim=1))

        raw = self.out(x)

        radius_delta = torch.clamp(raw[:, 0:1], -2.0, 2.0)
        opacity = torch.sigmoid(raw[:, 1:2])
        color_residual = 0.1 * torch.tanh(raw[:, 2:5])

        return {
            "radius_delta": radius_delta,
            "opacity": opacity,
            "color_residual": color_residual,
            "raw": raw,
        }


if __name__ == "__main__":
    input_tensor = torch.rand((1, 9, 256, 256))
    model = UNet(in_channels=9, base_channels=32)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"{total_params:,} total parameters.")
    print(f"{trainable_params:,} trainable parameters.")

    outputs = model(input_tensor)
    for k, v in outputs.items():
        print(k, v.shape)