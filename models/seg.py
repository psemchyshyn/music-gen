import torch
import torch.nn as nn
import torch.nn.functional as F


class MusicSeg(nn.Module):
    def __init__(self):
        super(MusicSeg, self).__init__()
        # Encoding layers
        self.enc_conv1 = nn.Conv2d(in_channels = 4, out_channels = 64, kernel_size = (4, 4), stride = (4, 4)) # 20x9
        self.enc_conv2 = nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = (4, 4), stride = (4, 4)) # 5x2
        self.enc_conv3 = nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = (5, 2)) # 1x1 

        self.enc_batch1 = nn.BatchNorm2d(64)
        self.enc_batch2 = nn.BatchNorm2d(128)
        self.enc_batch3 = nn.BatchNorm2d(256)

        self.dropout = nn.Dropout(0.4)
        # Decoding layers
        self.dec_conv1 = nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = (5, 2)) # 5 x 2
        self.dec_conv2 = nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = (4, 4), stride = (4, 4), output_padding=(1, 0)) # 21x8
        self.dec_conv3 = nn.ConvTranspose2d(in_channels = 64, out_channels = 4, kernel_size = (3, 4), stride = (4, 4)) # 83*32

        self.dec_batch3 = nn.BatchNorm2d(64)
        self.dec_batch2 = nn.BatchNorm2d(128)


    def forward(self, x):
        x = F.relu(self.enc_batch1(self.enc_conv1(x)))
        x = self.dropout(x)
        x = F.relu(self.enc_batch2(self.enc_conv2(x)))
        x = self.dropout(x)
        x = F.relu(self.enc_batch3(self.enc_conv3(x)))

        x = F.relu(self.dec_batch2(self.dec_conv1(x)))
        x = F.relu(self.dec_batch3(self.dec_conv2(x)))
        x = F.relu(self.dec_conv3(x))

        return x