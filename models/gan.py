from grpc import GenericRpcHandler
import torch.nn as nn
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()

        # 4 x 83 x 32
        self.conv1 = nn.Conv2d(4, 64, kernel_size=(1, 1)) # 83 x 32
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(12, 1), stride=(12, 1), padding=(6, 0)) # 7x32 
        self.conv3 = nn.Conv2d(128, 128, kernel_size=(7, 1), stride=(7, 1), padding=(3, 0)) # 1x32
        self.conv4 = nn.Conv2d(128, 128, kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)) # 1 x 17
        self.conv5 = nn.Conv2d(128, 128, kernel_size=(1, 2), stride=(1, 2), padding=(0, 1)) # 1 x 9
        self.conv6 = nn.Conv2d(128, 256, kernel_size=(1, 4), stride=(1, 2), padding=(0, 1)) # 1x4
        self.conv7 = nn.Conv2d(256, 512, kernel_size=(1, 3), stride=(1, 2)) # 1 x 1

        self.flatten = nn.Flatten()

        self.dense = nn.Linear(512, 1024)
        self.output = nn.Linear(1024, 1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = F.leaky_relu(self.conv3(x))
        x = F.leaky_relu(self.conv4(x))
        x = F.leaky_relu(self.conv5(x))
        x = F.leaky_relu(self.conv6(x))
        x = F.leaky_relu(self.conv7(x))

        x = self.flatten(x)
        x = self.dense(x)

        return self.output(F.leaky_relu(x))


class Generator(nn.Module):
    def __init__(self, noise_size=512) -> None:
        super(Generator, self).__init__()
        self.noise_size = noise_size

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(512, 1, 1)) # 1 x 1
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=(1, 3), stride=(1, 2)) # 1 x 3
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=(1, 4), stride=(1, 2)) # 1 x 8
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=(1, 2), stride=(1, 2)) # 1 x 16
        self.deconv4 = nn.ConvTranspose2d(128, 128, kernel_size=(1, 2), stride=(1, 2)) # 1 x 32
        self.deconv5 = nn.ConvTranspose2d(128, 128, kernel_size=(7, 1), stride=(7, 1)) # 7 x 32
        self.deconv6 = nn.ConvTranspose2d(128, 128, kernel_size=(12, 1), stride=(12, 1), padding=(1, 0)) # 82 x 32
        self.deconv7 = nn.ConvTranspose2d(128, 64, kernel_size=(2, 1), stride=(1, 1)) # 83 x 32
        self.deconv8 = nn.ConvTranspose2d(64, 4, kernel_size=(1, 1), stride=(1, 1)) # 83 x 32


        self.batch_n1 = nn.BatchNorm2d(512)
        self.batch_n2 = nn.BatchNorm2d(256)
        self.batch_n3 = nn.BatchNorm2d(128)
        self.batch_n4 = nn.BatchNorm2d(128)
        self.batch_n5 = nn.BatchNorm2d(128)
        self.batch_n6 = nn.BatchNorm2d(128)
        self.batch_n7 = nn.BatchNorm2d(64)

    def forward(self, x):
        x = self.unflatten(x)
        x = F.relu(self.batch_n1(self.deconv1(x)))
        x = F.relu(self.batch_n2(self.deconv2(x)))
        x = F.relu(self.batch_n3(self.deconv3(x)))
        x = F.relu(self.batch_n4(self.deconv4(x)))
        x = F.relu(self.batch_n5(self.deconv5(x)))
        x = F.relu(self.batch_n6(self.deconv6(x)))
        x = F.relu(self.batch_n7(self.deconv7(x)))
        x = F.tanh(self.deconv8(x))
        return x
