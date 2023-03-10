from misc.utils import *

class audnet(Cell):
    def __init__(self, cfg):
        super(audnet, self).__init__()
        self.conv1 = Conv2d(1, 64, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn1 = BatchNorm2d(64)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=(1,3),stride=(1,3))

        self.conv2 = Conv2d(64, 192, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn2 = BatchNorm2d(192)
        self.relu2 = ReLU()
        # self.pool2 = MaxPool2d(kernel_size=(1,3))

        self.conv3 = Conv2d(192, 384, kernel_size=(3,3), stride=(2,1), padding=0)
        self.bn3 = BatchNorm2d(384)
        self.relu3 = ReLU()

        self.conv4 = Conv2d(384, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn4 = BatchNorm2d(256)
        self.relu4 = ReLU()

        self.conv5 = Conv2d(256, 256, kernel_size=(3,3), stride=(2,2), padding=0)
        self.bn5 = BatchNorm2d(256)
        self.relu5 = ReLU()
        self.pool5 = MaxPool2d(kernel_size=(2,2),stride=(2,2))

        self.conv6 = Conv2d(256, cfg.output_dim, kernel_size=(3,2), padding=0)
        self.bn6 = BatchNorm2d(cfg.output_dim)
        self.relu6 = ReLU()
        #初始化方法不同，torch1.7.1为均匀分布，mindspore weight为标准正态分布，bias为0
        self.fc = Dense(cfg.output_dim, cfg.output_dim) 

    def forward(self, x):  # [bs,1,257,90]
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.pool5(self.relu5(self.bn5(self.conv5(x))))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = x.squeeze()
        out = self.fc(x)
        return out