import torch.nn as nn
import torch


class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 192, kernel_size=3, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc_layers = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.6),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x,labels,epoch):
        conv_features = self.features(x)

        # x_icnn = conv_features
        # label_filter_range = x_icnn.shape[1] / 10
        # if (epoch % 3 >= 0 and self.training):
        #     labels = labels * label_filter_range
        #     label_ones = torch.zeros(x_icnn.shape, device=x_icnn.device)
        #     for idx, la in enumerate(labels):
        #         label_ones[idx, la:la + label_filter_range, :, :] = 1.0
        #     label_ones = label_ones.to(x_icnn.device)
        #     x_icnn = x_icnn * label_ones

        conv_features = self.pooling(conv_features)
        flatten = conv_features.view(conv_features.size(0), -1)
        fc = self.fc_layers(flatten)
        print ('alex')

        # x_icnn = self.pooling(x_icnn)
        # flatten_icnn = x_icnn.view(x_icnn.size(0), -1)
        # fc_icnn = self.fc_layers(flatten_icnn)

        return fc