'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)
        self.labels = None
        self.epoch = None

    def forward(self, x, labels, epoch):

        self.labels = labels
        self.epoch = epoch

        out = self.features(x)
        #ICNN
        x_icnn = out
        label_filter_range = x_icnn.shape[1] / 10
        if (epoch % 3 >= 0 and self.training):
            labels = labels * label_filter_range
            label_ones = torch.zeros(x_icnn.shape, device=x_icnn.device) + 1.0
            for idx, la in enumerate(labels):
                label_ones[idx, la:la + label_filter_range, :, :] = 100.0
            label_ones = label_ones.to(x_icnn.device)
            x_icnn = x_icnn * label_ones
            # out = x_icnn
        x_icnn = x_icnn.view(out.size(0), -1)
        x_icnn = self.classifier(x_icnn)

        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out, x_icnn

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for idx, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                if idx == 11:
                    layers += [ICNNMask(labels=self.labels, epoch=self.epoch)]

            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

class ICNNMask(nn.Modele):

    def __init__(self, labels, epoch):
        super(nn.Modele, self).__init__()
        self.labels = labels
        self.epoch = epoch

    def forward(self, input):
        return input * 1.0

def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
