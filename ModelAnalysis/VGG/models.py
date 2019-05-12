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

    def forward(self, x, labels, epoch):
        x = self.features(x)

        #ICNN
        # 强制mask掉class1 的filters
        label_filter_range = int(x.shape[1] / 10)
        labels = labels * label_filter_range
        label_ones = torch.zeros(x.shape, device=x.device) + 1.0

        label_ones[:, 0 * label_filter_range:1 * label_filter_range, :, :] = 1.0
        # label_ones[:, 4 * label_filter_range:5 * label_filter_range, :, :] = 0
        # label_ones[:10*label_filter_range:,:,:] = 0

        # random filters drop
        # random_filter_indices = torch.from_numpy(np.random.randint(low=0, high=x.shape[1], size=label_filter_range))
        # label_ones[:, random_filter_indices, :, :] = 0
        x = x * label_ones

        out = x
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
