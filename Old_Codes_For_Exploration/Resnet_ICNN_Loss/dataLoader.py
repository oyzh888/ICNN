import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

class DataLoader():
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def load_data(self):
        data_dir = '/home/ouyangzhihao/.keras/datasets'
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
            ])
        }
        if self.dataset == 'cifar-10':
            data_train = datasets.CIFAR10(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR10(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)
        if self.dataset == 'cifar-100':
            data_train = datasets.CIFAR100(root=data_dir,
                                          transform=data_transforms['train'],
                                          train=True,
                                          download=True)

            data_test = datasets.CIFAR100(root=data_dir,
                                         transform=data_transforms['val'],
                                         train=False,
                                         download=True)

        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size,
                                                           shuffle=True,
                                                           num_workers=2)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size,
                                                         shuffle=False,
                                                         num_workers=2)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders,dataset_sizes




