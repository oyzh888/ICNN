import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch

class DataLoader():
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def load_data(self):
        data_dir = '/home/zengyuyuan/data/CIFAR10'
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
        if self.dataset == 'mnist':
            data_dir = '/home/zengyuyuan/data/MNIST'
            mnist_transforms = {
                'train': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307, ], [0.3081])
                ]),
                'val': transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize([0.1307, ], [0.3081])
                ])
            }
            data_train = datasets.MNIST(root=data_dir,
                                        transform=mnist_transforms['train'],
                                        train=True,
                                        download=True)
            data_test = datasets.MNIST(root=data_dir,
                                       transform=mnist_transforms['val'],
                                       train=False,
                                       download=True)

        image_datasets = {'train': data_train, 'val': data_test}
        # change list to Tensor as the input of the models
        dataloaders = {}
        dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'],
                                                           batch_size=self.batch_size,
                                                           shuffle=True)
        dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'],
                                                         batch_size=self.batch_size,
                                                         shuffle=False)

        dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

        return dataloaders,dataset_sizes




