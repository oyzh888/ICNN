import os
import sys
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('home/zengyuyuan/ICNN')

from models import ResNet as resnet_cifar
import argparse
from torch.optim.lr_scheduler import MultiStepLR
from dataLoader import DataLoader
from summaries import TensorboardSummary
import numpy as np

# parameters setting
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--name', default='test_model', help='filename to output best model') #save output
parser.add_argument('--dataset', default='cifar-10',help="datasets")
parser.add_argument('--depth', default=20,type=int,help="depth of resnet model")
parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
parser.add_argument('--batch_size', default=64,type=int, help='batch size')
parser.add_argument('--epoch', default=200,type=int, help='epoch')
parser.add_argument('--exp_dir',default='./',help='dir for tensorboard')
parser.add_argument('--res', default='./result.txt', help="file to write best result")


args = parser.parse_args()

if os.path.exists(args.exp_dir):
    print ('Already exist and will continue training')
    # exit()
summary = TensorboardSummary(args.exp_dir)
tb_writer = summary.create_summary()

def one_hot(labels):
    one_hot_labels = torch.zeros(labels.shape[0],num_classes).scatter_(1,labels,1)
    return one_hot_labels

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()

    #best_model = model.state_dic()
    best_acc = 0.0
    best_train_acc = 0.0
    # Load unfinished model
    unfinished_model_path = os.path.join(args.exp_dir , 'unfinished_model.pt')
    if(os.path.exists(unfinished_model_path)):
        checkpoint = torch.load(unfinished_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
    else: epoch = 0

    while epoch < num_epochs:
        epoch_time = time.time()
        print('-'*10)
        print('Epoch {}/{}'.format(epoch,num_epochs-1))

        #each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0.0
            top5_corrects = 0.0

            # change tensor to variable(including some gradient info)
            # use variable.data to get the corresponding tensor
            for data in dataloaders[phase]:
                #782 batch,batch size= 64
                inputs,labels = data
                # print (inputs.shape)
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)
                #zero the parameter gradients
                optimizer.zero_grad()
                #forward
                outputs = model(inputs, labels, epoch)
                cur_loss = 0
                for cls in range(num_classes):
                    label = torch.full(size = labels.shape,fill_value=cls,dtype=torch.long).cuda()
                    binary_labels = (labels == label).float()
                    binary_labels = binary_labels.view(outputs[cls].shape)
                    loss = criterion(outputs[cls], binary_labels)
                    cur_loss += loss
                if phase == 'train':
                    cur_loss.backward()
                    optimizer.step()

                outputs = torch.cat(outputs,dim=1)
                outputs = torch.sigmoid(outputs)
                # print (outputs)
                _, preds = torch.max(outputs.data, 1)
                # print (preds)

                y = labels.data

                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = float(running_corrects) /dataset_sizes[phase]

            print('%s Loss: %.4f top1 Acc:%.4f'%(phase,epoch_loss,epoch_acc))
            if phase == 'train':
                tb_writer.add_scalar('train/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('train/acc_epoch', epoch_acc, epoch)
                if best_train_acc < epoch_acc:
                    best_train_acc = epoch_acc
            if phase == 'val':
                tb_writer.add_scalar('val/total_loss_epoch', epoch_loss, epoch)
                tb_writer.add_scalar('val/acc_epoch', epoch_acc, epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model.state_dict()
            cost_time = time.time() - epoch_time
            print('Epoch time cost {:.0f}m {:.0f}s'.format(cost_time // 60, cost_time % 60))
        # Save model periotically
        if(epoch % 2 == 0):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, os.path.join(args.exp_dir , 'unfinished_model.pt'))
        epoch += 1

    cost_time = time.time() - since
    print ('Training complete in {:.0f}m {:.0f}s'.format(cost_time//60,cost_time%60))
    print ('Best Train Acc is {:.4f}'.format(best_train_acc))
    print ('Best Val Acc is {:.4f}'.format(best_acc))
    model.load_state_dict(best_model)
    return model,cost_time,best_acc,best_train_acc


if __name__ == '__main__':
    print ('DataSets: '+args.dataset)
    print ('ResNet Depth: '+str(args.depth))
    loader = DataLoader(args.dataset,batch_size=args.batch_size)
    dataloaders,dataset_sizes = loader.load_data()
    num_classes = 10
    if args.dataset == 'cifar-10':
        num_classes = 10
    if args.dataset == 'cifar-100':
        num_classes = 100

    model = resnet_cifar(depth=args.depth, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, nesterov=True, weight_decay=1e-4)

    # define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    scheduler = MultiStepLR(optimizer, milestones=[args.epoch*0.4, args.epoch*0.6, args.epoch*0.8], gamma=0.1)

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        # patch_replication_callback(model)
        model = model.cuda()
    model,cost_time,best_acc,best_train_acc = train_model(model=model,
                                            optimizer=optimizer,
                                            criterion=criterion,
                                            scheduler=scheduler,
                                            num_epochs=args.epoch)

    exp_name = 'resnet%d dataset: %s batchsize: %d epoch: %d bestValAcc: %.4f bestTrainAcc: %.4f \n' % (
    args.depth, args.dataset,args.batch_size, args.epoch,best_acc,best_train_acc)

    os.system('rm ' + os.path.join(args.exp_dir , 'unfinished_model.pt') )
    torch.save(model.state_dict(), os.path.join(args.exp_dir , 'saved_model.pt'))
    with open(args.res,'a') as f:
        f.write(exp_name)
        f.close()

