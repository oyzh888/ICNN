import os
import time
import torch
import torch.nn as nn
from torch.autograd import Variable

from models import ResNet as resnet_cifar
import pandas as pd
import argparse
import csv
from torch.optim.lr_scheduler import MultiStepLR
from dataLoader import DataLoader
from summaries import TensorboardSummary

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
    print ('Already exist')
    # exit()
summary = TensorboardSummary(args.exp_dir)
tb_writer = summary.create_summary()

def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
    since = time.time()

    #best_model = model.state_dic()
    best_acc = 0.0
    best_train_acc = 0.0

    for epoch in range(num_epochs):
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
                outputs = model(inputs)
                # loss = criterion(outputs, labels)
                # print(loss)

                # import ipdb; ipdb.set_trace()
                def softmax(x):
                    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)
                def nl(input, target):
                    return -input[range(target.shape[0]), target].log().mean()
                def nl_js(input, target):
                    losses = -input[range(target.shape[0]), target].log()

                    # print(losses.shape)
                    # import ipdb; ipdb.set_trace()

                    group = 8
                    num_of_items_group = int(losses.shape[0] / group)

                    global_average_f = losses.mean()
                    group_mean = []
                    group_var = []
                    for i in range(group):
                        group_mean.append(losses[i * num_of_items_group:(i + 1) * num_of_items_group].mean())
                        group_var.append(losses[i * num_of_items_group:(i + 1) * num_of_items_group].var())

                    gama = 0
                    for i in range(0,group):
                        gama += 1 / ( num_of_items_group / torch.pow(group_var[i], 2)  * torch.pow(group_mean[i] - global_average_f, 2 ) )
                        print(( num_of_items_group / torch.pow(group_var[i], 2)  * torch.pow(group_mean[i] - global_average_f, 2 ) ))
                    gama = (group - 3 ) * gama
                    # import ipdb; ipdb.set_trace()
                    if(gama > group):
                        gama = group
                    print('gama',gama)

                    js_loss = 0
                    for i in range(group):
                        js_loss += global_average_f + (1 - gama) * (group_mean[i] - global_average_f)

                    print('js_loss',js_loss)
                    print('*****'*10)

                    return js_loss

                pred = softmax(outputs)
                # loss = nl(pred, labels)
                loss = nl_js(pred, labels)


                # loss
                # print(loss2)
                # print(loss2.shape)
                # loss =


                _, preds = torch.max(outputs.data, 1)
                # _,top5_preds = torch.topk(outputs.data,k=5,dim=1)
                # print ('group loss:',group_loss[0])

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                y = labels.data
                batch_size = labels.data.shape[0]
                # print(y.resize_(batch_size,1))
                running_loss += loss.item()
                running_corrects += torch.sum(preds == y)
                # top5_corrects += torch.sum(top5_preds == y.resize_(batch_size,1))

            epoch_loss = running_loss /dataset_sizes[phase]
            epoch_acc = float(running_corrects) /dataset_sizes[phase]
            # top5_acc = top5_corrects /dataset_sizes[phase]

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
    criterion = nn.CrossEntropyLoss()
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
    with open(args.res,'a') as f:
        f.write(exp_name)
        f.close()

