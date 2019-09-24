import os

import matplotlib
import numpy as np
import torch
from models import ResNet as resnet_cifar
from two_step_model import ResNet as icnn_resnet_cifar
from torch.autograd import Variable
from dataLoader import DataLoader

from sklearn.cluster import KMeans
# from models import ResNet as resnet_cifar
matplotlib.use('Agg')
import matplotlib.pyplot as plt    # 绘图库
from collections import Counter
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import pca

print('Initialization')
print('Init Finished!')

args_depth = 20
num_classes = 10
args_dataset = 'cifar-10'
args_batch_size = 256
# PATH = './tb_dir/cifar_exp/conv_classifier'
# PATH = './tb_dir/mnist_exp/icnn'
# PATH = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/learnable_mask_baseline/TwoStep_Algorithm1_Clip_L21_LimitChannelClass'
# PATH = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/reproduce_8_20/alg1_9_2/TwoStep_Algorithm1_Clip_L11e_3_LimitChannelClass1'
# PATH = '/data/zengyuyuan/Model/ICNN/NaiveCNN_baseline'
PATH = '/data/zengyuyuan/Model/ICNN/reproduce_8_20/TwoStep_Algorithm1_Clip_L11e_3_LimitChannelClass'
# PATH ='/home/zengyuyuan/ICNN/tb_dir/mnist_exp/baseline'
model_path = os.path.join(PATH, 'saved_model.pt')

# model = resnet_cifar(depth=args_depth, num_classes=num_classes)
model = icnn_resnet_cifar(depth=args_depth, num_classes=num_classes)
model = model.cuda()
# model = torch.nn.DataParallel(model)
# model = torch.load(model_path)
checkpoint = torch.load(model_path)
state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint.items()}
model.load_state_dict(state_dict)
model.eval()
print('Successfully Load Model: ', os.path.basename(model_path))

# 该数组记录中间层结果
features_blobs = []
# 该函数由register_forward_hook调用，类似于event handler，当resnet前向传播时记录所需中间层结果
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
# 需要输出的中间层名称，名称为resnet_for_vis的__init__函数中声明的。
finalconv_name = 'layer3'
# print(model._modules)
# model._modules.layer3.register_forward_hook(hook_feature)
# model._modules['module'].layer3.register_forward_hook(hook_feature)
model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# Data Loader
loader = DataLoader(args_dataset,batch_size=args_batch_size)
dataloaders,dataset_sizes = loader.load_data()
labels_name = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
colors = ['b','g','r','k','c','m','y','#e24fff','#524C90','#845868']

def visulize(train_data,labels):
    print ('PCA Embedding')
    tsne = TSNE(n_components=2)
    embed_feature = []
    batch_size = 10000
    slices = 0
    while slices + batch_size <= len(train_data):
        print ('processing %d/%d'%(slices,len(train_data)))
        tsne.fit_transform(train_data[slices:slices+batch_size])
        embed_feature.append(tsne.embedding_)
        slices = slices + batch_size
    if slices < len(train_data):
        tsne.fit_transform(train_data[slices:])
        embed_feature.append(tsne.embedding_)

    embedding = np.concatenate(embed_feature,axis=0)
    print (embedding.shape)

    # PCA = pca.PCA(n_components=2).fit(train_data)
    # embedding = PCA.transform(train_data)
    print ('Visulize Embedding')
    fig = plt.figure()
    for cls in range(num_classes):
        index = np.array([i for i, x in labels if x == cls], dtype=int)
        print ('class %d: %d'%(cls,len(index)))
        # candidates = labels[index]
        candidates = embedding[index]
        print(candidates.shape)
        plt.scatter(candidates[:,0], candidates[:,1], label=labels_name[cls], color=colors[cls], s=0.5)
        plt.legend()
        # for num,i in enumerate(index):
        #     plt.scatter(embedding[i][0],embedding[i][1],label=labels_name[cls],color=colors[cls],s=0.1)
        # plt.scatter(centroid[i][0], embedding[i][1], marker='x', color=colors[cls], linewidths=12)
    # plt.legend()
    plt.title('Naive CNN')
    plt.savefig('figures/cifar_baseline_stne_scatter.png')

def visulizeCentroid(centroid):
    PCA = pca.PCA(n_components=2).fit(centroid)
    embedding = PCA.transform(centroid)
    for cls in range(num_classes):
        plt.scatter(embedding[cls][0], embedding[cls][1], marker='x',color=colors[cls], s=12)
    plt.savefig('icnn_centroid.png')

def normalize(centoids):
    L2_norm = np.sqrt(np.sum(np.square(centoids),-1))
    centoids = centoids / np.reshape(L2_norm,(num_classes,1))
    return centoids


def train_kmeans(train_data,labels):
    print ('training_kmeans')
    print ('Size of Train Data,',len(train_data))
    cluster = KMeans(n_clusters=10,max_iter=500)
    kmeans = cluster.fit(train_data)
    centoids = kmeans.cluster_centers_
    pred_labels = list(enumerate(kmeans.labels_))
    # print (pred_labels)
    pairs = {}
    labels = np.array(labels)
    print (kmeans.inertia_)

    dis = 0
    centoids = normalize(centoids)
    for i in range(num_classes):
        for j in range(num_classes):
            dis += np.sqrt(np.sum((centoids[i] - centoids[j]) * (centoids[i] - centoids[j])))
    dis = dis / (num_classes * num_classes)
    print ('Centoids Distance',dis)

    # visulize(train_data,pred_labels)
    # visulizeCentroid(centoids)

    # fig = plt.figure(figsize=(6,3))
    # ax = fig.add_subplot(111)
    # im = ax.imshow(centoids,cmap=plt.cm.hot_r)
    # plt.colorbar(im,orientation='horizontal')
    # plt.savefig('baseline_heatmap.png')

    for cls in range(num_classes):
        index = np.array([i for i,x in pred_labels if x == cls],dtype=int)
        candidates = labels[index]
        # cluster_feat = np.mean(train_data[index],dim=-1)
        count = Counter(candidates)
        sorted_count = sorted(count.items(),key = lambda item:item[1],reverse=True)
        center_label = sorted_count[0][0]
        pairs[cls] = center_label
    return kmeans,pairs

def kmeans_pred(kmeans,pairs,test_data,labels):
    print ('Size of Test Data,', len(test_data))
    preds = kmeans.predict(test_data)
    true_labels = []
    print (pairs)
    for i in preds:
        true_labels.append(pairs[i])
    print (true_labels[:10])
    print (labels[:10])
    # visulize(test_data, list(enumerate(true_labels)))
    correts = np.array(np.array(true_labels) == np.array(labels),dtype=int)
    acc = np.sum(correts)/len(labels)

    SW = np.zeros((64,64))
    SB = np.zeros((64,64))
    M = np.mean(test_data,0)
    # M = M / np.linalg.norm(M,axis=-1,keepdims=True)
    N = len(labels)
    print(N)
    for i in range(10):
        # index = np.where(preds==i)
        index = np.where(np.array(labels) == i)
        features_cls = torch.Tensor(test_data)[index[0]]
        features_cls = features_cls.numpy()
        # print(features_cls.shape)
        n = features_cls.shape[0]
        # print(n)
        mean_features = np.mean(features_cls,0,keepdims=True)
        # mean_features = mean_features / np.linalg.norm(mean_features,axis=-1,keepdims=True)
        # features_cls = features_cls / np.linalg.norm(features_cls,axis=-1,keepdims=True)
        intra_dis = features_cls - mean_features
        sj = np.zeros((64,64))
        for j in range(n):
            xj = np.reshape(intra_dis[j],(64,1))
            T = xj.dot(xj.T)
            sj += T
            # print((intra_dis[j].T).dot(intra_dis[j]))
            # print(T.shape)
        inter_dis = np.reshape(mean_features - M,(64,1))
        mj = (inter_dis).dot(inter_dis.T)
        SW += sj / N
        SB += n/float(N) * mj
        # print(n/float(N) * mj)
    S = SW + SB
    S_ves = np.linalg.inv(S)
    # print(SW.shape)
    # print(SW)
    print(SB.shape)
    print(SB)
    # print(S.shape)
    print('Intra Schatter',np.linalg.det(SW))
    print('Inter Schatter',np.linalg.det(SB))
    print('Total/Intra',np.linalg.det(S_ves.dot(SW)))
    print('Total/Inter',np.linalg.det(S_ves.dot(SB)))
    # print('Total/Intra', np.linalg.det(np.linalg.lstsq(S,SW)[0]))
    # print('Total/Inter', np.linalg.det(np.linalg.lstsq(S,SB)[0]))
    return acc

def printF(i, total=100):
    i = int( i / total * 100) + 1
    total = 100
    k = i + 1
    str_ = '>'*i + '' ''*(total-k)
    sys.stdout.write('\r'+str_+'[%s%%]'%(i+1))
    sys.stdout.flush()
    if(i >= total -1): print()

def generate_features(phase):
    true_labels = []
    avgpool = torch.nn.AvgPool2d(8)
    features = []
    features_blobs.clear()
    data_len = len(dataloaders[phase])
    for idx, data in enumerate(dataloaders[phase]):
        printF(idx, data_len)
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        # forward
        outputs = model(inputs, labels, epoch)

        y = labels.data
        true_labels.extend(y.cpu().numpy())
    for feat in features_blobs:
        x = avgpool(torch.FloatTensor(feat)).view(feat.shape[0],-1)
        features.extend(x.cpu().numpy())
    return features,true_labels




use_gpu = True
epoch = 1
phase = 'train'

train_x,train_y = generate_features('train')
kmeans,pairs = train_kmeans(train_x,train_y)
test_x,test_y = generate_features('val')
acc = kmeans_pred(kmeans,pairs,test_x,test_y)
print ('Cluster Acc:%.4f'%(acc))
