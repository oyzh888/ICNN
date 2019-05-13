import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 150
# exp_dir = '/data/ouyangzhihao/Exp/ICNN/Resnet_ICNN/tb_dir/cifar_exp/test_256bs_150epoch_layer3_mask1_inception3_baseline'
exp_dir = '/data/ouyangzhihao/Exp/ICNN/Resnet_ICNN/tb_dir/cifar_exp/vgg/test_256bs_150epoch_vgg_layer3_1vs-1'
res = exp_dir + 'res.txt'


# Delete the previous file folder
# os.system('rm -r %s' % exp_dir)
for data in datasets:
    for depth in depths:
        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
              % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)
