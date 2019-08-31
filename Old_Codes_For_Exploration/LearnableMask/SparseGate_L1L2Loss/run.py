import os

# rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 150
# exp_dir = './tb_dir/cifar_exp/test_256bs_150epoch_layer3_open_10'
exp_dir = '../tb_dir/learnable_mask/test_256bs_200epoch_layer3_L21e-3'
res = exp_dir + 'res.txt'
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

for data in datasets:
    for depth in depths:
        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
              % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



