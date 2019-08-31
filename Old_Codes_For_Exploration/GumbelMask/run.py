import os

# rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 150
# exp_dir = './tb_dir/cifar_exp/test_256bs_150epoch_layer3_open_10'
exp_dir = './tb_exp/gumbel_mask/test_256bs_150epoch_layer3_senet'
res = exp_dir + 'res.txt'
CUDA_DEVICES = 'CUDA_VISIBLE_DEVICES=0'


# Delete the previous file folder
# os.system('rm -r %s' % exp_dir)
for data in datasets:
    for depth in depths:
        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = CUDA_DEVICES + ' python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
              % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



