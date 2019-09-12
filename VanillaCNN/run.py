import os

datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 150
exp_dir = './tb_dir/cifar_exp/convNet_256bs_150epoch_icnnreg'
res = exp_dir + 'res.txt'
CUDA = 'CUDA_VISIBLE_DEVICES=2 '


# Delete the previous file folder
# os.system('rm -r %s' % exp_dir)
for data in datasets:
    for depth in depths:
        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = CUDA + 'python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
              % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



