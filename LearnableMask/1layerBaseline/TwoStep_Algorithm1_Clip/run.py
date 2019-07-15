import os

# rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 150
# exp_dir = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/learnable_mask_baseline/Debug'
exp_dir = '/data/ouyangzhihao/Exp/ICNN/LearnableMask/tb_dir/learnable_mask_baseline/TwoStep_Algorithm1_Clip_Lmax_LimitChannelClass'
res = exp_dir + 'res.txt'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# os.system('rm -r ' + exp_dir)
print('run ', exp_dir.split('/')[-1])
for data in datasets:
    for depth in depths:
        # cmd = rlaunch + '-- python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
        #                         %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        cmd = 'python3 ./train.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
              % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



