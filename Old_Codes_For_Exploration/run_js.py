import os

rlaunch = 'rlaunch --cpu=4 --memory=4096 --gpu=1 --preemptible=no '
datasets = ['cifar-10']
depths = [20]
gpu_id = '0'
batchsize = 256
epoch = 50
exp_dir = './tb_dir/cifar_exp/test_debug'
res = exp_dir + 'res.txt'
# is_rlaunch = True
is_rlaunch = False

for data in datasets:
    for depth in depths:
        if(is_rlaunch):
            cmd = rlaunch + '-- python3 ./train_js.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
                                    %(data,depth,res,gpu_id,batchsize,epoch,exp_dir)
        else:
            cmd = 'python3 ./train_js.py --dataset %s --depth %d --res %s --gpu-ids %s --batch_size %d --epoch %d --exp_dir %s' \
                  % (data, depth, res, gpu_id, batchsize, epoch, exp_dir)
        os.system(cmd)



