import torch.nn as nn
import torch as torch
import horovod.torch as hvd
import os, glob
import csv
import numpy as np
import hvd_util as hu
import pdb

from larcv import larcv_interface
from collections import OrderedDict

hvd.init()
seed = 314159
print("hvd.size() is: " + str(hvd.size()))
print("hvd.local_rank() is: " + str(hvd.local_rank()))
print("hvd.rank() is: " + str(hvd.rank()))

print("Number of gpus per rank {:d}".format(torch.cuda.device_count()))
# Horovod: pin GPU to local rank.
#torch.cuda.set_device(hvd.local_rank())

os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
torch.cuda.manual_seed(seed)

global_Nclass = 3 # bkgd, 0vbb, 2vbb
global_n_iterations_per_epoch = 1000
global_n_iterations_val = 4
global_n_epochs = 10
global_batch_size = 64  ## Can be at least 32, but need this many files to pick evts from in DataLoader
vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
nvox = int(1500/vox) # num bins in x,y dimension 
nvoxz = int(3000/vox) # num bins in z dimension 
voxels = (int(1500/vox),int(1500/vox),int(3000/vox) ) # These are 1x1x1cm^3 voxels

mode = 'train'
aux_mode ='test'
max_voxels= '1000'
producer='sparse3d_voxels_group'


import torch.optim as optim
import torch.nn.functional as F
import sparseconvnet as scn

dimension = 3
nPlanes = 1

def larcvsparse_to_scnsparse_3d(input_array):
    # This format converts the larcv sparse format to
    # the tuple format required for sparseconvnet

    # First, we can split off the features (which is the pixel value)
    # and the indexes (which is everythin else)

    n_dims = input_array.shape[-1]
    split_tensors = np.split(input_array, n_dims, axis=-1)

    # To map out the non_zero locations now is easy:
    non_zero_inds = np.where(split_tensors[-1] != -999)

    # The batch dimension is just the first piece of the non-zero indexes:
    batch_size  = input_array.shape[0]
    batch_index = non_zero_inds[0]

    # Getting the voxel values (features) is also straightforward:
    features = np.expand_dims(split_tensors[-1][non_zero_inds],axis=-1)

    # Lastly, we need to stack up the coordinates, which we do here:
    dimension_list = [0]*(len(split_tensors)-1)
    for i in range(len(split_tensors) - 1):
        dimension_list[i] = split_tensors[i][non_zero_inds]

    # Tack on the batch index to this list for stacking:
    dimension_list.append(batch_index)

    # And stack this into one np array:
    dimension = np.stack(dimension_list, axis=-1)
    #coords = np.array([dimension_list[iwt] for iwt in range(len(dimension_list))])

    output_array = (torch.from_numpy(dimension).to(torch.device("cuda")),
                    torch.from_numpy(features).to(torch.device("cuda")),
                    batch_size,)
    return output_array

'''
Model below is an example, inspired by 
https://github.com/facebookresearch/SparseConvNet/blob/master/examples/3d_segmentation/fully_convolutional.py
Not yet even debugged!
EC, 24-March-2019
'''

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, (nvox,nvox,nvoxz), mode=3)).add(
                scn.SubmanifoldConvolution(dimension, nPlanes, 4, 3, False)).add(
                    scn.MaxPooling(dimension, 3, 3)).add(
                    scn.SparseResNet(dimension, 4, [
                        ['b', 8, 2, 1],
                        ['b', 16, 2, 1],
                        ['b', 24, 2, 1]])).add(
#                        ['b', 32, 2, 1]])).add(
                            scn.Convolution(dimension, 24, 32, 5, 1, False)).add(
                                scn.BatchNormReLU(32)).add(
                                    scn.SparseToDense(dimension, 32))
#        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
        self.linear = nn.Linear(int(32*46*46*96), 32)
        self.linear2 = nn.Linear(32,global_Nclass)
    def forward(self,x):
        x = self.sparseModel(x)
        x = x.view(-1, 32*46*46*96)
        x = nn.functional.elu(self.linear(x))
        x = self.linear2(x)
        x = nn.functional.softmax(x, dim=1)
        return x
 
net = Model().cuda()

modelfilepath = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/models/'
try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load(modelfilepath+'model-scn3dsigbkd-diffusion-larcv.pkl'))
    net.eval()
    print("Succeeded.")
except:
    print ("Failed to read pkl model. Proceeding from scratch.")

# Horovod: broadcast parameters.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)

learning_rate = 0.001 # 0.010
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
                      momentum=0.9)
# Horovod: wrap optimizer with DistributedOptimizer.
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=hvd.Compression.none)  # to start

# This moves the optimizer to the GPU:
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()

hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs

# config files
main_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_train.txt'
aux_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_test.txt'

print('initializing larcv io')
# initilize io
_larcv_interface = larcv_interface.larcv_interface()

# Prepare data managers:
io_config = {
    'filler_name' : 'TrainIO',
    'filler_cfg'  : main_fname,
    'verbosity'   : 1,
    'make_copy'   : True
}
aux_io_config = {
    'filler_name' : 'TestIO',
    'filler_cfg'  : aux_fname,
    'verbosity'   : 1,
    'make_copy'   : True
}
# Build up the data_keys:
data_keys = OrderedDict()
data_keys['image'] = 'data'
data_keys['label'] = 'label'
aux_data_keys = OrderedDict()
aux_data_keys['image'] = 'test_data'
aux_data_keys['label'] = 'test_label'

print('preparing larcv interface manager')
_larcv_interface.prepare_manager(mode, io_config, global_batch_size, data_keys)
_larcv_interface.prepare_manager(aux_mode, aux_io_config, global_batch_size, aux_data_keys)


if hvd.rank()==0:
    filename = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/'+'history-diffusion-larcv.csv'
    csvfile = open(filename,'w')


fieldnames = ['Training_Validation', 'Iteration', 'Epoch', 'Loss',
              'Accuracy', "Learning Rate"]

# only let one core write to this file.
if hvd.rank()==0:
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()

train_loss = hu.Metric('train_loss')
train_accuracy = hu.Metric('train_accuracy')
val_loss = hu.Metric('val_loss')
val_accuracy = hu.Metric('val_accuracy')

print('start training')
for epoch in range (global_n_epochs):

    tr_epoch_size = _larcv_interface.size('train')

    lr_step.step()
    for param_group in optimizer.param_groups:
        print('learning rate: %s'%param_group['lr'])

    for iteration in range(tr_epoch_size):
        net.train()
        optimizer.zero_grad()

        minibatch_data = _larcv_interface.fetch_minibatch_data(mode, fetch_meta_data=False)
        minibatch_dims = _larcv_interface.fetch_minibatch_dims(mode)

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = np.reshape(minibatch_data[key], minibatch_dims[key])

        # Strip off the primary/aux label in the keys:
        for key in minibatch_data:
            new_key = key.replace('aux_','')
            minibatch_data[new_key] = minibatch_data.pop(key)            
        
        minibatch_data['image'] = larcvsparse_to_scnsparse_3d(minibatch_data['image'])

        yhat = net(minibatch_data['image'])

        target = torch.from_numpy(minibatch_data['label']).max(1, keepdim=True)[1]
        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        loss = nn.functional.cross_entropy(yhat, target.reshape(-1).to(torch.device("cuda")))
        train_loss.update(loss)

        loss.backward()

        optimizer.step()

        net.eval()

        print("Train.Rank,Epoch: {},{}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g}]".format(hvd.rank(), 
                                                   epoch, iteration,float(train_loss.avg), train_accuracy.avg))

        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                  'Accuracy':train_accuracy.avg.data, "Learning Rate":learning_rate}
        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()

        # below is to keep this from exceeding 4 hrs
        if iteration > global_n_iterations_per_epoch:
            break



    # done with iterations within a training epoch
    te_epoch_size = _larcv_interface.size('test')
    for iteration in range(te_epoch_size):
        minibatch_data = _larcv_interface.fetch_minibatch_data(aux_mode, fetch_meta_data=False)
        minibatch_dims = _larcv_interface.fetch_minibatch_dims(aux_mode)

        for key in minibatch_data:
            if key == 'entries' or key == 'event_ids':
                continue
            minibatch_data[key] = np.reshape(minibatch_data[key], minibatch_dims[key])

        # Strip off the primary/aux label in the keys:
        for key in minibatch_data:
            new_key = key.replace('aux_','')
            minibatch_data[new_key] = minibatch_data.pop(key)            
        
        minibatch_data['image'] = larcvsparse_to_scnsparse_3d(minibatch_data['image'])

        yhat = net(minibatch_data['image'])
        
        target = torch.from_numpy(minibatch_data['label']).max(1, keepdim=True)[1]
        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        val_accuracy.update(acc)
        loss = nn.functional.cross_entropy(yhat, target.reshape(-1).to(torch.device("cuda")))
        val_loss.update(loss)

        print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

        output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                  'Loss':float(val_loss.avg), 'Accuracy':val_accuracy.avg, "Learning Rate":learning_rate}
        if hvd.rank()==0:
            history_writer.writerow(output)
        if iteration>=global_n_iterations_val:
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        csvfile.flush()

hostname = "hidden"
try:
    hostname = os.environ["HOSTNAME"]
except:
    pass
print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 


print("end of epoch")
torch.save(net.state_dict(), modelfilepath+'model-scn3dsigbkd-diffusion-larcv.pkl')

