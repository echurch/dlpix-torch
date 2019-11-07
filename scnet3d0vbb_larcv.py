import torch.nn as nn
import torch as torch
import horovod.torch as hvd
import os, glob
import csv
import numpy as np
import hvd_util as hu
from mpi4py import MPI
import pdb

##from larcv import larcv_interface
from larcv.distributed_queue_interface import queue_interface
from collections import OrderedDict

hvd.init()
seed = 314159
print("hvd.size() is: " + str(hvd.size()))
print("hvd.local_rank() is: " + str(hvd.local_rank()))
print("hvd.rank() is: " + str(hvd.rank()))

# Horovod: pin GPU to local rank.
#torch.cuda.set_device(hvd.local_rank())

os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
torch.cuda.manual_seed(seed)

global_Nclass = 2 # bkgd, 0vbb, 2vbb
global_n_iterations_per_epoch = 500
global_n_iterations_val = 52
global_n_epochs = 40
global_batch_size = hvd.size()*200  ## Can be at least 32, but need this many files to pick evts from in DataLoader
vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
nvox = int(2600/vox) # num bins in x,y dimension 
nvoxz = int(2600/vox) # num bins in z dimension 
voxels = (nvox,nvox,nvoxz ) # These are 1x1x1cm^3 voxels

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

    output_array = (dimension, features, batch_size,)
    return output_array

def to_torch(minibatch_data):
    for key in minibatch_data:
        if key == 'entries' or key =='event_ids':
            continue
        if key == 'image':
            minibatch_data['image'] = (
                    torch.tensor(minibatch_data['image'][0]).long(),
                    torch.tensor(minibatch_data['image'][1], device=torch.device('cuda')).float(),
                    minibatch_data['image'][2],
                )
        else:
            minibatch_data[key] = torch.tensor(minibatch_data[key],device=torch.device('cuda'))
    
    return minibatch_data

'''
Model below is an example, inspired by 
https://github.com/facebookresearch/SparseConvNet/blob/master/examples/3d_segmentation/fully_convolutional.py
Not yet even debugged!
EC, 24-March-2019
'''

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.initialconv = scn.SubmanifoldConvolution(dimension, nPlanes, 32, 5, False)
        self.residual = scn.Identity()
        self.add = scn.AddTable()
        self.sparsebl1 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 32, 32, 3, False)).add(
            scn.BatchNormLeakyReLU(32)).add(
            scn.SubmanifoldConvolution(dimension, 32, 32, 3, False)).add(
            scn.BatchNormalization(32))
        self.sparsebl2 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False)).add(
            scn.BatchNormLeakyReLU(64)).add(
            scn.SubmanifoldConvolution(dimension, 64, 64, 3, False)).add(
            scn.BatchNormalization(64))
        self.sparsebl3 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 96, 96, 3, False)).add(
            scn.BatchNormLeakyReLU(96)).add(
            scn.SubmanifoldConvolution(dimension, 96, 96, 3, False)).add(
            scn.BatchNormalization(96))
        self.sparsebl4 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False)).add(
            scn.BatchNormLeakyReLU(128)).add(
            scn.SubmanifoldConvolution(dimension, 128, 128, 3, False)).add(
            scn.BatchNormalization(128))
        self.sparsebl5 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 160, 160, 3, False)).add(
            scn.BatchNormLeakyReLU(160)).add(
            scn.SubmanifoldConvolution(dimension, 160, 160, 3, False)).add(
            scn.BatchNormalization(160))
        self.sparsebl6 = scn.Sequential().add(
            scn.SubmanifoldConvolution(dimension, 192, 192, 3, False)).add(
            scn.BatchNormLeakyReLU(192)).add(
            scn.SubmanifoldConvolution(dimension, 192, 192, 3, False)).add(
            scn.BatchNormalization(192))
        self.downsample1 = scn.Sequential().add(
            scn.Convolution(dimension, 32, 64, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormalization(64)).add(
            scn.LeakyReLU())
        self.downsample2 = scn.Sequential().add(
            scn.Convolution(dimension, 64, 96, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormalization(96)).add(
            scn.LeakyReLU())
        self.downsample3 = scn.Sequential().add(
            scn.Convolution(dimension, 96, 128, [3,3,3], [2,2,2], False)).add(
            scn.BatchNormalization(128)).add(
            scn.LeakyReLU())
        self.downsample4 = scn.Sequential().add(
            scn.Convolution(dimension, 128, 160, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormalization(160)).add(
            scn.LeakyReLU())
        self.downsample5 = scn.Sequential().add(
            scn.Convolution(dimension, 160, 192, [2,2,2], [2,2,2], False)).add(
            scn.BatchNormalization(192)).add(
            scn.LeakyReLU())
        self.finaldownsample = scn.SubmanifoldConvolution(dimension, 192, 2, 3, False)
        self.sparsetodense = scn.SparseToDense(dimension, 2)
        self.inputLayer = scn.InputLayer(dimension, voxels, mode=3)
        self.linear = nn.Linear(int(2*8*8*8), 2)
        self.linear3 = nn.Linear(2,1)
    def forward(self,x):
        x = self.inputLayer(x)
        x = self.initialconv(x)
        res = self.residual(x)
        x = self.sparsebl1(x)
        x = self.add([x,res])
        res = self.residual(x)
        x = self.sparsebl1(x)
        x = self.add([x,res])
        x = self.downsample1(x)
        res = self.residual(x)
        x = self.sparsebl2(x)
        x = self.add([x,res])
        res = self.residual(x)
        x = self.sparsebl2(x)
        x = self.add([x,res])
        x = self.downsample2(x)
        res = self.residual(x)
        x = self.sparsebl3(x)
        x = self.add([x,res])
        res = self.residual(x)
        x = self.sparsebl3(x)
        x = self.add([x,res])
        x = self.downsample3(x)
        res = self.residual(x)
        x = self.sparsebl4(x)
        x = self.add([x,res])
        res = self.residual(x)
        x = self.sparsebl4(x)
        x = self.add([x,res])
        x = self.downsample4(x)
        res = self.residual(x)
        x = self.sparsebl5(x)
        x = self.add([x,res])
        res = self.residual(x)
        x = self.sparsebl5(x)
        x = self.add([x,res])
        x = self.downsample5(x)
        x = self.finaldownsample(x)
        x = self.sparsetodense(x)
        x = x.view(-1, 2*8*8*8)
        x = nn.functional.elu(self.linear(x))
        x = torch.sigmoid(self.linear3(x))
        return x

net = Model().cuda()
criterion = torch.nn.BCELoss().cuda()

modelfilepath = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/models/'
try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load(modelfilepath+'model-scn3dsigbkd-fanal-10cm-larcv-48gpu-nextnew.pkl'))
    #net.eval()
    print("Succeeded.")
except:
    print ("Failed to read pkl model. Proceeding from scratch.")


learning_rate = 0.01 # 0.010
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
                      momentum=0.9)
# Horovod: wrap optimizer with DistributedOptimizer.
'''
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=hvd.Compression.none)  # to start
'''
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters())

# This moves the optimizer to the GPU:
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.cuda()

# Horovod: broadcast parameters.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs

# config files
main_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_train.txt'
aux_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_test.txt'

# initilize io
root_rank = hvd.size() - 1
# read_option=1 is ==> read_from_all_ranks
#_larcv_interface = larcv_interface(root=root_rank, read_option='read_from_all_ranks', 
#                                    local_rank=hvd.local_rank(), local_size=hvd.local_size())
'''
_larcv_interface = larcv_interface(root=root_rank, read_option='read_from_single_rank', 
                                    local_rank=hvd.local_rank(), local_size=hvd.local_size())
'''
##_larcv_interface = larcv_interface.larcv_interface()
_larcv_interface = queue_interface( random_access_mode="random_events" )

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

_larcv_interface.prepare_manager(mode, io_config, global_batch_size, data_keys, color = 0)
_larcv_interface.prepare_manager(aux_mode, aux_io_config, global_batch_size, aux_data_keys, color = 0)


historyfilepath = os.environ['PROJWORK']+'/nph133/next1t/csvout/'
if hvd.rank()==0:
    filename = historyfilepath+'history-fanal-10cm-larcv-48gpu-next1ton-noshf.csv'
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

for epoch in range (global_n_epochs):

    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    print('batches per epoch per rank: %s'%tr_epoch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    print('test batches per epoch per rank: %s'%te_epoch_size)

    lr_step.step()
    for param_group in optimizer.param_groups:
        lrnow = param_group['lr']
        print('learning rate: %s'%param_group['lr'])

    net.train()
    for iteration in range(tr_epoch_size):
        net.train()
        optimizer.zero_grad()

        _larcv_interface.prepare_next(mode)
        minibatch_data = _larcv_interface.fetch_minibatch_data(mode, pop=True, fetch_meta_data=False)
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
        minibatch_data = to_torch(minibatch_data)

        yhat = net(minibatch_data['image'])

        values, target = torch.max(minibatch_data['label'], dim=1)

        '''
        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        '''

        #loss = torch.nn.functional.cross_entropy(yhat, target)
        loss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1))
        '''
        train_loss.update(loss)
        '''

        loss.backward()

        optimizer.step()

        '''
        print("Train.Rank,Epoch: {},{}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g}]".format(hvd.rank(), 
                                                   epoch, iteration,float(train_loss.avg), float(train_accuracy.avg)))

        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                  'Accuracy':float(train_accuracy.avg.data), "Learning Rate":learning_rate}
        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()
        '''

        # below is to keep this from exceeding 4 hrs
        if iteration > global_n_iterations_per_epoch:
            break



    # done with iterations within a training epoch
    net.eval()
    for iteration in range(tr_epoch_size):
        net.eval()
        _larcv_interface.prepare_next(mode)
        minibatch_data = _larcv_interface.fetch_minibatch_data(mode, pop=True, fetch_meta_data=False)
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
        minibatch_data = to_torch(minibatch_data)

        yhat = net(minibatch_data['image'])

        values, target = torch.max(minibatch_data['label'], dim=1)
        
        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        tloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1))
        #tloss = torch.nn.functional.cross_entropy(yhat,target)
        train_loss.update(tloss)

        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                  'Accuracy':float(train_accuracy.avg.data), "Learning Rate":lrnow}

        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()

        # below is to keep this from exceeding 4 hrs
        if iteration > global_n_iterations_per_epoch:
            break

    for iteration in range(te_epoch_size):
        net.eval()

        _larcv_interface.prepare_next(aux_mode)
        minibatch_data = _larcv_interface.fetch_minibatch_data(aux_mode, pop=True, fetch_meta_data=False)
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
        minibatch_data = to_torch(minibatch_data)

        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)

        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        val_accuracy.update(acc)

        #vloss = torch.nn.functional.cross_entropy(yhat, target)
        vloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1))
        val_loss.update(vloss)

        print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

        output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                  'Loss':float(val_loss.avg), 'Accuracy':float(val_accuracy.avg), "Learning Rate":lrnow}

        if hvd.rank()==0:
            history_writer.writerow(output)
        if iteration>=global_n_iterations_val:
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        csvfile.flush()

#scfieldnames = ['Iteration', 'Class', 'Score0', 'Score1']
scfieldnames = ['Iteration', 'Class', 'Score0']

if hvd.rank()==0:
    scfilename = historyfilepath+'scoreeval-fanal-10cm-larcv-48gpu-next1ton-noshf.csv'
    sccsvfile = open(scfilename,'w')
    score_writer = csv.DictWriter(sccsvfile, fieldnames=scfieldnames)
    score_writer.writeheader()

te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
net.eval()
for iteration in range(te_epoch_size):
    net.eval()

    _larcv_interface.prepare_next(aux_mode)
    minibatch_data = _larcv_interface.fetch_minibatch_data(aux_mode, pop=True, fetch_meta_data=False)
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
    minibatch_data = to_torch(minibatch_data)

    yhat = net(minibatch_data['image'])
    
    values, target = torch.max(minibatch_data['label'], dim=1)

    for ievt in range(len(target)):
        targ = int(target[ievt])
        scr0 = float(yhat[ievt])
        #scr1 = float(yhat[ievt][1])

        #output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'Score1':scr1}
        output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0}

        if hvd.rank()==0:        
            score_writer.writerow(output)
        if iteration>=global_n_iterations_val:
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        sccsvfile.flush()



hostname = "hidden"
try:
    hostname = os.environ["HOSTNAME"]
except:
    pass
print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 

if hvd.rank() == 0:
    torch.save(net.state_dict(), modelfilepath+'model-scn3dsigbkd-fanal-10cm-larcv-48gpu-nextnew.pkl')
    #torch.save(net, modelfilepath+'model-scn3dsigbkd-fanal-10cm-larcv_test.pkl')

