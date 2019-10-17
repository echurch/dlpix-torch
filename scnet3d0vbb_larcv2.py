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
global_n_iterations_per_epoch = 485
global_n_iterations_val = 52
global_n_epochs = 50 # 48 nodes
global_batch_size = hvd.size()*50  ## Can be at least 32, but need this many files to pick evts from in DataLoader
vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
# 2.4 MeV e's/gammas shouldn't go more than ~30cm.
nvox = int(2700/vox) # num bins in x,y dimension  # using KW's shrunk-volume now in larcvsparse_to_scnsparse_3d()
nvoxz = int(2700/vox) # num bins in z dimension 


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

    '''
    For this script variant don't shift and crop.

    # normalize the positions (have each event start at 0,0,0
    offset_x = [0]*len(batch_index)
    offset_y = [0]*len(batch_index)
    offset_z = [0]*len(batch_index)
    for i in np.unique(batch_index):
        idxs = np.where(batch_index == i)[0]
        for j in idxs:
            offset_x[j] = min(dimension_list[0][idxs.min():idxs.max()+1])
            offset_y[j] = min(dimension_list[1][idxs.min():idxs.max()+1])
            offset_z[j] = min(dimension_list[2][idxs.min():idxs.max()+1])

    dimension_list[0] = dimension_list[0] - offset_x
    dimension_list[1] = dimension_list[1] - offset_y
    dimension_list[2] = dimension_list[2] - offset_z
    '''

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
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, (nvox,nvox,nvoxz), mode=3)).add(
                scn.SubmanifoldConvolution(dimension, nPlanes, 16,  2, False)).add(  
                scn.SubmanifoldConvolution(dimension,      16,  16, 15, False)).add(  # Added this line. EC, 12-Oct-2019
                scn.SubmanifoldConvolution(dimension,      16,   4,  5, False)).add(  # Added this line. EC, 12-Oct-2019
                    scn.MaxPooling(dimension, 3, 3)).add(
                    scn.Dropout(0.4)).add(
                    scn.SparseResNet(dimension, 4, [
                        ['b', 8, 2, 1],
                        ['b', 12, 2, 1],
                        ['b', 16, 2, 1]])).add(
#                        ['b', 32, 2, 1]])).add(
                            scn.Convolution(dimension, 16, 16, 3, 1, False)).add(
                            scn.Convolution(dimension,  16, 16, 5, 1, False)).add(
                                scn.Dropout(0.4)).add(
                                scn.BatchNormLeakyReLU(16)).add(
#                                    scn.SubmanifoldConvolution(dimension, 16, 2, 3, False)).add(
                                        scn.SparseToDense(dimension, 16))
#        self.spatial_size = self.sparseModel.input_spatial_size(torch.LongTensor([1, 1]))
# The MaxPooling in above model striding leaves us with a dimension of nvox/3 x nvox/3 x nvoxz/3 ... and somehow -2 from padding

        self.linear = nn.Linear(16*84*84*84, 10)  ## int(16*(nvox/3-2) * (nvox/3-2) * (nvoxz/3-2)), 10)
        self.linear2 = nn.Linear(10, global_Nclass)

    def forward(self,x):
        x = self.sparseModel(x)

        x = x.view(-1, 16*84*84*84) ## int(16*(nvox/3-2) * (nvox/3-2) * (nvoxz/3-2)) )

#        x = nn.functional.relu(self.linear(x))
        x = self.linear(x)
        x = self.linear2(x)
#        x = torch.sigmoid(x)
#        x = nn.functional.softmax(x, dim=1)
        return x
 
net = Model().cuda()

criterion = torch.nn.CrossEntropyLoss()

modelfilepath = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/models/'
try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load(modelfilepath+'model-scn3dsigbkd-fanal-10mm-larcv2.pkl'))
    net.eval()
    print("Succeeded.")
except:
    print ("Failed to read pkl model. Proceeding from scratch.")


learning_rate = 0.050 # 0.010
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(net.parameters(), lr=learning_rate,  ##  * hvd.size(),
                      momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=learning_rate , betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)   #learning_rate*hvd.size()

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

lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8) # lr drops to lr*0.8^N after 2N epochs

## config files
main_fname = '/ccs/home/echurch/dlpix-torch/larcvconfig_train.txt'
aux_fname = '/ccs/home/echurch/dlpix-torch/larcvconfig_test.txt'

#main_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_train.txt'
#aux_fname = '/ccs/home/kwoodruff/dlpix-torch/larcvconfig_test.txt'

print('initializing larcv io')
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
_larcv_interface = queue_interface()

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
_larcv_interface.prepare_manager(mode, io_config, global_batch_size, data_keys, color = MPI.COMM_WORLD.rank)
_larcv_interface.prepare_manager(aux_mode, aux_io_config, global_batch_size, aux_data_keys, color = MPI.COMM_WORLD.rank)


if hvd.rank()==0:
    filename = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/'+'history-fanal-10mm-larcv2.csv'
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

    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    print('batches per epoch: %s'%tr_epoch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    print('test batches per epoch: %s'%te_epoch_size)

    lr_step.step()
    for param_group in optimizer.param_groups:
        print('learning rate: %s'%param_group['lr'])
        learning_rate = param_group['lr']

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

#        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
#        train_accuracy.update(acc)

        loss = criterion(yhat, target)
        train_loss.update(loss)

        loss.backward()

        optimizer.step() # hvd threads joined here (I think!), weights updated


        values, target = torch.max(minibatch_data['label'], dim=1)
        
        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        tloss = criterion(yhat,target)
        #tloss = torch.nn.functional.cross_entropy(yhat,target)
        train_loss.update(tloss)

        
        print("Train.Rank,Epoch: {},{}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g}]".format(hvd.rank(), 
                                                   epoch, iteration,float(train_loss.avg), train_accuracy.avg))
        
        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),                                                                                                                          'Accuracy':float(train_accuracy.avg.data), "Learning Rate":learning_rate}                                                                                                                                      
        if hvd.rank()==0:                                                                                                                                                                                                       
            history_writer.writerow(output)                                                                                                                                                                                     
            csvfile.flush()                                                                                                                                                                                                             
        ''' below is to keep this from exceeding 4 hrs '''
        if iteration > global_n_iterations_per_epoch:
            print ("Breaking out of training iterations loop after iteration: " + str(iteration))
            break



    ''' done with iterations within a training epoch. Now eval over training, using the full epoch's model from above.
    '''
    '''
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

#        print('target:')
#        print(target)
#        print('yhat:')
#        print(yhat)
        
        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        tloss = criterion(yhat,target)
        #tloss = torch.nn.functional.cross_entropy(yhat,target)
        train_loss.update(tloss)

        output = {'Training_Validation':'Training', 'Iteration':iteration, 'Epoch':epoch, 'Loss': float(train_loss.avg),
                  'Accuracy':float(train_accuracy.avg.data), "Learning Rate":learning_rate}

        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()


    # done with training iterations within a training epoch. Now look at some Val iterations.
    '''

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

        acc = hu.accuracy(yhat, target, weighted=True, nclass=global_Nclass)
        val_accuracy.update(acc)

        loss = criterion(yhat, target)
        val_loss.update(loss)

        print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

        output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                  'Loss':float(val_loss.avg), 'Accuracy':float(val_accuracy.avg), "Learning Rate":learning_rate}

        if hvd.rank()==0:
            history_writer.writerow(output)
        if iteration>=global_n_iterations_val:
            print ("Breaking out of validation iterations loop after iteration: " + str(iteration))
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        csvfile.flush()


# KW's loop, lastly, to store away the actual scores of each validation evt, using final model

scfieldnames = ['Iteration', 'Class', 'Score0', 'Score1']
if hvd.rank()==0:
    scfilename = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/'+'history-fanal-10mm-larcv_scores2.csv'
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
    print ("Made it here 0.")
    for ievt in range(len(target)):
        targ = int(target[ievt])
#        scr0 = float(yhat[ievt])
        scr0 = float(yhat[ievt][0])
        scr1 = float(yhat[ievt][1])
        print ("Made it here 1.")
        output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'Score1':scr1}
        #output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0}
        print ("Made it here 2.")
        if hvd.rank()==0:
            print ("Made it here 3.")
            score_writer.writerow(output)
        if iteration>=global_n_iterations_val:
            print ("Made it here 4.")
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        print ("Made it here 5.")
        sccsvfile.flush()



hostname = "hidden"
try:
    hostname = os.environ["HOSTNAME"]
except:
    pass
print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 


print("end of epoch")
torch.save(net.state_dict(), modelfilepath+'model-scn3dsigbkd-fanal-10cm-larcv2.pkl')

