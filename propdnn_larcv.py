import torch.nn as nn
import torch as torch
import horovod.torch as hvd
import os, glob
import csv
import numpy as np
import hvd_util as hu

##from larcv import larcv_interface
from larcv.distributed_queue_interface import queue_interface
from collections import OrderedDict

import torch.optim as optim
import sparseconvnet as scn

hvd.init()
#seed = 314159
seed = 12188
print("hvd.size() is: " + str(hvd.size()))
print("hvd.local_rank() is: " + str(hvd.local_rank()))
print("hvd.rank() is: " + str(hvd.rank()))

# Horovod: pin GPU to local rank.
#torch.cuda.set_device(hvd.local_rank())

os.environ["CUDA_VISIBLE_DEVICES"] = str(hvd.local_rank())
torch.cuda.manual_seed(seed)

global_Nclass = 2 # bkgd, 0vbb, 2vbb
global_n_iterations_per_epoch = 800
global_n_iterations_val = 80
global_n_epochs = 50
global_batch_size = hvd.size()*4049  ## Can be at least 32, but need this many files to pick evts from in DataLoader
vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
nvox = int(2600/vox) # num bins in x,y dimension 
nvoxz = int(2600/vox) # num bins in z dimension 
voxels = (nvox,nvox,nvoxz ) # These are 1x1x1cm^3 voxels

#learning_rate = 0.001 # 0.011
learning_rate = 0.025509168221734477/hvd.size()
#learning_rate = 0.0002094711377124768/hvd.size()

#mode = 'train'
#aux_mode ='test'
max_voxels= '1000'
producer='sparse3d_voxels_group'

dimension = 3
nPlanes = 1

modelfilepath = os.environ['MEMBERWORK']+'/nph133/'+os.environ['USER']+'/next1t/models/'
#modelname = 'model-fulldnn-next1ton-test1.pkl'
modelname = 'model-fulldnn-next1ton-lr001-gpu48-filter4-prod.pkl'
historyname = 'history-fulldnn-next1ton-filter4-test2.csv'
scorename = 'scoreeval-fulldnn-next1ton-lr001-gpu48-f4.csv'

'''
Loss function
'''
criterion = torch.nn.BCELoss().cuda()
    
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
        self.initialconv = scn.SubmanifoldConvolution(dimension, nPlanes, 32, 4, False)
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

def load_model():
    net = Model().cuda()
    try:
        print ("Reading weights from file")
        net.load_state_dict(torch.load(modelfilepath+modelname))
        #net.eval()
        print("Succeeded.")
    except:
        print ("Failed to read pkl model. Proceeding from scratch.")

    return net

def init_optimizer(net):
    # Horovod: scale learning rate by the number of GPUs.
    #optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
    #                      momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate * hvd.size())
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

    return lr_step, optimizer

def load_data():
    ''' initialize data loading '''
    # config files
    main_fname = os.environ['HOME']+'/dlpix-torch/larcvconfig_train.txt'
    aux_fname = os.environ['HOME']+'/dlpix-torch/larcvconfig_test.txt'
    # initilize io
    root_rank = hvd.size() - 1
    ##_larcv_interface = larcv_interface.larcv_interface()
    #_larcv_interface = queue_interface( random_access_mode="serial_access" )
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
    
    _larcv_interface.prepare_manager('train', io_config, global_batch_size, data_keys, color = 0)
    _larcv_interface.prepare_manager('test', aux_io_config, global_batch_size, aux_data_keys, color = 0)

    return _larcv_interface

def init_logger():
    historyfilepath = os.environ['PROJWORK']+'/nph133/next1t/csvout/kwoodruff/'
    fieldnames = ['Training_Validation', 'Iteration', 'Epoch', 'Loss',
                  'Accuracy', "Learning Rate"]
    if hvd.rank()==0:
        filename = historyfilepath+historyname
        csvfile = open(filename,'w')
        history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        history_writer.writeheader()

    return history_writer,csvfile
   
def prepare_batch(_larcv_interface,mode,iteration):
    if iteration != 0:
        _larcv_interface.prepare_next(mode)
        minibatch_data = _larcv_interface.fetch_minibatch_data(mode, pop=True, fetch_meta_data=False)
        minibatch_dims = _larcv_interface.fetch_minibatch_dims(mode)
    else:
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
    minibatch_data = to_torch(minibatch_data)

    return minibatch_data,minibatch_dims
 

def train_fullysupervised():
   
    net = load_model() 
    lr_step,optimizer = init_optimizer(net) 
    _larcv_interface = load_data()
    if hvd.rank()==0:
        history_writer,csvfile = init_logger()
    else:
        history_writer = None
        csvfile = None

    train_loss = hu.Metric('train_loss')
    train_accuracy = hu.Metric('train_accuracy')
    val_loss = hu.Metric('val_loss')
    val_accuracy = hu.Metric('val_accuracy')

    metrics = train_loss,train_accuracy,val_loss,val_accuracy
    
    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    Niterations = min(tr_epoch_size,global_n_iterations_per_epoch+1)

    for epoch in range(global_n_epochs):
    
        lr_step.step()
        net.train()
        for iteration in range(tr_epoch_size):
            net.train()
            optimizer.zero_grad()
   
            minibatch_data,minibatch_dims = prepare_batch(_larcv_interface,'train',iteration) 

            yhat = net(minibatch_data['image'])
            values, target = torch.max(minibatch_data['label'], dim=1)

            # backprop and optimizer step
            loss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1)) #\
            #     + torch.abs(nposy/bagsize - labelprop)
            loss.backward()
            optimizer.step()
    
            # below is to keep this from exceeding 4 hrs
            if iteration > global_n_iterations_per_epoch:
                break

        '''
        Run evaluation per epoch
        '''
        eval_epoch(net,optimizer,_larcv_interface,history_writer,csvfile,metrics,epoch)

        '''
        Save model
        '''
        if hvd.rank() == 0:
            torch.save(net.state_dict(), modelfilepath+modelname)
 

def train_semisupervised():
   
    net = load_model() 
    lr_step,optimizer = init_optimizer(net) 
    _larcv_interface = load_data()
    if hvd.rank()==0:
        history_writer,csvfile = init_logger()
    else:
        history_writer = None

    train_loss = hu.Metric('train_loss')
    train_accuracy = hu.Metric('train_accuracy')
    val_loss = hu.Metric('val_loss')
    val_accuracy = hu.Metric('val_accuracy')

    metrics = train_loss,train_accuracy,val_loss,val_accuracy
    
    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    Niterations = min(tr_epoch_size,global_n_iterations_per_epoch+1)

    C = 1.
    deltaC = 0.5
    bsize = global_batch_size//hvd.size()
    Ctmp = C*1e-5
    
    for epoch in range(global_n_epochs):
    
        Ctmp = min(Ctmp*(1+deltaC),C)
        lr_step.step()
        net.train()
        for iteration in range(tr_epoch_size):
            net.train()
            optimizer.zero_grad()
   
            minibatch_data,minibatch_dims = prepare_batch(_larcv_interface,'train',iteration) 
            '''
            Fix w and b and calculate y
            '''
            yhat = net(minibatch_data['image'])
            values, target = torch.max(minibatch_data['label'], dim=1)
    
            # fraction of labels that are positive:
            bagsize = minibatch_dims['label'][0]
            labelprop = target.sum().float()/bagsize
    
            yhat = net(minibatch_data['image'])
            ylatent = torch.zeros(bagsize)
            delta = torch.zeros(bagsize)
    
            # calculate reduction in loss first term
            # flip each ylatent from 0 to 1 and check loss change:
            loss1 = criterion(yhat, ylatent.type(torch.FloatTensor).cuda().view(-1,1))
            for j in range(bagsize):
                ylatent[j] = 1.
                tmploss1 = criterion(yhat, ylatent.type(torch.FloatTensor).cuda().view(-1,1))
                delta[j] = loss1 - tmploss1
                ylatent[j] = 0.
            # sort in descending order and 
            # find the number of pos. ys that min. loss
            deltasort,deltaorder = torch.sort(delta,0,descending=True)
            nposy = 0
            ylatent = torch.zeros(bagsize)
            yloss = Ctmp*criterion(yhat, ylatent.type(torch.FloatTensor).cuda().view(-1,1)) \
                  + torch.abs(0 - labelprop)
            for j in range(1,bagsize):
                ylatent[deltaorder[:j]] = 1.
                tmploss = Ctmp*criterion(yhat, ylatent.type(torch.FloatTensor).cuda().view(-1,1)) \
                        + torch.abs(j/bagsize - labelprop)
                if tmploss < yloss:
                    nposy = j
                    yloss = tmploss
                ylatent = torch.zeros(bagsize)
            
            # set ylatent to optimal set:
            ylatent = torch.zeros(bagsize)
            ylatent[deltaorder[:nposy]] = 1.
            #print('ylatent:')
            #print(ylatent)
            #print('target-ylatent:')
            #print(target.type(torch.FloatTensor) - ylatent)
            #print('yhat:')
            #print(yhat)
    
            '''
            Now fix y and calculate classic loss once for weights and biases:
            '''
            # backprop and optimizer step
            loss = criterion(yhat,ylatent.type(torch.FloatTensor).cuda().view(-1,1)) #\
            #     + torch.abs(nposy/bagsize - labelprop)
            loss.backward()
            optimizer.step()
    
            # below is to keep this from exceeding 4 hrs
            if iteration > global_n_iterations_per_epoch:
                break

        '''
        Run evaluation per epoch
        '''
        eval_epoch(net,optimizer,_larcv_interface,history_writer,csvfile,metrics,epoch)

        '''
        Save model
        '''
        if hvd.rank() == 0:
            torch.save(net.state_dict(), modelfilepath+modelname)


def eval_epoch(net,optimizer,_larcv_interface,history_writer,csvfile,metrics,epoch):
    tr_epoch_size = int(_larcv_interface.size('train')/global_batch_size)
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)

    train_loss,train_accuracy,val_loss,val_accuracy = metrics

    for param_group in optimizer.param_groups:
        lrnow = param_group['lr']
    
    net.eval()
    for iteration in range(tr_epoch_size):
        net.eval()
        minibatch_data,minibatch_dims = prepare_batch(_larcv_interface,'train',iteration) 

        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])

        values, target = torch.max(minibatch_data['label'], dim=1)
        
        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        train_accuracy.update(acc)
        tloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1))
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

        minibatch_data,minibatch_dims = prepare_batch(_larcv_interface,'test',iteration) 

        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)

        acc = hu.accuracy(yhat, target.type(torch.FloatTensor), weighted=True, nclass=global_Nclass)
        val_accuracy.update(acc)

        vloss = criterion(yhat,target.type(torch.FloatTensor).cuda().view(-1,1))
        val_loss.update(vloss)

        print("Val.Epoch: {}, Iteration: {}, Train,Val Loss: [{:.4g},{:.4g}], *** Train,Val Accuracy: [{:.4g},{:.4g}] ***".format(epoch, iteration,float(train_loss.avg), val_loss.avg, train_accuracy.avg, val_accuracy.avg ))

        output = {'Training_Validation':'Validation','Iteration':iteration, 'Epoch':epoch, 
                  'Loss':float(val_loss.avg), 'Accuracy':float(val_accuracy.avg), "Learning Rate":lrnow}

        if hvd.rank()==0:
            history_writer.writerow(output)
            csvfile.flush()
        if iteration>=global_n_iterations_val:
            break # Just check val for 4 iterations and pop out

    if hvd.rank()==0:        
        csvfile.flush()

def score_new_events():

    net = load_model()
    _larcv_interface = load_data()

    #scfieldnames = ['Iteration', 'Class', 'Score0', 'Score1']
    scfieldnames = ['Iteration', 'Class', 'Score0','X','Y','Z','E']
    historyfilepath = os.environ['PROJWORK']+'/nph133/next1t/csvout/kwoodruff/'
    
    if hvd.rank()==0:
        scfilename = historyfilepath+scorename
        sccsvfile = open(scfilename,'w')
        score_writer = csv.DictWriter(sccsvfile, fieldnames=scfieldnames)
        score_writer.writeheader()
    
    te_epoch_size = int(_larcv_interface.size('test')/global_batch_size)
    print('te_epoch_size: %s'%te_epoch_size)
    print('larcv size: %s'%_larcv_interface.size('test'))
    print('global_batch_size: %s'%global_batch_size)
    net.eval()
    for iteration in range(te_epoch_size):
        net.eval()
   
        minibatch_data,minibatch_dims = prepare_batch(_larcv_interface,'test',iteration) 
    
        '''
        Evaluate
        ''' 
        yhat = net(minibatch_data['image'])
        
        values, target = torch.max(minibatch_data['label'], dim=1)
   
        evtidc = torch.transpose(minibatch_data['image'][0],0,1)[-1] 
        for ievt in range(len(target)):
            targ = int(target[ievt])
            scr0 = float(yhat[ievt])
            #scr1 = float(yhat[ievt][1])
            img = torch.transpose(minibatch_data['image'][0][np.where(evtidc == ievt)],0,1).type(torch.float)
            val = minibatch_data['image'][1][np.where(evtidc == ievt)].type(torch.float)
            xmn = float(img[0].mean())
            ymn = float(img[1].mean())
            zmn = float(img[2].mean())
            emn = float(val.sum())
    
            #output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'Score1':scr1}
            output = {'Iteration':iteration, 'Class':targ, 'Score0':scr0, 'X':xmn, 'Y':ymn, 'Z':zmn, 'E':emn}
    
            if hvd.rank()==0:        
                score_writer.writerow(output)
            if iteration>=global_n_iterations_val:
                break # Just check val for 4 iterations and pop out
    
        if hvd.rank()==0:        
            sccsvfile.flush()


