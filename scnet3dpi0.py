import torch.nn as nn
import torch as torch
import math
import torch.utils.model_zoo as model_zoo
import horovod.torch as hvd
import os, glob
import numpy as np
import threading
import pdb


###########################################################
#
# Sparse Semantic segmentation network used by MicroBooNE
# to label (sub)dominant Energy gammas from pi0s.
#
# implementation from SCN example code (cite)
#
#
###########################################################


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
dtype = 'torch.cuda.FloatTensor' 
dtypei = 'torch.cuda.LongTensor' 

from matplotlib.patches import FancyArrowPatch
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)
                                                                    


global_Nclass = 3 
global_n_iterations_per_epoch = 20
global_batch_size = 4 # 32
vox = 2 # int divisor of 250 and 600 and 200. Cubic voxel edge size in cm.
nvox = int(200/vox) # num bins in each dimension 
voxels = (int(250/vox),int(600/vox),int(250/vox) ) # These are 2x2x2cm^3 voxels

plotted = np.zeros((8),dtype=bool) # 8 files 
plotted = np.ones((59999),dtype=bool) # 

#from mpi4py import MPI
#print("hvd.size()/MPI_COMM_RANK are: " + str(hvd.size()) + "/" + str(MPI.COMM_WORLD.Get_size())) 


# next 4 are globals
fit_pi0 = True
Epi = np.empty([1])
mpi_pca = np.empty([1])
mpi_constrained = np.empty([1])
Epi_cuts = np.empty([1])
mpi_pca_cuts = np.empty([1])
mpi_constrained_cuts = np.empty([1])


def eigenFunc( ix,iy,iz, H):
    indices = np.where(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    arr = np.array((np.arange(ix,ix+nvox)[indices[0]],np.arange(iy,iy+nvox)[indices[1]],np.arange(iz,iz+nvox)[indices[2]]))
    rep = np.array(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices],dtype='int64')
    xyz = np.repeat(arr,rep,axis=1)  # This is a 3xN matrix with each row of N one x,y,z coordinate wtd by energy deposition .
    xyz = arr # effectively don't wt by population of charge created in this voxel by this gamma, afterall
    pca = None
    try:
        pca = PCA(xyz.T)
    except:
        pass
    return pca

    '''
# The linalg calls give mkl runtime errors
    R = np.cov(xyz.T, rowvar=False)
    evals, evecs = linalg.eigh(R)
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    return evals,evecs
'''


def minConFit(paramsIn, data1, data2): # note the unpacked tuple as passed in ..
    " first set of points "
    x1 = data1[0]
    y1 = data1[1]
    z1 = data1[2]
    " second set of points "
    x2 = data2[0]
    y2 = data2[1]
    z2 = data2[2]
    Npts1 = x1.size
    Npts2 = x2.size

    
    x01 = paramsIn[0]
    y01 = paramsIn[1]
    z01 = paramsIn[2]
    x02 = paramsIn[3]
    y02 = paramsIn[4]
    z02 = paramsIn[5]
    m1x = paramsIn[6]
    m1y = paramsIn[7]
    m2x = paramsIn[8]
    m2y = paramsIn[9]


    m1z = np.sqrt(1.0 - m1x**2 - m1y**2)
    m2z = np.sqrt(1.0 - m2x**2 - m2y**2)

#    chi21 = (z1 - m1z/m1x*(x1-x01)  - z01)**2./2. + (z1 - m1z/m1y*(y1-y01)  - z01)**2./2.
#    chi22 = (z2 - m2z/m2x*(x2-x02)  - z02)**2./2. + (z2 - m2z/m2y*(y2-y02)  - z02)**2./2.
    chi21 = (z1 - (m1z/m1x)*(x1-x01) - z01)**2 + (z1 - (m1z/m1y)*(y1-y01)  - z01)**2. # EC, 29-Aug-2018
    chi22 = (z2 - (m2z/m2x)*(x2-x02) - z02)**2 + (z2 - (m2z/m2y)*(y2-y02)  - z02)**2.

    chi21x = (x1 - (m1x/m1y)*(y1-y01) - x01)**2  + (x1 - (m1x/m1z)*(z1-z01) - x01)**2
    chi22x = (x2 - (m2x/m2y)*(y2-y02) - x02)**2  + (x2 - (m2x/m2z)*(z2-z02) - x02)**2
    chi21y = (y1 - (m1y/m1x)*(x1-x01) - y01)**2  + (y1 - (m1y/m1z)*(z1-z01) - y01)**2
    chi22y = (y2 - (m2y/m2x)*(x2-x02) - y02)**2  + (y2 - (m2y/m2z)*(z2-z02) - y02)**2
    
    chi2Imp = (x01-x02)**2. + (y01-y02)**2. + (z01-z02)**2.

#    chi2 = chi21.sum() # + chi22.sum() + chi2Imp*(Npts1+Npts2)

#    chi2 = np.concatenate((chi21,chi22,chi21y,chi22y,chi21x,chi22x,np.array([chi2Imp]))) # for least-squares
    chi2 = np.concatenate((chi21,chi21y,chi21x,chi22,chi22y,chi22x,np.array([chi2Imp])))  # ,chi22,chi22y,chi22x)) # for least-squares
#    chi2 = np.concatenate((chi21,chi22))

    
#    print ("chi21,chi21y,chi21x, chi21: " + str(chi21.sum()/1.E6) + ", " + str(chi21y.sum()/1.E6) + ", " + str(chi21x.sum()/1.E6) + ", "  + str(chi2.sum()/1.E6))
#    print ("chi21,chi21y,chi21x, chi21: " + str(chi22.sum()/1.E6) + ", " + str(chi22y.sum()/1.E6) + ", " + str(chi22x.sum()/1.E6) + ", "  + str(chi2.sum()/1.E6))
    
    return chi2




def vtxConFit( ix,iy,iz, H, H1, H2, e1, e2):
    indices = np.where(H1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    # Next step by indexing with indices array collapses all activity to be contiguous, no more empty gaps, which compresses xyz1 in space, which .nonzero() does not do.
    arr = np.array((np.arange(ix,ix+nvox)[indices[0]],np.arange(iy,iy+nvox)[indices[1]],np.arange(iz,iz+nvox)[indices[2]]))
    rep = np.array(H1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices],dtype='int64')
    xyz1 = np.repeat(arr,rep,axis=1)  # This is a 3xN matrix with each row of N one x,y,z coordinate wtd by energy deposition by photon 1.
    xyz1 = arr # effectively don't wt by population of charge created in this voxel by this gamma, afterall
    xyz1 = H1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].nonzero()  # sanity!, 10-Sep-2018
    
    indices = np.where(H2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    arr = np.array((np.arange(ix,ix+nvox)[indices[0]],np.arange(iy,iy+nvox)[indices[1]],np.arange(iz,iz+nvox)[indices[2]]))
    rep = np.array(H2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices],dtype='int64')
    xyz2 = np.repeat(arr,rep,axis=1)  # This is a 3xN matrix with each row of N one x,y,z coordinate wtd by energy deposition by photon 2.
    xyz2 = arr # effectively don't wt by population of charge created in this voxel by this gamma, afterall
    xyz2 = H2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].nonzero()  # sanity!, 10-Sep-2018
    
#    p0 = np.array([xyz1[0].mean(),xyz1[1].mean(),xyz1[2].mean(), xyz2[0].mean(),xyz2[1].mean(),xyz2[2].mean(),  0.707,0.707,0.707,0.707]) # starting point for parameters
    if (not (e1 is None)) and (not (e2 is None)):
        p0 = np.array([xyz1[0].mean(),xyz1[1].mean(),xyz1[2].mean(), xyz2[0].mean(),xyz2[1].mean(),xyz2[2].mean(), e1.Wt[0,0],e1.Wt[0,1],e2.Wt[0,0],e2.Wt[0,1] ]) 
    else:
        p0 = np.array([xyz1[0].mean(),xyz1[1].mean(),xyz1[2].mean(), xyz2[0].mean(),xyz2[1].mean(),xyz2[2].mean(),  0.577,0.577,0.577,0.577]) # starting point for parameters
        
#    res = minimize(minConFit, p0, args=(xyz1,xyz2), jac=conFitDer, method='Newton-CG' ) # ,method='BFGS','SLSQP', options={'xtol':1E-6,'maxiter':6000,'disp':True})
#    res = minimize(minConFit, p0, args=(xyz1,xyz2), method='Nelder-Mead', options={'xtol':1E-6,'maxiter':60000,'disp':True})

    print ("p0: " + str(p0[0:10]))

    res = leastsq(minConFit, p0, args=(xyz1,xyz2), maxfev=20000)

    
#    print (res)

    return res,xyz1,xyz2


def energyGamma( ix,iy,iz, Hg, H):
    indices = np.where(Hg[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox])
    E = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox][indices].sum()
    return E


''' I'm  obsolescing this gen_waveform with DataLoader, and this will disappear soon ....'''
def gen_waveform(n_iterations_per_epoch=10, mini_batch_size=6):

    Nclass = global_Nclass

    x = np.ndarray(shape=(mini_batch_size, 1, nvox, nvox, nvox))
    y = np.ndarray(shape=(mini_batch_size, ))
    datapaths = "/ccs/home/echurch/data/*ana*"

    elec2MeV = 42700   # mip electrons/MeV
    # birks = 0.65 # rough-rough recombination
    birks = 1. # LArSoft has not imposed this, so don't correct for it. 19-July-2018
    
    files = [ i for i in glob.glob(datapaths)]
    truth = ["pi0"]
    labels = torch.FloatTensor([mini_batch_size, nvox, nvox, nvox])
    weight = torch.FloatTensor([mini_batch_size, nvox, nvox, nvox])
    global fit_pi0
    
    for iteration in range(n_iterations_per_epoch):
        for mini_batch in range(mini_batch_size):

            ind_file = np.random.randint(len(files)-1,size=1)[0] # save the last file for validation
            current_file = np.load(files[ind_file])
            current_index = np.random.randint(current_file.shape[0], size=1)[0]
            dim3 = np.array((mini_batch_size, nvox,nvox,nvox))
            np_labels  = np.zeros( dim3, dtype=np.int )
            np_weights = np.zeros( dim3, dtype=np.float32 )

            # Form labels ala H below.
            
            # sdTPC==3 is the DUNE TPC in 6x2x1 geometry in which most of these events are fully contained
            data = np.array((current_file[current_index,]['sdX'][current_file[current_index,]['sdTPC']==3],current_file[current_index,]['sdY'][current_file[current_index,]['sdTPC']==3], current_file[current_index,]['sdZ'][current_file[current_index,]['sdTPC']==3] ))
            dataT = data.T
            
            if dataT.sum() is 0:
                print("Problem! Image is empty for current_index " + str(current_index))
                raise
            

            xmin = np.argmin(dataT[:,0][dataT[:,0]>2])
            ymin = np.argmin(dataT[:,1][dataT[:,1]>2])
            zmin = np.argmin(dataT[:,2][dataT[:,2]>2])
            weights=current_file[current_index,]['sdElec'][current_file[current_index,]['sdTPC']==3]
            ##  view,chan,x
            
            H,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=weights)

            pixlabs1 = current_file[current_index,]['sdgamma1'][current_file[current_index,]['sdTPC']==3]
            pixlabs2 = current_file[current_index,]['sdgamma2'][current_file[current_index,]['sdTPC']==3]


            Hpl1,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs1) # original pixel value 1
            Hpl2,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs2/2.) # original pixel value 2

            (xmax,ymax,zmax) = np.unravel_index(np.argmax(H,axis=None),H.shape)
            # Crop this back to central 2mx2mx2m about max activity point
            ix = np.maximum(xmax-nvox/vox,0); ix = int(np.minimum(ix,voxels[0]-nvox))
            iy = np.maximum(ymax-nvox/vox,0); iy = int(np.minimum(iy,voxels[1]-nvox))
            iz = np.maximum(zmax-nvox/vox,0); iz = int(np.minimum(iz,voxels[2]-nvox))

# This fitting is slow, not so much the PCA, but the constrained 2-line fit, so only do it until we make plots
            if fit_pi0:
                
                eigs1 = eigenFunc(ix,iy,iz,Hpl1)
                eigs2 = eigenFunc(ix,iy,iz,Hpl2)
            
                fitParams,xyz1,xyz2 = vtxConFit(ix,iy,iz,H,Hpl1,Hpl2,eigs1,eigs2)
            
                en1 = energyGamma(ix,iy,iz,Hpl1,H)/birks
                en2 = energyGamma(ix,iy,iz,Hpl2,H)/birks
                costh = 1.0

                if (not (eigs1 is None)) and (not (eigs2 is None)):
                    costh = (eigs1.Wt[0,]*eigs2.Wt[0,]).sum() # 0th eigenvec is from largest eigenval
                if costh<(-1/np.sqrt(2)): # presume we got head/tail wrong on one of the PCA directions
                    costh = np.abs(costh)
                    
                mpi02a = 2*en1*en2*(1.0/elec2MeV)**2.*(1.0-costh)
                print ("'true' mpi0 from 2 separate PCAs: " + str(np.sqrt(mpi02a)))

                global mpi_pca
                global mpi_constrained
                global Epi
                global mpi_pca_cuts
                global mpi_constrained_cuts
                global Epi_cuts

                Epi = np.append(Epi,(en1+en2)/elec2MeV)
                    
                print ("Epi: " + str(Epi[-1]))
            
                mpi_pca = np.append(mpi_pca,np.sqrt(mpi02a))
                    
                if en1>30 and en2>30 and costh<0.93969:  # 20 deg
                    Epi_cuts = np.append(Epi_cuts,Epi[-1])
                    mpi_pca_cuts = np.append(mpi_pca_cuts,mpi_pca[-1])
                                
# [0] for leastsq, ["x"] for minimize
                
                m1z = np.sqrt(1.0 - fitParams[0][6]**2-fitParams[0][7]**2)
                m2z = np.sqrt(1.0 - fitParams[0][8]**2-fitParams[0][9]**2)
                vec1 = np.array([fitParams[0][6],fitParams[0][7],m1z])
                vec2 = np.array([fitParams[0][8],fitParams[0][9],m2z])
                costh = (vec1*vec2).sum()
                
                if costh>1: # Fitter may've returned default which can give 1.0000001
                    costh=1.0
                                
                mpi02b = 2*en1*en2*(1.0/elec2MeV)**2.*(1.0-costh)
                print ("'true' mpi0 from vtx-constrained 2-line fit: " + str(np.sqrt(mpi02b)))
                mpi_constrained = np.append(mpi_constrained,np.sqrt(mpi02b))

                if en1>30 and en2>30 and costh<0.93969:  # 20 deg
                    mpi_constrained_cuts = np.append(mpi_constrained_cuts,mpi_constrained[-1])
            
                
                if not plotted[ind_file] and  (not (eigs1 is None)) and (not (eigs2 is None)):
                    plotted[ind_file] = True
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D

                    fig = plt.figure(1)
                    fig.clf()
                    ax = Axes3D(fig)
                    xp1,yp1,zp1 = Hpl1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].nonzero()
                    xp2,yp2,zp2 = Hpl2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].nonzero()
                    
                    xp1 *= vox; xp2 *= vox;
                    yp1 *= vox; yp2 *= vox;
                    zp1 *= vox; zp2 *= vox; 

                    
                    ax.scatter(xp1,yp1,zp1, 'o', c='r', s=3) # s=np.log10(weights/np.mean(weights))+3 )
                    ax.scatter(xp2,yp2,zp2, 'o', c='b', s=3) # s=np.log10(weights/np.mean(weights))+3 )
                    cnt=0
                    vec1b = None
                    vec2b = None
                    
                    for v in  eigs1.Wt:
                        mute=10
                        cnt+=1
                        if cnt==1:
                            mute=20
                        a = Arrow3D([np.mean(xp1), 40*v[0]+np.mean(xp1)], [np.mean(yp1), 40*v[1]+np.mean(yp1)],
                                    [np.mean(zp1), 40*v[2]+np.mean(zp1)], mutation_scale=mute,
                                    lw=3, arrowstyle="-|>", color="black")
                        if cnt==1:   # Only add the 1st principle component
                            ax.add_artist(a)
                            vec1b=np.array([v[0],v[1],v[2]])
                    cnt=0
                    for v in eigs2.Wt:
                        mute=10
                        cnt+=1
                        if cnt==1:
                            mute=20
                        
                        a = Arrow3D([np.mean(xp2), 40*v[0]+np.mean(xp2)], [np.mean(yp2), 40*v[1]+np.mean(yp2)],
                                    [np.mean(zp2), 40*v[2]+np.mean(zp2)], mutation_scale=mute,
                                    lw=3, arrowstyle="-|>", color="black")
                        if cnt==1:   # Only add the 1st principle component
                            ax.add_artist(a)
                            vec2b=np.array([v[0],v[1],v[2]])
                            
                    ax.set_xlabel('X (drift)')
                    ax.set_ylabel('Y (up-down)')
                    ax.set_zlabel('Z (beam)')
                    ax.set_xlim(0.,nvox*vox)
                    ax.set_ylim(0.,nvox*vox)
                    ax.set_zlim(0.,nvox*vox)
#                    ax.axis('equal')
#                    ax.axis('tight')
                    plt.draw()
                    plt.savefig('scat_xyz_pi0_'+str(ind_file)+'_'+str(current_index)+'.png')
                    #                plt.close()
                    plt.clf()
                
                    fig = plt.figure(1)
                    fig.clf()

                    ax = Axes3D(fig)
                    t = np.linspace(-100,100,100)
                    # Note the b's need to be dropped eventually in next 2 lines!
                    ax.plot(fitParams[0][0]*vox+t*vec1[0],fitParams[0][1]*vox+t*vec1[1],fitParams[0][2]*vox+t*vec1[2],c='black')
                    ax.plot(fitParams[0][3]*vox+t*vec2[0],fitParams[0][4]*vox+t*vec2[1],fitParams[0][5]*vox+t*vec2[2],c='black')

                    
# Below looks reasonable, though it's merely stealing from the PCA firts.
#                    ax.plot(np.mean(xp1)+t*vec1[0],np.mean(yp1)+t*vec1[1],np.mean(zp1)+t*vec1[2],c='black')
#                    ax.plot(np.mean(xp2)+t*vec2[0],np.mean(yp2)+t*vec2[1],np.mean(zp2)+t*vec2[2],c='black')

                    ax.scatter(xp1,yp1,zp1, 'o', c='r', s=3) # s=np.log10(weights/np.mean(weights))+3 ) # r
                    ax.scatter(xp2,yp2,zp2, 'o', c='b', s=3) # s=np.log10(weights/np.mean(weights))+3 ) # b

                    
                    plt.xlabel('X (drift)')
                    plt.ylabel('Y (up-down)')
                    ax.set_zlabel('Z (beam)')
                    ax.set_xlim(0.,nvox*vox)
                    ax.set_ylim(0.,nvox*vox)
                    ax.set_zlim(0.,nvox*vox)
#                    ax.axis('equal')
#                    ax.axis('tight')

                    plt.draw()
                    plt.savefig('scat_xyz_pi0_fit_'+str(ind_file)+'_'+str(current_index)+'.png')
                    plt.close()
                    print('Saved file to disk:' + 'scat_xyz_pi0_fit_'+str(ind_file)+'_'+str(current_index)+'.png') 
                    
            # Back to the network building. Inside of mini_batch loop

            x[mini_batch][0] = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox] # The 0th element is for 1st (only) layer.
#            y[mini_batch] = labelvectmp
#            x[mini_batch][0] = H
#            y[mini_batch] = truth.index(ptype)
            np_labels[mini_batch] = np.zeros(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].shape)
            indx1 = np.where(Hpl1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            indx2 = np.where(Hpl2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            np_labels[mini_batch][indx1] = 1 # Eg1 pixels. Eg1>Eg2
            np_labels[mini_batch][indx2] = 2 # Eg1 pixels

            np_weights[mini_batch] = np_labels[mini_batch].astype('bool').astype('int')
            np_weights[mini_batch][np_weights[mini_batch]==0] = 1.0/np.prod(Hpl1.shape)/1000.


#            labels[mini_batch] = torch.from_numpy(np_labels[np_labels])
#            weight[mini_batch] = torch.from_numpy(np_weights[np_weights])

            # replace y's truth.index(ptype) with pixel labels array. Further, must add pixel weight array too.

        yield (Variable(torch.from_numpy(x).float().cuda()),
               Variable(torch.from_numpy(np_labels).long().cuda(),requires_grad=False),
               Variable(torch.from_numpy(np_weights).float().cuda(),requires_grad=False))



def accuracy(output, target, imgdata):
    """Computes the accuracy. we want the aggregate accuracy along with accuracies for the different labels. easiest to just use numpy..."""
    profile = False
    # needs to be as gpu as possible!
    maxk = 1


    batch_size = target.size(0)
    if profile:
        torch.cuda.synchronize()
        start = time.time()    
    #_, pred = output.topk(maxk, 1, True, False) # on gpu. slow AF
    _, pred = output.max( 1, keepdim=False) # on gpu
    if profile:
        torch.cuda.synchronize()
        print ("time for topk: "+str(time.time()-start)+" secs")

    if profile:
        start = time.time()
    #print "pred ",pred.size()," iscuda=",pred.is_cuda
    #print "target ",target.size(), "iscuda=",target.is_cuda
    targetex = target.resize_( pred.size() ) # expanded view, should not include copy


    correct = pred.eq( targetex.type(dtypei))  #.to(torch.device("cuda")) ) # on gpu
    #print "correct ",correct.size(), " iscuda=",correct.is_cuda    
    if profile:
        torch.cuda.synchronize()
        print ("time to calc correction matrix: "+str(time.time()-start)+" secs")

    # we want counts for elements wise

    num_per_class = {}
    corr_per_class = {}
    total_corr = 0
    total_pix  = 0

    if profile:
        torch.cuda.synchronize()            
        start = time.time()
    for c in range(output.size(1)):
        # loop over classes
        classmat = targetex.eq(int(c)).long() # elements where class is labeled
        #print "classmat: ",classmat.size()," iscuda=",classmat.is_cuda
        num_per_class[c] = classmat.long().sum()
        corr_per_class[c] = (correct.long()*classmat.type(dtypei)).long().sum() # mask by class matrix, then sum
        total_corr += corr_per_class[c].long()
        total_pix  += num_per_class[c].long()
    print ("total_pix: " + str(total_pix))
    print ("total_corr: " + str(total_corr))

    if profile:
        torch.cuda.synchronize()                
        print ("time to reduce: "+str(time.time()-start)+" secs")
        
    # make result vector
    res = []



    for c in range(output.size(1)):
        if num_per_class[c]>0:
            res.append( float(corr_per_class[c])/float(num_per_class[c])*100.0 )
        else:
            res.append( 0.0 )

    # totals
    if total_pix==0:
        res.append(0.0)
        print ("Mysteriously in here - total_pix: " +str(total_pix)  )
    else:
        res.append( 100.0*float(total_corr)/float(total_pix) )


    if num_per_class[1]==0 and num_per_class[2]==0:
        res.append(0.0)
        print ("Mysteriously in here: num-per-class" +str(num_per_class[1]) +", " +str(num_per_class[2]) )
    else:
        res.append( 100.0*float(corr_per_class[1]+corr_per_class[2])/float(num_per_class[1]+num_per_class[2]) ) # track/shower acc

    return res
                                                            

import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import sparseconvnet as scn
import time
import sys
import math


dimension = 3
reps = 1 #Conv block repetition factor
m = 32 #Unet number of features
nPlanes = [m, 2*m, 3*m, 4*m, 5*m] #UNet number of features per level
nPlanes = [m, 2*m, 3*m] # My 100x100x100 resolution doesn't allow to go further than this.

class Model(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(dimension, torch.LongTensor([nvox]*3), mode=3)).add(
           scn.SubmanifoldConvolution(dimension, 1, m, 3, False)).add(
           scn.UNet(dimension, reps, nPlanes, residual_blocks=False, downsample=[2,2])).add(
           scn.BatchNormReLU(m)).add(
           scn.OutputLayer(dimension))
        self.linear = nn.Linear(m, global_Nclass)
    def forward(self,x):
        x=self.sparseModel(x)
        x=self.linear(x)
        return x
 
net = Model()
# print(net) # this is lots of info
Net = net.cuda()

tensor_list = []
for dev_idx in range(torch.cuda.device_count()):
    tensor_list.append(torch.FloatTensor([1]).cuda(dev_idx))


# Horovod: broadcast parameters.
hvd.broadcast_parameters(net.state_dict(), root_rank=0)


try:
    print ("Reading weights from file")
    net.load_state_dict(torch.load('./model-scn3dpi0.pkl'))
    net.eval()
    print("Succeeded.")
except:
    print ("Failed to read pkl model. Proceeding from scratch.")
#    raise 

# Next two functions taken from Taritree's train_wlarcv1.py
# We define a pixel wise L2 loss

# taken from torch.nn.modules.loss
def _assert_no_grad(variable):
    assert not variable.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these variables as not requiring gradients"

import torch.nn.functional as F    
class PixelWiseNLLLoss(nn.modules.loss._WeightedLoss):
    def __init__(self,weight=None, size_average=True, ignore_index=-100 ):  ## size_average=True, 
        super(PixelWiseNLLLoss,self).__init__(weight,size_average)
        self.ignore_index = ignore_index
        self.weight = weight
        self.reduce = False # "mean" # False
        self.size_average = size_average
#        self.reduce = True

        self.mean = torch.mean # torch.mean.cuda() fails with 'has no attribute cuda" ....
        
    def forward(self,predict,target,pixelweights):
        """
        predict: (b,c,h,w) tensor with output from logsoftmax
        target:  (b,h,w) tensor with correct class
        pixelweights: (b,h,w) tensor with weights for each pixel
        """
        _assert_no_grad(target)
        _assert_no_grad(pixelweights)
        # reduce for below is false, so returns (b,h,w)

        pixelloss = F.nll_loss(predict,target, self.weight,self.size_average, self.ignore_index, self.reduce)

        return self.mean(pixelloss*pixelweights)

loss = PixelWiseNLLLoss().cuda()

learning_rate = 0.001 # 0.010
#optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# Horovod: scale learning rate by the number of GPUs.
optimizer = optim.SGD(net.parameters(), lr=learning_rate * hvd.size(),
                      momentum=0.9)
# Horovod: wrap optimizer with DistributedOptimizer.
compression = hvd.Compression.none  # .fp16 # don't use compression
optimizer = hvd.DistributedOptimizer(optimizer,
                                     named_parameters=net.named_parameters(),
                                     compression=hvd.Compression.none)  # to start
hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)

lr_step = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9) # lr drops to lr*0.9^N after 5N epochs
#val_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)


class BinnedDataset(Dataset):

    def __init__(self, path, frac_train, train=True, thresh=3, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        ftype = "*ana*"
        self.files = [ i for i in glob.glob(path+"/"+ftype)]
        dim3 = np.array(( nvox,nvox,nvox))
        self.np_labels  = np.zeros( dim3, dtype=np.int )
        self.np_weights = np.zeros( dim3, dtype=np.float32 )
        self.frac_train = frac_train
        self.valid_train = 1.0 - self.frac_train
        self.train = train
        self.path = path
        self.thresh = thresh
        self.lock = threading.Lock()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        with self.lock:
            x = np.ndarray(shape=( 1, nvox, nvox, nvox))

            ind_file = idx
            #        print ("ind_file: " +str(ind_file)+ " self.files is " + str(self.files))
            current_file = np.load(self.files[ind_file])
            if self.train:
                current_index = np.random.randint(int(current_file.shape[0]*self.frac_train), size=1)[0]
            else:
                current_index = np.random.randint(int(current_file.shape[0]*self.frac_train),int(current_file.shape[0]), size=1)[0]
                
            data = np.array((current_file[current_index,]['sdX'][current_file[current_index,]['sdTPC']==3],current_file[current_index,]['sdY'][current_file[current_index,]['sdTPC']==3], current_file[current_index,]['sdZ'][current_file[current_index,]['sdTPC']==3] ))
            dataT = data.T
            
            if dataT.sum() is 0:
                print("Problem! Image is empty for current_index " + str(current_index))
                raise
            
            xmin = np.argmin(dataT[:,0][dataT[:,0]>2])
            ymin = np.argmin(dataT[:,1][dataT[:,1]>2])
            zmin = np.argmin(dataT[:,2][dataT[:,2]>2])
            weights=current_file[current_index,]['sdElec'][current_file[current_index,]['sdTPC']==3]
            ##  view,chan,x

            voxels = (int(250/vox),int(600/vox),int(250/vox) ) # These are 2x2x2cm^3 voxels
            
            H,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=weights)
            
            pixlabs1 = current_file[current_index,]['sdgamma1'][current_file[current_index,]['sdTPC']==3]
            pixlabs2 = current_file[current_index,]['sdgamma2'][current_file[current_index,]['sdTPC']==3]

            Hpl1,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs1) # original pixel value 1
            Hpl2,edges = np.histogramdd(dataT,bins=voxels,range=((0.,250.),(0.,600.),(0.,250.)),weights=pixlabs2/2.) # original pixel value 2

            (xmax,ymax,zmax) = np.unravel_index(np.argmax(H,axis=None),H.shape)
            # Crop this back to central 2mx2mx2m about max activity point
            ix = np.maximum(xmax-nvox/vox,0); ix = int(np.minimum(ix,voxels[0]-nvox))
            iy = np.maximum(ymax-nvox/vox,0); iy = int(np.minimum(iy,voxels[1]-nvox))
            iz = np.maximum(zmax-nvox/vox,0); iz = int(np.minimum(iz,voxels[2]-nvox))

            x[0] = H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox] # The 0th element is for 1st (only) layer.
            self.np_labels = np.zeros(H[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox].shape)
            indx1 = np.where(Hpl1[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            indx2 = np.where(Hpl2[ix:ix+nvox,iy:iy+nvox,iz:iz+nvox]!=0)
            self.np_labels[indx2] = 2 # Eg1 pixels

            self.np_weights = self.np_labels.astype('bool').astype('int')
            self.np_weights[self.np_weights==0] = 1.0/np.prod(Hpl1.shape)/1000.


            return ( x[0], self.np_labels, self.np_weights )
            

        ''' Note that scn.InputLayer() expects 1 input layer not potentially N of them, so collapse x now.'''
'''
        return (torch.from_numpy(x.reshape((nvox,nvox,nvox))).float(),
                torch.from_numpy(self.np_labels).long(),
                torch.from_numpy(self.np_weights).float())
'''


binned_tdata = BinnedDataset(path='/ccs/home/echurch/pi0',frac_train=0.8,train=True)
binned_vdata = BinnedDataset(path='/ccs/home/echurch/pi0',frac_train=0.8,train=False)

import csv
with open('history.csv','w') as csvfile:
    fieldnames = ['Iteration', 'Epoch', 'Train Loss',
                  'Validation Loss', 'Train Accuracy', 'Validation Accuracy', "Learning Rate"]
    history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    history_writer.writeheader()
    thresh = 3

    for epoch in range (15):  # (400)
#        train_gen = gen_waveform(n_iterations_per_epoch=global_n_iterations_per_epoch,mini_batch_size=global_batch_size)

        train_gen = DataLoader(dataset=binned_tdata, batch_size=global_batch_size,
                               shuffle=False, num_workers=global_batch_size) #global_batch_size)
        lr_step.step()

        for iteration, minibatch in enumerate(train_gen):

            net.train()
#            torch.distributed.init_process_group(backend='nccl',
#                                                 init_method='env://',
#                                                 world_size=4,
#                                                 rank = hvd.rank())

            optimizer.zero_grad()

            feats, labels_var, weight_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])
            
            train_loss = loss(yhat, labels_var[indspgen].type(dtypei), weight_var[indspgen].type(dtype)) 
            train_loss.backward()
#            optimizer.synchronize()
            optimizer.step()



            ''' After some diagnostic period let's put this outside the Iteration loop'''
            train_accuracy = accuracy(yhat, labels_var[indspgen], feats[indspgen])    # None)
            net.eval()

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g}], Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data), train_accuracy[0], train_accuracy[1], train_accuracy[2]))


        val_gen = DataLoader(dataset=binned_vdata, batch_size=global_batch_size,
                                 shuffle=True, num_workers=global_batch_size)

        for iteration, minibatch in enumerate(val_gen):
            feats, labels_var, weight_var = minibatch            

            tmp = np.nonzero(feats>thresh)
            # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
            coords = tmp
            bno = coords[:,0].clone() # wo clone this won't copy, it seems.
            coords[:,0:3] = tmp[:,1:4]
            coords[:,3] = bno
            indspgen = feats>thresh

            yhat = net([coords,feats[indspgen].type(dtype).unsqueeze(1), global_batch_size])
            
            val_loss = loss(yhat,labels_var[indspgen].type(dtypei), weight_var[indspgen].type(dtype)) 
            #            val_accuracy = accuracy(y, yhat)
            val_accuracy = accuracy(yhat, labels_var[indspgen], feats[indspgen])   

            print("Epoch: {}, Iteration: {}, Loss: [{:.4g},{:.4g}], *** Train Accuracy: [{:.4g},{:.4g},{:.4g}, ***,  Val Accuracy: [{:.4g},{:.4g},{:.4g}]".format(epoch, iteration,float(train_loss.data), val_loss, train_accuracy[0], train_accuracy[1], train_accuracy[2], val_accuracy[0], val_accuracy[1], val_accuracy[2]))

            
            #                if (iteration%1 ==0) and (iteration>0):
                
            #            for g in optimizer.param_groups:
            #                learning_rate = g['lr']
            output = {'Iteration':iteration, 'Epoch':epoch, 'Train Loss': float(train_loss.data),
                      'Validation Loss':val_loss, 'Train Accuracy':train_accuracy, 'Validation Accuracy':val_accuracy, "Learning Rate":learning_rate}
            history_writer.writerow(output)
            break # Just do it once and pop out

        csvfile.flush()

        hostname = "hidden"
        try:
            hostname = os.environ["HOSTNAME"]
        except:
            pass
        print("host: hvd.rank()/hvd.local_rank() are: " + str(hostname) + ": " + str(hvd.rank())+"/"+str(hvd.local_rank()) ) 


    print("end of epoch")
    torch.save(net.state_dict(), 'model-scn3dpi0.pkl')

