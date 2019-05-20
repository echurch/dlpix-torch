import os, glob
import numpy as np
import h5py


cdef int vox = 10 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
cdef int nvox = int(1500/vox) # num bins in x,y dimension 
cdef int nvoxz = int(3000/vox) # num bins in z dimension 
cdef (int,int,int) voxels = (int(1500/vox),int(1500/vox),int(3000/vox) ) # These are 1x1x1cm^3 voxels

def diffusion(hitset):
    """Simulates diffusion in the MC hits"""
    cdef float Ionization = 22./1.e6 # number of ionization electrons
    cdef float Dif_tran_star  = 3500. # rootbar micron rootcm
    cdef float Dif_long_star  = 1000. # rootbar micron rootcm
    cdef float Drift_vel      = 1.    # units mm/mus
    cdef float Pressure       = 10.   # units of bar
    cdef float Lifetime       = 5e3  # units mus
    cdef float Space_sigma    = 0.1  # units of mm (100 microns)

    H = np.zeros((150,150,300))
    cdef int Nelectrons
    cdef (float, float, float) hit_3pos
    cdef float DT,DL,Drift_time
    cdef float esigma_xy, esigma_z

    cdef object hit
    for hit in hitset:
        Nelectrons = int(hit['hit_energy']/Ionization)
        hit_3pos = (hit['hit_position'][0] + 750., hit['hit_position'][1] + 750., hit['hit_position'][2] + 1500.)

        DT = Dif_tran_star*np.sqrt(hit_3pos[2]*1e-1/Pressure)*1e-3 
        DL = Dif_long_star*np.sqrt(hit_3pos[2]*1e-1/Pressure)*1e-3
        Drift_time = hit_3pos[2]/(Drift_vel)

        #  variance is the combination of spatial variance of electron production and variance from diffusion
        esigma_xy = np.sqrt(Space_sigma**2 + DT**2)
        esigma_z  = np.sqrt(Space_sigma**2 + DL**2)
        epos = (esigma_xy,esigma_xy,esigma_z)*np.random.randn(Nelectrons,3) + hit_3pos
            
        Htmp,_ = np.histogramdd(epos, bins=voxels, range=((0.,1500.),(0.,1500.),(0.,3000.)))
        H += Htmp

    return H


cdef str path=os.environ['HOME']+'/NEXT1Ton'
cdef str fsample = 'Tl208'
cdef str fgeom = 'OUTER_PLANES'
cdef list ftype = [ fsample+"/*"+fgeom+".h5" ]

def process_MC():
    cdef list files = []
    cdef str ft 
    for ft in ftype:
        files.extend( glob.glob(path+"/"+ft) )
    print('Found %s files.'%len(files))
    
    cdef str foutname = os.environ['HOME']+'/NEXT1Ton/preprocess/'+fsample+'-'+fgeom+'_diffusion.h5'
    cdef object h5file = h5py.File(foutname,'w')
    
    cdef object grpw=h5file.create_group('weights')
    cdef object grpc=h5file.create_group('coords')
    
    cdef int outidx = 0
    cdef object extentset, hitset
    cdef int ievt, current_index, current_starthit, current_endhit, ifile
    cdef float thresh = 0.003 # deposited energy

    for ifile in range(len(files)):
        print('Opening file %s'%files[ifile])
        with h5py.File(files[ifile],'r') as current_file:
            extentset = current_file['MC']['extents']
            for ievt in range(len(extentset)):
    
                current_index = ievt
             
                if current_index != 0:
                    current_starthit = int(extentset[current_index - 1]['last_hit'] + 1)
                else:
                    current_starthit = 0
             
                current_endhit = int(extentset[current_index]['last_hit'])
             
                hitset = current_file['MC']['hits'][current_starthit:current_endhit]
             
                feats = diffusion(hitset)
             
                tmp = np.nonzero(feats>thresh)
                # below 4-lines are torch-urous equivalent of numpy moveaxis to get batch indx on far right column.
                coords = tmp
                #bno = coords[:,0].clone() # wo clone this won't copy, it seems.
                #coords[:,0:3] = tmp[:,1:4]
                #coords[:,3] = bno
                indspgen = feats>thresh
    
                print('Saving %s voxels.'%len(feats[indspgen]))         
    
                grpw.create_dataset(str(outidx),data=feats[indspgen])
                grpc.create_dataset(str(outidx),data=coords)
                
                outidx += 1
    
    print('Done processing files.')
    
