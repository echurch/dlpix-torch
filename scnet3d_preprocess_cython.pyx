import os, glob
import numpy as np
import h5py

cimport numpy as np
from multiprocessing import Pool
from functools import partial

cdef int vox = 5 # int divisor of 1500 and 1500 and 3000. Cubic voxel edge size in mm.
cdef int nvox = int(1500/vox) # num bins in x,y dimension 
cdef int nvoxz = int(3000/vox) # num bins in z dimension 
cdef (int,int,int) voxels = (int(1500/vox),int(1500/vox),int(3000/vox) ) # These are 1x1x1cm^3 voxels

cdef int nworkers = 80

cpdef np.ndarray diffusion(np.ndarray hitset):
    """Simulates diffusion in the MC hits"""
    cdef float Ionization = 22./1.e6 # number of ionization electrons
    cdef float Dif_tran_star  = 3500. # rootbar micron rootcm
    cdef float Dif_long_star  = 1000. # rootbar micron rootcm
    cdef float Drift_vel      = 1.    # units mm/mus
    cdef float Pressure       = 10.   # units of bar
    cdef float Lifetime       = 5e3  # units mus
    cdef float Space_sigma    = 0.1  # units of mm (100 microns)

    cdef float escale = 100. # actually generate 1/escale electrons and weight hists

    cdef np.ndarray H = np.zeros(voxels)
    cdef np.ndarray Htmp

    cdef int Nelectrons
    cdef (float, float, float) hit_3pos
    cdef float DT,DL,Drift_time
    cdef float esigma_xy, esigma_z
    cdef np.ndarray epos

    cdef np.ndarray hit
    cdef int ihit
    for ihit in range(len(hitset)):
        hit = np.asarray(hitset[ihit])
        Nelectrons = int(hit['hit_energy']/Ionization)
        hit_3pos = (hit['hit_position'][0] + 750., hit['hit_position'][1] + 750., hit['hit_position'][2] + 1500.)

        DT = Dif_tran_star*np.sqrt(hit_3pos[2]*1e-1/Pressure)*1e-3 
        DL = Dif_long_star*np.sqrt(hit_3pos[2]*1e-1/Pressure)*1e-3
        #Drift_time = hit_3pos[2]/(Drift_vel)

        #  variance is the combination of spatial variance of electron production and variance from diffusion
        esigma_xy = np.sqrt(Space_sigma**2 + DT**2)
        esigma_z  = np.sqrt(Space_sigma**2 + DL**2)
        epos = (esigma_xy,esigma_xy,esigma_z)*np.random.randn(int(Nelectrons/escale),3) + hit_3pos
            
        Htmp,_ = np.histogramdd(epos, bins=voxels, weights=np.ones(len(epos))*escale, range=((0.,1500.),(0.,1500.),(0.,3000.)))
        H += Htmp

    return H




cdef str path=os.environ['HOME']+'/NEXT1Ton'
cdef str fsample = 'bb0nu'
cdef str fgeom = 'ACTIVE'
cdef list ftype = [ fsample+"/*"+'-0000-'+fgeom+".h5" ]
#cdef list ftype = [ fsample+"/*"+fgeom+".h5" ]

def process_MC():
    cdef list files = []
    cdef str ft 
    for ft in ftype:
        files.extend( glob.glob(path+"/"+ft) )
    print('Found %s files.'%len(files))
   
    cdef str foutname = os.environ['PROJWORK']+'/nph133/next1t/batch_datafiles/'+fsample+'-00-'+fgeom+'_diffusion.h5'
    cdef object h5file = h5py.File(foutname,'w')
    cdef np.ndarray fname

    cdef str fl
    cdef list flengths = [len(h5py.File(fl)['MC']['extents']) for fl in files ]
    
    cdef np.ndarray feats
    cdef float thresh = 45 # number of electrons
    
    cdef object grpw = h5file.create_group('weights')
    cdef object grpc = h5file.create_group('coords')
    cdef object grpf = h5file.create_group('filenames')
    cdef object grpe = h5file.create_group('events')
    
    
    cdef list hitpool

    cdef np.ndarray extentset, hitset, events
    cdef int ievt, outid, current_index, current_starthit, current_endhit, ifile, ifl

    for ifile in range(len(files)):
        print('Opening file %s'%files[ifile])
        with h5py.File(files[ifile],'r') as current_file:
            fname = np.array([np.string_(fsample + '/' + files[ifile].split('/')[-1])])
            
            extentset = current_file['MC']['extents'][:]

            hitpool = []

            for ievt in range(len(extentset)):
                outid = sum([flengths[ifl] for ifl in range(ifile)]) + ievt

                current_index = ievt
    
                if current_index != 0:
                    current_starthit = int(extentset[current_index - 1]['last_hit'] + 1)
                else:
                    current_starthit = 0
    
                current_endhit = int(extentset[current_index]['last_hit'])
    
                hitpool.append(current_file['MC']['hits'][current_starthit:current_endhit])
    
            print('Processing %s events.'%len(extentset))
            with Pool(processes=nworkers) as pool:
                featset = pool.map(diffusion, hitpool)

            print('Saving %s events.'%len(extentset))
            for ievt in range(len(extentset)):
                outid = sum([flengths[ifl] for ifl in range(ifile)]) + ievt
                feats = featset[ievt]
                print('Saving %s voxels.'%len(feats[feats > thresh]))
                events = np.array([extentset['evt_number'][ievt]])
            
                grpw.create_dataset( str(outid), data=feats[feats > thresh] )
                grpc.create_dataset( str(outid), data=np.nonzero(feats > thresh) )
                grpf.create_dataset( str(outid), data=fname )
                grpe.create_dataset( str(outid), data=events )

    h5file.close()
                
    print('Done processing files.')
    
