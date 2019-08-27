from larcv import larcv

import glob
import os
import numpy as np
import pandas as pd
from random import shuffle

nevtsperfile = 5000000
train_frac = 0.8

def main():
    # This code loops over training set files:
    top_input_path="/gpfs/alpine/proj-shared/nph133/next1t/FANAL_datafiles/"
    output_path="/gpfs/alpine/proj-shared/nph133/next1t/larcv_datafiles/"
    glob_filter=["*/*.h5"]
    #glob_filter=["*/*.h5"]
    group_key = "/FANALIC/RECO_fwhm_07_voxel_10x10x10/voxels/"

    #files = glob.glob(top_input_path + glob_filter)
    files = []
    for gf in glob_filter:
        files.extend(glob.glob(top_input_path + gf))

    print('Found %s input files'%len(files))
    #print(files)

    # Each data file is processed independently
    dframes = []
    for f in files:
        try:
            dframes.append( pd.read_hdf(f,group_key) )
        except Exception:
            pass

    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 3000, 300, -1500)
    next_new_meta.set_dimension(1, 3000, 300, -1500)
    next_new_meta.set_dimension(2, 3000, 300, -1500)

    # shuffle events
    flengths = [ len(df.groupby('event_id')) for df in dframes ]
    fidcs = []
    eidcs = []
    for ifl,fl in enumerate(flengths):
        fidcs.extend([ifl for i in range(fl)])
        eidcs.extend([i for i in range(fl)])
    idcs = [ i for i in range(sum(flengths)) ]
    shuffle(idcs)

    # separate in test/train
    idcs_train = idcs[:int(len(idcs)*train_frac)]
    idcs_test = idcs[int(len(idcs)*train_frac):]

    print('Found %s events. Splitting into %s training and %s testing events.'%(len(idcs),len(idcs_train),len(idcs_test)))

    noutf_train = int(np.ceil(len(idcs_train)/nevtsperfile))
    print('Creating %s training output files'%noutf_train)
    for ifout in range(noutf_train):
        output_trn = os.path.basename('Next1Ton_10cm_fwhm07_larcv_'+str(ifout)+'_train.h5')
        output_trn = output_path + "/" + output_trn
        io_manager_trn = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager_trn.set_out_file(output_trn)
        io_manager_trn.initialize()

        # convert train files
        print('Converting train files')
        tmpid0 = ifout*nevtsperfile
        tmpid1 = (ifout+1)*nevtsperfile
        convert_files(io_manager_trn, next_new_meta, dframes, files, idcs_train[tmpid0:tmpid1], eidcs, fidcs)
        io_manager_trn.finalize()

    noutf_test = int(np.ceil(len(idcs_test)/nevtsperfile))
    print('Creating %s testing output files'%noutf_test)
    for ifout in range(noutf_test):
        output_tst = os.path.basename('Next1Ton_10cm_fwhm07_larcv_'+str(ifout)+'_test.h5')
        output_tst = output_path + "/" + output_tst
        io_manager_tst = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager_tst.set_out_file(output_tst)
        io_manager_tst.initialize()

        # convert test files
        print('Converting test files')
        tmpid0 = ifout*nevtsperfile
        tmpid1 = (ifout+1)*nevtsperfile
        convert_files(io_manager_tst, next_new_meta, dframes, files, idcs_test[tmpid0:tmpid1], eidcs, fidcs)
        io_manager_tst.finalize()


def convert_files(io_manager, next_new_meta, dframes, files, idcs, eidcs, fidcs):
    
    # loop over shuffled indices
    for ishf in idcs:

        df = dframes[fidcs[ishf]]
        ievt = eidcs[ishf]
        event = df.index[ievt][0]

        file_name = files[fidcs[ishf]]
        run = 0
        label = 0
        if 'bb0nu' in file_name:
            run = 1
            label = 1
        if 'bb2nu' in file_name:
            run = 2
            label = 2
        if 'Bi214' in file_name:
            run = 3
        if 'Tl208' in file_name:
            run = 4

        io_manager.set_id(run, fidcs[ishf], int(event))

        ################################################################################
        # Store the particle information:
        larcv_particle = larcv.EventParticle.to_particle(io_manager.get_data("particle", "label"))
        particle = larcv.Particle()
        particle.energy_init(0.)
        particle.pdg_code(label)
        larcv_particle.emplace_back(particle)
        ################################################################################

        ################################################################################
        # Store the voxel information:
        event_sparse3d = larcv.EventSparseTensor3D.to_sparse_tensor(
            io_manager.get_data("sparse3d", "voxels"))

        st = larcv.SparseTensor3D()
        st.meta(next_new_meta)

        position = larcv.VectorOfDouble()
        position.resize(3)
        '''
        xpos = df.loc[[df.index.get_level_values(0)[event]]].X.values
        ypos = df.loc[[df.index.get_level_values(0)[event]]].Y.values
        zpos = df.loc[[df.index.get_level_values(0)[event]]].Z.values
        '''
        xpos = df.loc[[event]].X.values
        ypos = df.loc[[event]].Y.values
        zpos = df.loc[[event]].Z.values

        weights = df.loc[[event]].E.values
        for ipos in range(len(xpos)):
            position[0] = xpos[ipos]
            position[1] = ypos[ipos]
            position[2] = zpos[ipos]
            index = next_new_meta.position_to_index(position)

            if index >= next_new_meta.total_voxels():
                print("Skipping voxel at original coordinates ({}, {}, {}) as it is out of bounds".format(
                    row.X, row.Y, row.Z))
                continue

            st.emplace(larcv.Voxel(index, weights[ipos]))

        event_sparse3d.emplace(st)
        ################################################################################

        io_manager.save_entry()

    return


if __name__ == '__main__':
    main()
