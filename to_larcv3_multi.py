from larcv import larcv

import glob
import os
import numpy as np
import pandas as pd
from random import shuffle

nevtsperfile = 5000000
train_frac = 0.9

def main():
    # This code loops over training set files:
    top_input_path="/gpfs/alpine/proj-shared/nph133/next1t/FANAL_datafiles/"
    output_path="/gpfs/alpine/proj-shared/nph133/next1t/larcv_datafiles/"
    '''
    glob_filter_train=["Bi214/*-00[0-7]*-FIELD_CAGE.h5","Bi214/*-00[0-7]*-OUTER_PLANE.h5","Bi214/*-00[0-7]*-INNER_SHIELDING.h5","Bi214/*-00[0-7]*-VESSEL.h5","Bi214/*-00[0-7]*-READOUT_PLANE.h5",
                       "Tl208/*-00*-FIELD_CAGE.h5","Tl208/*-01[0-5]*-FIELD_CAGE.h5","Tl208/*-00*-READOUT_PLANE.h5","Tl208/*-01[0-5]*-READOUT_PLANE.h5","Tl208/*-000[0-7]-VESSEL.h5",
                       "Tl208/*-000[0-7]-OUTER_PLANE.h5","Tl208/*-00[0-3]*-INNER_SHIELDING.h5","bb0nu/*-0[0-2]*-ACTIVE.h5","bb0nu/*-03[0-1]*-ACTIVE.h5"]
    glob_filter_test=["Bi214/*-00[8-9]*-FIELD_CAGE.h5","Bi214/*-00[8-9]*-OUTER_PLANE.h5","Bi214/*-00[8-9]*-INNER_SHIELDING.h5","Bi214/*-00[8-9]*-VESSEL.h5","Bi214/*-00[8-9]*-READOUT_PLANE.h5",
                      "Tl208/*-01[6-9]*-FIELD_CAGE.h5","Tl208/*-01[6-9]*-READOUT_PLANE.h5","Tl208/*-000[8-9]-VESSEL.h5","Tl208/*-000[8-9]-OUTER_PLANE.h5","Tl208/*-004*-INNER_SHIELDING.h5","bb0nu/*-03[2-9]*-ACTIVE.h5"]
    '''
    glob_filter_train = ["FWHM_07/train/*/*.h5"]
    #glob_filter_test = ["FWHM_07/validate/*/*.h5"]
    group_key = "/FANALIC/RECO_fwhm_07_voxel_10x10x10/voxels/"
    event_key = "/FANALIC/RECO_fwhm_07_voxel_10x10x10/events/"

    #files = glob.glob(top_input_path + glob_filter)
    files_train = []
    for gf in glob_filter_train:
        files_train.extend(glob.glob(top_input_path + gf))
    #files_test = []
    #for gf in glob_filter_test:
    #    files_test.extend(glob.glob(top_input_path + gf))

    print('Found %s input training files'%len(files_train))
    #print('Found %s input testing files'%len(files_test))

    # Each data file is processed independently
    dframes_train = []
    devents_train = []
    for f in files_train:
        try:
            dframes_train.append( pd.read_hdf(f,group_key) )
            devents_train.append( pd.read_hdf(f,event_key) )
        except Exception:
            pass
    
    #dframes_test = []
    #devents_test = []
    #for f in files_test:
    #    try:
    #        dframes_test.append( pd.read_hdf(f,group_key) )
    #        devents_test.append( pd.read_hdf(f,event_key) )
    #    except Exception:
    #        pass

    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 2600, 260, -1300)
    next_new_meta.set_dimension(1, 2600, 260, -1300)
    next_new_meta.set_dimension(2, 2600, 260, -1300)

    # shuffle events
    flengths_train = [ len(df.groupby('event_id')) for df in dframes_train ]
    fidcs_train = []
    eidcs_train = []
    for ifl,fl in enumerate(flengths_train):
        fidcs_train.extend([ifl for i in range(fl)])
        eidcs_train.extend([i for i in range(fl)])
    #idcs_train = [ i for i in range(sum(flengths_train)) ]
    #shuffle(idcs_train)
    idcs = [ i for i in range(sum(flengths_train)) ]
    shuffle(idcs)

    #flengths_test = [ len(df.groupby('event_id')) for df in dframes_test ]
    #fidcs_test = []
    #eidcs_test = []
    #for ifl,fl in enumerate(flengths_test):
    #    fidcs_test.extend([ifl for i in range(fl)])
    #    eidcs_test.extend([i for i in range(fl)])
    #idcs_test = [ i for i in range(sum(flengths_test)) ]
    #shuffle(idcs_test)

    # separate in test/train
    idcs_train = idcs[:int(len(idcs)*train_frac)]
    idcs_test = idcs[int(len(idcs)*train_frac):]

    print('len of idcs_train: %s'%len(idcs_train))
    print('len of idcs_test: %s'%len(idcs_test))
    print('overlap of idcs_train/test:')
    print([value for value in idcs_train if value in idcs_test])

    print('Found %s training and %s testing events.'%(len(idcs_train),len(idcs_test)))

    noutf_train = int(np.ceil(len(idcs_train)/nevtsperfile))
    print('Creating %s training output files'%noutf_train)
    for ifout in range(noutf_train):
        output_trn = os.path.basename('Next1Ton_10cm_fwhm07_larcv_balanced_'+str(ifout)+'_9010_train.h5')
        output_trn = output_path + "/" + output_trn
        io_manager_trn = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager_trn.set_out_file(output_trn)
        io_manager_trn.initialize()

        # convert train files
        print('Converting train files')
        tmpid0 = ifout*nevtsperfile
        tmpid1 = (ifout+1)*nevtsperfile
        convert_files(io_manager_trn, next_new_meta, dframes_train, devents_train, files_train, idcs_train[tmpid0:tmpid1], eidcs_train, fidcs_train)
        #convert_files(io_manager_trn, next_new_meta, dframes_train, files_train, idcs_train[tmpid0:tmpid1], eidcs_train, fidcs_train)
        io_manager_trn.finalize()

    noutf_test = int(np.ceil(len(idcs_test)/nevtsperfile))
    print('Creating %s testing output files'%noutf_test)
    for ifout in range(noutf_test):
        output_tst = os.path.basename('Next1Ton_10cm_fwhm07_larcv_balanced_'+str(ifout)+'_9010_test.h5')
        output_tst = output_path + "/" + output_tst
        io_manager_tst = larcv.IOManager(larcv.IOManager.kWRITE)
        io_manager_tst.set_out_file(output_tst)
        io_manager_tst.initialize()

        # convert test files
        print('Converting test files')
        tmpid0 = ifout*nevtsperfile
        tmpid1 = (ifout+1)*nevtsperfile
        #convert_files(io_manager_tst, next_new_meta, dframes_test, devents_test, files_test, idcs_test[tmpid0:tmpid1], eidcs_test, fidcs_test)
        convert_files(io_manager_tst, next_new_meta, dframes_train, devents_train, files_train, idcs_test[tmpid0:tmpid1], eidcs_train, fidcs_train)
        io_manager_tst.finalize()


#def convert_files(io_manager, next_new_meta, dframes, files, idcs, eidcs, fidcs):
def convert_files(io_manager, next_new_meta, dframes, devents, files, idcs, eidcs, fidcs):
    
    # loop over shuffled indices
    for ishf in idcs:

        df = dframes[fidcs[ishf]]
        dfe = devents[fidcs[ishf]]
        ievt = eidcs[ishf]
        event = df.index[ievt][0]

        if ~(dfe[dfe.index == event]['fid_filter'].values[0]):
            continue

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
