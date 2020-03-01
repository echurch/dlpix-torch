from larcv import larcv

import glob
import os
import numpy as np
import pandas as pd
from random import shuffle

group_key = "/FANALIC/RECO_fwhm_07_voxel_3x3x3/voxels/"
event_key = "/FANALIC/RECO_fwhm_07_voxel_3x3x3/events/"

def main():
    # This code loops over training set files:
    top_input_path="/gpfs/alpine/proj-shared/nph133/next1t/FANAL_datafiles/"
    output_path="/gpfs/alpine/proj-shared/nph133/next1t/larcv_datafiles/"
    #glob_filter_train = ["FWHM_07/train/Tl208/*-00{00..14}-*.h5","FWHM_07/train/Bi214/*-00{00..14}-*.h5","FWHM_07/train/bb0nu/*-00{00..25}-*.h5"]
    #glob_filter_train = ["FWHM_07/train/Tl208/*-000[0-2]-*.h5","FWHM_07/train/Bi214/*-000[0-2]-*.h5","FWHM_07/train/bb0nu/*-000[0-4]-*.h5"]
    glob_filter_train = ["FWHM_05/train/Tl208/*.h5","FWHM_05/train/Bi214/*.h5","FWHM_05/train/bb0nu/*.h5"]
    #glob_filter_test = ["FWHM_07/validate/Tl208/*-008[0-2]-*.h5","FWHM_07/validate/Bi214/*-008[0-2]-*.h5","FWHM_07/validate/bb0nu/*-045[0-4]-*.h5"]
    #glob_filter_test = ["FWHM_05/validate/Tl208/*.h5","FWHM_05/validate/Bi214/*.h5","FWHM_05/validate/bb0nu/*.h5"]
    glob_filter_test=["FWHM_05/analyze/Bi214/FIELD_CAGE/*.h5"]

    #files = glob.glob(top_input_path + glob_filter)
    files_train = []
    '''
    for gf in glob_filter_train:
        files_train.extend(glob.glob(top_input_path + gf))
    '''
    files_test = []
    for gf in glob_filter_test:
        files_test.extend(glob.glob(top_input_path + gf))

    print('Found %s input training files'%len(files_train))
    print('Found %s input testing files'%len(files_test))

    next_new_meta = larcv.ImageMeta3D()
    next_new_meta.set_dimension(0, 2600, 867, -1300)
    next_new_meta.set_dimension(1, 2600, 867, -1300)
    next_new_meta.set_dimension(2, 2600, 867, -1300)
    #next_new_meta.set_dimension(0, 2600, 260, -1300)
    #next_new_meta.set_dimension(1, 2600, 260, -1300)
    #next_new_meta.set_dimension(2, 2600, 260, -1300)

    '''
    output_trn = os.path.basename('Next1Ton_3mm_fwhm07_larcv_balanced_0_noshf_train.h5')
    output_trn = output_path + "/" + output_trn
    io_manager_trn = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager_trn.set_out_file(output_trn)
    io_manager_trn.initialize()
    # convert train files
    print('Converting train files')
    convert_files(io_manager_trn, next_new_meta, files_train)
    io_manager_trn.finalize()
    '''

    output_tst = os.path.basename('Next1Ton_3mm_fwhm05_larcv_bi214_fieldcage.h5')
    output_tst = output_path + "/" + output_tst
    io_manager_tst = larcv.IOManager(larcv.IOManager.kWRITE)
    io_manager_tst.set_out_file(output_tst)
    io_manager_tst.initialize()
    # convert test files
    print('Converting test files')
    convert_files(io_manager_tst, next_new_meta, files_test)
    io_manager_tst.finalize()


#def convert_files(io_manager, next_new_meta, dframes, files, idcs, eidcs, fidcs):
def convert_files( io_manager, next_new_meta, files ):

    for fidc,f in enumerate(files):

        try:
            df = pd.read_hdf(f,group_key)
            dfe = pd.read_hdf(f,event_key)
        except Exception:
            continue

        evts = np.unique(np.array([idc[0] for idc in df.index.values]))
        for ievt,event in enumerate(evts):

            # check if passes fiducial cut:
            if ~(dfe[dfe.index == event]['fid_filter'].values[0]):
                continue

            run = 0
            label = 0
            if 'bb0nu' in f:
                run = 1
                label = 1
            if 'bb2nu' in f:
                run = 2
                label = 2
            if 'Bi214' in f:
                run = 3
            if 'Tl208' in f:
                run = 4
            
            io_manager.set_id(run, fidc, int(event))
 
            ################################################################################
            # Store the particle information:
            larcv_particle = larcv.EventParticle.to_particle(io_manager.get_data("particle", "label"))
            particle = larcv.Particle()
            particle.energy_init(0.)
            particle.pdg_code(label)
            larcv_particle.emplace_back(particle)
            ################################################################################
            # Store the voxel information:
            event_sparse3d = larcv.EventSparseTensor3D.to_sparse_tensor(
                io_manager.get_data("sparse3d", "voxels"))
 
            st = larcv.SparseTensor3D()
            st.meta(next_new_meta)
 
            position = larcv.VectorOfDouble()
            position.resize(3)
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
