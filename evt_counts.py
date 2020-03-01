import glob
import os
import numpy as np
import pandas as pd
import csv

event_key = "/FANALIC/RECO_fwhm_07_voxel_10x10x10/events/"

# This code loops over training set files:
top_input_path="/gpfs/alpine/proj-shared/nph133/next1t/FANAL_datafiles/"
output_path="/gpfs/alpine/proj-shared/nph133/next1t/larcv_datafiles/"
glob_filter=["FWHM_07/analyze/Bi214/FIELD_CAGE/*.h5"]

files = []
for gf in glob_filter:
    files.extend(glob.glob(top_input_path + gf))

print('Found %s input testing files'%len(files))

output_tst = os.path.basename('Next1Ton_10cm_fwhm07_bi214_fieldcage_counts.csv')
filename = output_path + "/" + output_tst

fieldnames = ['file','N_events','N_smE_filter','N_fid_filter']

csvfile = open(filename,'w')
history_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

history_writer.writeheader()



for fidc,f in enumerate(files):

    try:
        dfe = pd.read_hdf(f,event_key)
    except Exception:
        continue

    nevts = len(dfe.index.unique())
    nsme = len(dfe[dfe['smE_filter'] == True].index.unique())
    nfid = len(dfe[(dfe['smE_filter'] == True)&(dfe['fid_filter'] == True)].index.unique())

    output = {'file':fidc, 'N_events':nevts, 'N_smE_filter':nsme, 'N_fid_filter':nfid}
    history_writer.writerow(output)
    csvfile.flush()
