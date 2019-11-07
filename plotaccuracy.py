import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
traindata1 = pd.read_csv('../history-nodiffusion.csv')
traindata2 = pd.read_csv('../history-nodiffusion-2.csv')

traindata2['Epoch'] = traindata2['Epoch']+10
traindata = pd.concat([traindata1,traindata2], ignore_index=True)
'''
traindata = pd.read_csv('../history-fanal-10cm-larcv_treval.csv')
nepoch = len(traindata.groupby('Epoch'))

plt.figure(figsize=(8,5))

plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').mean()['Accuracy'], yerr=traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').std()['Accuracy'],fmt='-',color='navy',label='Training')
plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').mean()['Accuracy'], yerr=traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').std()['Accuracy'],fmt='-',color='tomato',label='Validation')

plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()

plt.savefig('../plots/trainingaccuracy_fanal_10cm_larcv.pdf',bbox_inches="tight")


plt.figure(figsize=(8,5))

plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').mean()['Loss'], yerr=traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').std()['Loss'],fmt='-',color='navy',label='Training')
plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').mean()['Loss'], yerr=traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').std()['Loss'],fmt='-',color='tomato',label='Validation')

plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()

plt.savefig('../plots/trainingloss_fanal_10cm_larcv.pdf',bbox_inches="tight")
