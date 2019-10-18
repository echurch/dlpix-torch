import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pdb

'''
traindata1 = pd.read_csv('../history-nodiffusion.csv')
traindata2 = pd.read_csv('../history-nodiffusion-2.csv')

traindata2['Epoch'] = traindata2['Epoch']+10
traindata = pd.concat([traindata1,traindata2], ignore_index=True)
'''
#traindata = pd.read_csv('../history-fanal-10cm-larcv_treval.csv')

MEMBERWORK = "/gpfs/alpine/scratch/echurch"
traindata = pd.read_csv(MEMBERWORK+'/nph133/echurch/next1t/history-fanal-10mm-larcv2.csv')
scoredata = pd.read_csv(MEMBERWORK+'/nph133/echurch/next1t/history-fanal-10mm-larcv_scores2.csv')
nepoch = len(traindata.groupby('Epoch'))

plt.figure(figsize=(8,5))

plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').mean()['Accuracy'], yerr=traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').std()['Accuracy'],fmt='-',color='navy',label='Training')
plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').mean()['Accuracy'], yerr=traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').std()['Accuracy'],fmt='-',color='tomato',label='Validation')

#pdb.set_trace()
plt.ylabel('Accuracy', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()

plt.savefig('./plots/trainingaccuracy_fanal_10mm_larcv.pdf',bbox_inches="tight")


plt.figure(figsize=(8,5))

plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').mean()['Loss'], yerr=traindata[(traindata['Training_Validation'] == 'Training')].groupby('Epoch').std()['Loss'],fmt='-',color='navy',label='Training')
plt.errorbar(range(nepoch), traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').mean()['Loss'], yerr=traindata[(traindata['Training_Validation'] == 'Validation')].groupby('Epoch').std()['Loss'],fmt='-',color='tomato',label='Validation')

plt.ylabel('Loss', fontsize=12)
plt.xlabel('Epoch', fontsize=12)
plt.legend(loc='upper right', fontsize=12)

plt.tight_layout()

plt.savefig('./plots/trainingloss_fanal_10mm_larcv.pdf',bbox_inches="tight")






minx  = min(min(scoredata['Score0']),min(scoredata['Score1']))
maxx  = max(max(scoredata['Score0']),max(scoredata['Score1']))
scores = np.linspace(minx, maxx, num=200)

pur = np.empty([0])
eff = np.empty([0])
for score in scores:
    numchosenright = ( scoredata[(scoredata['Class']==1) & (scoredata['Score1']>=score)] ).shape[0]
    numchosen = (scoredata['Score1']>=score).sum()
    numposs = ((scoredata['Class']==1)|(scoredata['Class']==0)).sum()


    if numchosen and numposs:
        pur = np.append(pur,numchosenright / numchosen)
        eff = np.append(eff,numchosen / numposs)


plt.figure(figsize=(8,5))

plt.plot(eff,pur,color='tomato',label='ROC curve')

plt.ylabel('Purity', fontsize=12)
plt.xlabel('Efficiency', fontsize=12)
plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()

plt.savefig('./plots/validationroc_fanal_10mm_larcv.pdf',bbox_inches="tight")

