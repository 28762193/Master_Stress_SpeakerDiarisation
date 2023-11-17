# Format WESAD Data: Pickel file to csv.
import pandas as pd
import numpy as np
import glob

listFiles = glob.glob('/WESAD/S*/S*.pkl')
for i in listFiles:
    df = pd.read_pickle(i)

    data = np.concatenate((df['signal']['chest']['ACC'],df['signal']['chest']['ECG'],df['signal']['chest']['Resp']),axis=1)

    df2 = pd.DataFrame(data,columns=('X-Axis','Y-Axis','Z-Axis','ECG','Resp'))
    df2['Labels'] = df['label']
    df2.drop(df2[(df2['Labels']==0) | (df2['Labels']==5) | (df2['Labels']==6) | (df2['Labels']==7)].index,inplace=True)
    # The code below adds label names to the corresponding label integer:
    #df2['Label_Names'] = np.where(df2['Labels']==1,'Baseline',np.where(df2['Labels']==2,'Stress',np.where(df2['Labels']==3,'Amusement',np.where(df2['Labels']==4,'Meditation',False))))

    fileLoc = '/WESAD/MAT_Files/'
    fileNum = df['subject'] + '.csv'
    fileName = fileLoc + fileNum

    df2.to_csv(fileName,index=False)

