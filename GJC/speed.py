import pandas as pd
import numpy as np
from numpy import *
import time
import other

import gc

def speed(DFrame):
    df = pd.Series()
    #for i in range(60):
    #    for j in range(6):
    #        df[str(i)+'_'+str(j)] = [-1]
    #        df[str(i)+'_'+str(j)] = df[str(i)+'_'+str(j)].asytpe(np.int16)
    #for i in range(DFrame.shape[0]):
        #column = str(DFrame.loc[i,'MIN'])+'_'+str(DFrame.loc[i,'SEC'])
        #df[column] = [DFrame.loc[i,'O_SPEED']]
    #DFrame['MIN'] = DFrame['O_TIME'].str.split(':',expand=True)[1]
    #DFrame['SEC'] = DFrame['O_TIME'].str.split(':',expand=True)[2]
    DFrame['MIN'] = [i[3:5] for i in DFrame['O_TIME']]
    DFrame['SEC'] = [i[6:8] for i in DFrame['O_TIME']]
    DFrame['MIN'] = DFrame['MIN'].astype(np.int8)
    DFrame['SEC'] = DFrame['SEC'].astype(np.int8)
    DFrame['SEC'] = DFrame['SEC']//10
    a = -ones(60*6)
    a = a.astype(int16)
    #DFrame = DFrame.reset_index(drop=1)
    Speed = list(DFrame['O_SPEED'])
    MIN = list(DFrame['MIN'])
    SEC = list(DFrame['SEC'])
    #print(DFrame.head())
    for i in range(DFrame.shape[0]):
        M = MIN[i]
        S = SEC[i]
        a[M*6+S] = Speed[i]#DFrame.loc[i,'O_SPEED']
    b = str(a)[1:-1]
    #print(b)
    df['O_LINENO'] = list(DFrame['O_LINENO'])[0]
    df['O_TERMINALNO'] = list(DFrame['O_TERMINALNO'])[0]
    df['hour'] = list(DFrame['hour'])[0]
    df['speed'] = b
    return df
    
     

if __name__ == '__main__':
    for i in range(9,17):#修改时间段，可获得相应的其他的数据
        if i == 9:
            i = '09'
        a = './train9-16/train201710%s.csv'%i
        train = pd.read_csv(a)
        print("Load Data Complete")
        gc.enable()
        train = train.drop(['O_LONGITUDE','O_LATITUDE','O_MIDDOOR','O_REARDOOR','O_FRONTDOOR','O_UP','O_RUN','O_NEXTSTATIONNO'],1)
        gc.collect()
        #train['hour'] = train['O_TIME'].str.split(':',expand=True)[0]
        train['hour'] = [i[0:2] for i in train['O_TIME']]
        train['hour'] = train['hour'].astype(np.int8)
        train['O_LINENO'] = train['O_LINENO'].astype(np.int32)
        train['O_SPEED'] = train['O_SPEED'].astype(np.int32)
        train['O_TERMINALNO'] = train['O_TERMINALNO'].astype(np.int32)
        #train['O_DATE'] = i
        print("Start GroupBy")
        start = time.time()
        f = train.groupby(['O_LINENO','O_TERMINALNO','hour']).apply(speed)
        f = f.reset_index(drop=1)
        f['O_DATE'] = int(i)
        if i=='09':
            f.to_csv('./data/speed_9_16.csv',index=False,mode='a+',header=True)
        else:
            f.to_csv('./data/speed_9_16.csv',index=False,mode='a+',header=False)
        end = time.time()
        other.get_time(end-start)
