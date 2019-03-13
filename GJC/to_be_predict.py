import pandas as pd
import time

to_be_predict = pd.read_csv('toBePredicted_forUser.csv')
to_be_predict['time'] = to_be_predict['predHour'].str.split(':',expand=True)[0]
to_be_predict['time'] = to_be_predict['time'].astype(int)
to_be_predict['day'] = to_be_predict['O_DATA'].str.split('-',expand=True)[1]
to_be_predict['day'] = to_be_predict['day'].astype(int)

d = None

def get_recent(DFrame):
    global data
    global d
    """
    date = list(DFrame['O_DATA'])[0].replace('-','')
    date = './train25-31/train2017%s.csv'%date
    data = pd.read_csv(date)
    data = data.drop(['O_LONGITUDE','O_LATITUDE','O_MIDDOOR','O_REARDOOR','O_FRONTDOOR','O_RUN','O_SPEED','O_UP'],1)
    data['O_TIME2'] = data['O_TIME'].str.split(':',expand=True)[0]
    data['O_TIME2'] = data['O_TIME2'].astype(int)
    """
    O_LINENO = list(DFrame['O_LINENO'])
    O_TERMINALNO = list(DFrame['O_TERMINALNO'])
    Time = list(DFrame['time'])
    Start = list(DFrame['pred_start_stop_ID'])
    Day = list(DFrame['day'])
    Recent_station = []
    
    for i in range(DFrame.shape[0]):
        data2 = data[(data.O_LINENO == O_LINENO[i])]
        data2 = data2[(data2.O_TERMINALNO == O_TERMINALNO[i])]
        d = data2
        data2 = data2[(data2.day == Day[i])]
        #print(list(DFrame['O_DATA'])[i])
        #print(data2.head())
        #print(O_LINENO[i])
        #print(O_TERMINALNO[i])
        #print(list(DFrame['predHour'])[i])
        #print(Time[i])
        #print(Start[i])
        if data2.shape[0] == 0:
            Recent_station.append(-1)
            Time[i] = ''
            continue
        data2['O_TIME2'] = data2['O_TIME'].str.split(':',expand=True)[0]
        data2['O_TIME2'] = data2['O_TIME2'].astype(int)
        data2 = data2[(data2.O_TIME2 == Time[i]-1)]
        data2 = data2.sort_values('O_TIME')
        #data3 = data2[(data2.O_NEXTSTATIONNO == Start[i])]
        #if data3.shape[0] == 0:
        #    if Start[i] == 2:
        #        station = data2.iloc[-1,]['O_NEXTSTATIONNO']
        #        data3 = data2[(data2.O_NEXTSTATIONNO == station)]
        #    else:
        #        data3 = data2[(data2.O_NEXTSTATIONNO == Start[i]-1)]
        #        Recent_station.append(Start[i]-1)
        #else:
        #    Recent_station.append(/Start[i])
        if data2.shape[0] > 0:
            station = data2.iloc[-1,]['O_NEXTSTATIONNO']
            data2 = data2[(data2.O_NEXTSTATIONNO == station)]
            data2 = data2.sort_values('O_TIME').reset_index(drop=1)
            Recent_station.append(station)
            Time[i] = data2.iloc[0,]['O_TIME']
        #print(list(DFrame['O_DATA'])[i])
        #print(data2.head())
        #print(O_LINENO[i])
        #print(O_TERMINALNO[i])
        #print(list(DFrame['predHour'])[i])
        #print(Time[i])
        #print(Start[i])
        else:
            Recent_station.append(-1)
            Time[i] = ''
            
    print('OK?')
    DFrame['station'] = Recent_station
    DFrame['time'] = Time
    return DFrame


if __name__ == '__main__':
    #data = pd.DataFrame(columns=['O_LINENO','O_TERMINALNO','O_TIME','O_NEXTSTATIONNO','day'])
    #data.to_csv('all_25_31.csv',index=False)
    #for i in range(25,32):
    #    print(i)
    #    data_new = pd.read_csv('./train25-31/train201710%s.csv'%i)
    #    data_new = data_new.drop(['O_LONGITUDE','O_LATITUDE','O_MIDDOOR','O_REARDOOR','O_FRONTDOOR','O_RUN','O_SPEED','O_UP'],1)
    #    data_new['day'] = i
    #    data_new.to_csv('all_25_31.csv',index=False,header=False,mode='a+')
        #data = pd.concat([data,data_new])
        #del data_new
    data = pd.read_csv('all_25_31.csv')
    #data['O_TIME2'] = data['O_TIME'].str.split(':',expand=True)[0]
    #data['O_TIME2'] = data['O_TIME2'].astype(int)
    print("DONE")
    #to_be_predict = to_be_predict[to_be_predict.day == 26]
    print("开始时间: " + str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    to_be_predict = to_be_predict.groupby('O_DATA').apply(get_recent)
    to_be_predict.to_csv('to_be_predict.csv',index=False)
    print("结束时间: " + str(time.strftime("%m-%d %H:%M:%S",time.localtime())))

        
