import pandas as pd
import time
import other

def get_time(a,b):
    c1 = a.split(':')
    c2 = b.split(':')
    return (int(c2[0])-int(c1[0]))*3600 + (int(c2[1])-int(c1[1]))*60 + int(c2[2])-int(c1[2])


def build(td2):
    td2 = td2.sort_values('O_TIME')
    df = pd.DataFrame(columns=['O_LINENO','O_TERMINALNO','O_START','O_END','O_TIME','O_TIMESTEP','O_UP','O_DISTANCE','O_DATE'])
    O_line = list(td2['O_LINENO'])[0]
    O_term = list(td2['O_TERMINALNO'])[0]
    O_next = list(td2['O_NEXTSTATIONNO'])
    O_time = list(td2['O_TIME'])
    O_speed = list(td2['O_SPEED'])
    O_date = list(td2['O_DATE'])[0]
    O_up = list(td2['O_UP'])
    now = O_next[0]#现在的下一站
    now_time = O_time[0].split(':')[0]#现在的时间  小时
    now_time2 = O_time[0]
    now_up = O_up[0]
    now_station = -1   #现在的站点
    start = O_time[0]  #开始的时间
    distance = 0
    for i in range(1,td2.shape[0]):
        if O_next[i] == now:
            distance = distance + get_time(O_time[i-1],O_time[i])*O_speed[i]
        else:
            dic = {'O_LINENO':O_line,'O_TERMINALNO':O_term,'O_START':now_station,'O_END':now,'O_TIME':now_time,'O_UP':now_up,'O_TIMESTEP':get_time(now_time2,O_time[i]),'O_DISTANCE':distance,'O_DATE':O_date}
            #print(i)
            df = df.append(dic,ignore_index=True)
            now_station = now
            now = O_next[i]
            now_time = O_time[i].split(':')[0]
            now_time2 = O_time[i]
            distance = 0
            now_up = O_up[i]
    #print(O_line)
    #print(O_term)
    #try:
    dic = {'O_LINENO':O_line,'O_TERMINALNO':O_term,'O_START':now_station,'O_END':now,'O_TIME':now_time,'O_UP':now_up,'O_TIMESTEP':get_time(now_time2,O_time[td2.shape[0]-1]),'O_DISTANCE':distance,'O_DATE':O_date}
    #except:
    #    print(now_time2)
    #    print(get_time(now_time2,O_time[td2.shape[0]-1]))
    #    print(O_line)
    #    print(O_term)
    df = df.append(dic,ignore_index=True)
    return df

if __name__ == '__main__':
    for i in range(25,32):
        print(i)
        a = './train25-31/train201710%s.csv'%i
        train = pd.read_csv(a)
        train = train.drop(['O_LONGITUDE','O_LATITUDE','O_MIDDOOR','O_REARDOOR','O_FRONTDOOR','O_RUN'],1)
        start = time.time()
        train['O_DATE'] = i
        f = train.groupby(['O_LINENO','O_TERMINALNO']).apply(build)
        f = f.reset_index(drop=1)
        f.to_csv('./data/fun2_all_25_31.csv',index=False,mode='a+',header=False)
        end = time.time()
        other.get_time(end-start)
        del train
        
    """
    train = pd.read_csv('./train9-16/train20171012.csv')
    start = time.time()
    #train = train.sort_values('O_LINENO','O_TERMINALNO')
    train = train.drop(['O_LONGITUDE','O_LATITUDE','O_MIDDOOR','O_REARDOOR','O_FRONTDOOR','O_RUN'],1)
    train['O_DATE'] = 12
    f = train.groupby(['O_LINENO','O_TERMINALNO']).apply(build)
    f = f.reset_index(drop=1)
    f.to_csv('fun2_all.csv',index=False,mode='a+')
    end = time.time()
    get_time2(end-start)
    """




















