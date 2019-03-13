import pandas as pd
import numpy as np
from scipy import stats
import lightgbm as lgb
import time
import gc

i2 = 0
start = 0
importance=0

def deal_speed(df):
    #print(i2)
    #start = time.time()
    #df = pd.Series()
    #print("--------"+str(df.shape)+"------------")
    #print(list(df['O_LINENO'])[0])
    #print(list(df['O_TERMINALNO'])[0])
    #print(list(df['speed'])[0])
    a = list(df['speed'])[0][0:].replace('\r\n','')
    #print(a)
    a = a.split()
    a = [int(i) for i in a]
    df['max'] = max(a)
    df['min'] = min(a)
    df['sum'] = sum(a)
    df['mean'] = np.mean(a)
    df['kurtosis'] = stats.kurtosistest(a)[0]
    df['skew'] = stats.skewtest(a)[0]
    b = [i for i in a if i!=0 and i!=-1]
    for i in range(len(b),8):
        b.append(0)
    df['max2'] = max(b)
    df['min2'] = min(b)
    df['sum2'] = sum(b)
    df['mean2'] = np.mean(b)
    df['kurtosis2'] = stats.kurtosistest(b)[0]
    df['skew2'] = stats.skewtest(b)[0]
    a = np.array(a)
    b = a[1:] - a[:-1]
    df['A_max'] = max(b)
    df['A_min'] = min(b)
    df['A_sum'] = sum(b)
    df['A_mean'] = np.mean(b)
    df['A_kurtosis2'] = stats.kurtosistest(b)[0]
    df['A_skew2'] = stats.skewtest(b)[0]

    df['speed'] = 0
    global i2,start
    #if i2==0:
        #start = time.time()
    if i2%10000==0:
        #end = time.time()
        #gc.enable()
        #gc.collect()
        print(i2)
    i2 += 1
    #print("Done")
    #end = time.time()
    #print(end-start)
    #df['O_LINENO'] = list(DFrame['O_LINENO'])[0]
    #df['O_TERMINALNO'] = list(DFrame['O_TERMINALNO'])[0]
    #df['hour'] = list(DFrame['hour'])[0]
    #df['O_DATE'] = list(DFrame['O_DATE'])[0]
    #df['O_LINENO'] = DFrame.iloc[0,0]
    #df['O_TERMINALNO'] = DFrame.iloc[0,1]
    #df['hour'] = DFrame.iloc[0,2]
    #df['O_DATE'] = DFrame.iloc[0,4]
    #end2 = time.time()
    #print(end-start)
    #print(end2-end)
    #df = df.drop('speed',1)
    #print(df)
    return df

def deal_predict(test):
    start = time.time()
    predict = pd.read_csv('toBePredicted_forUser.csv')
    predict['Hour'] = predict['predHour'].str.split(':',expand=True)[0]
    predict['Hour'] = predict['Hour'].astype(np.int8)
    pred_timestep = []
    for i in range(predict.shape[0]):
        #begin = time.time()
        t = []
        t2 = 0
        t3 = []
        #t2 = ''
        tmp = test[test.O_TERMINALNO==predict.loc[i,'O_TERMINALNO']]
        tmp = tmp[(tmp.O_LINENO==predict.loc[i,'O_LINENO'])]
        #end3 = time.time()
        tmp = tmp[(tmp.O_DATA==predict.loc[i,'O_DATA']) & (tmp.O_TIME==predict.loc[i,'Hour']) & (tmp.O_UP==predict.loc[i,'O_UP'])]
        #end = time.time()
        j = predict.loc[i,'pred_start_stop_ID']
        tmp2 = tmp[(tmp.O_START==j-1) & (tmp.O_END==j)]
        t2 += tmp2.iloc[0,-1]/2
        t.append(t2/2)
        t3.append(t2/2)
        for j in range(predict.loc[i,'pred_start_stop_ID']+1,predict.loc[i,'pred_end_stop_ID']+1):
            tmp2 = tmp[(tmp.O_START==j-1) & (tmp.O_END==j)]
            #print(tmp2)
            t3.append(tmp2.iloc[0,-1])
            t2 += tmp2.iloc[0,-1]
            if t2>3600:
                t2=3570-(predict.loc[i,'pred_end_stop_ID']-j)*np.mean(t3)
            t.append(t2)
            #t2 = t2+str(t)+';'
        pred_timestep.append(str(t)[1:-1].replace(',',';'))
        #end2 = time.time()
        #print(end3-begin)
        #print(end-end3)
        #print(end2-end)
        #print(str(t)[1:-1].replace(',',';'))
    predict = predict.drop(['Hour','O_UP'],1)
    predict['pred_timeStamps'] = pred_timestep
    predict.to_csv('result17.csv',index=False)
    end = time.time()
    t = end-start
    m,s = divmod(t,60)
    h,m = divmod(m,60)
    print("%02d:%02d:%02d"%(h,m,s))


def lgb_train(df_train,df_test,test):
    #df_train = df_train.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    #df_test = df_test.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    #test = test.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    gc.enable()
    df_train_y = df_train['O_TIMESTEP']
    df_test_y = df_test['O_TIMESTEP']
    df_train = df_train.drop(['O_DISTANCE','O_TIMESTEP'],1)#,'O_LINENO', 'O_TERMINALNO'],1)
    df_test = df_test.drop(['O_DISTANCE','O_TIMESTEP'],1)#,'O_LINENO', 'O_TERMINALNO'],1)
    print(test.columns)
    test = test.drop(['O_DATA','predHour'],1)
    
    df_train['O_DATE'] = df_train['O_DATE']%7
    df_test['O_DATE'] = df_test['O_DATE']%7
    test['O_DATE'] = test['O_DATE']%7
    index = ['O_LINENO', 'O_END', 'O_START', 'O_TERMINALNO','O_UP', 'mean2', 'A_kurtosis', 'kurtosis2', 'skew2', 'A_skew', 'kurtosis', 'O_TIME']
    #index = ['O_LINENO', 'O_END', 'O_START', 'O_TERMINALNO','O_UP', 'mean2', 'A_kurtosis', 'kurtosis2', 'skew2', 'A_skew']
    df_train = df_train[index]
    df_test = df_test[index]
    test = test[index]
    gc.collect()
    #test = test.drop(['O_LINENO', 'O_TERMINALNO'],1)
    #Max = []
    #Min = []
    #Mean = []
    #Std = []
    """
    for i in test.columns:
        Max = max([max(df_train[i]),max(df_test[i]),max(test[i])])
        Min = min([min(df_train[i]),min(df_test[i]),min(test[i])])
        df_train[i] = (df_train[i]-Min)/(Max-Min)
        df_test[i] = (df_test[i]-Min)/(Max-Min)
        test[i] = (test[i]-Min)/(Max-Min)
    """
    #Max=max([max(df_train_y),max(df_test_y)])
    #Min=min([min(df_train_y),min(df_test_y)])
    #df_train_y = (df_train_y-Min)/(Max-Min)
    #df_test_y = (df_test_y-Min)/(Max-Min)
    """
    lgb_train = lgb.Dataset(df_train,df_train_y)
    lgb_eval = lgb.Dataset(df_test,df_test_y,reference=lgb_train)
    params = {  
        'boosting_type': 'gbdt',  
        'objective': 'regression',  
        #'metric': {'binary_logloss', 'auc'},
        #'metric': 'multi_error',
        'metric': 'auc',
        #'num_class': 500,
        'max_depth': 500,
        'num_leaves' : 3000,
        #'min_data_in_leaf': 50,  
        'learning_rate': 0.05,  
        #'feature_fraction': 0.8,  
        #'bagging_fraction': 0.8,0  
        'bagging_freq': 5,
        'lambda_l1': 0.2,    
        'lambda_l2': 0.1,  # 越小l2正则程度越高  
        'min_gain_to_split': 0.2,  
        'verbose': 5,  
        'is_unbalance': True  
    }
    gbm = lgb.train(params,  
            lgb_train,  
            num_boost_round=10000,  
            valid_sets=lgb_eval,  
            early_stopping_rounds=500)
    test = test.drop(['O_DATA','predHour'],1)
    pred = gbm.predict(test,num_iteration=gbm.best_iteration)
    """
    model = lgb.LGBMRegressor(n_estimators=1000000,learning_rate = 0.4)
    model.fit(df_train,df_train_y,eval_set=[(df_test,df_test_y)],eval_metric='l1',early_stopping_rounds = 100,verbose= 100)
    global importance
    importance = model.feature_importances_
    print(model.feature_importances_)
    best_iteration = model.best_iteration_
    pred = model.predict(test,num_iteration=best_iteration)
    a = pd.DataFrame()
    b = list(test.columns)
    a['feature'] = b
    a['importance'] = model.feature_importances_
    a = a.sort_values('importance',ascending=False)
    print(a)
    a['importance'] = a['importance']/sum(a['importance'])
    print(a)
    #test['TIMESTEP'] = pred
    #pred = pred*(Max-Min)+Min
    #pred = pred+Min
    return pred,model,a

if __name__ == '__main__':
    print("开始时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    #9-16
    """
    speed_9_16 = pd.read_csv('./data/speed_9_16.csv')
    #speed_9_16 = speed_9_16.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    #speed_1 = speed_9_16.iloc[0:150000,]
    #speed_2 = speed_9_16.iloc[150001:300000,]
    speed_3 = speed_9_16.iloc[300001:450000,]
    speed_4 = speed_9_16.iloc[450001:,]
    speed_1 = speed_1.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_1.to_csv('speed_9_16_1.csv',index=False)
    del speed_1
    gc.collect()
    speed_2 = speed_2.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_2.to_csv('speed_9_16_2.csv',index=False)
    del speed_2
    gc.collect()
    speed_3 = speed_3.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_3.to_csv('speed_9_16_3.csv',index=False)
    del speed_3
    gc.collect()
    speed_4 = speed_4.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_4.to_csv('speed_9_16_4.csv',index=False)
    del speed_4
    gc.collect()
    speed_1 = pd.read_csv('speed_9_16_1.csv')
    speed_2 = pd.read_csv('speed_9_16_2.csv')
    speed_3 = pd.read_csv('speed_9_16_3.csv')
    speed_4 = pd.read_csv('speed_9_16_4.csv')
    speed_9_16 = pd.concat([speed_1,speed_2,speed_3,speed_4])
    for i in ['max','min','sum','mean','kurtosis','skew','max2','min2','sum2','mean2','kurtosis2','skew2','A_max','A_min','A_sum','A_mean','A_kurtosis2','A_skew2']:
        speed_9_16[i] = speed_9_16[i].astype(np.float16)
    gc.enable()
    speed_9_16 = speed_9_16.drop('speed',1)
    speed_9_16.to_csv('speed_9_16.csv',index=False)
    gc.collect()    
    """
    #17-24
    """
    speed_17_24 = pd.read_csv('./data/speed_17_24.csv')
    #speed_17_24 = speed_17_24.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    speed_1 = speed_17_24.iloc[0:100000,]
    speed_2 = speed_17_24.iloc[100001:200000,]
    speed_3 = speed_17_24.iloc[200001:300000,]
    speed_4 = speed_17_24.iloc[300001:,]
    speed_1 = speed_1.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_1.to_csv('speed_17_24_1.csv',index=False)
    del speed_1
    gc.collect()
    speed_2 = speed_2.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_2.to_csv('speed_17_24_2.csv',index=False)
    del speed_2
    gc.collect()
    speed_3 = speed_3.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_3.to_csv('speed_17_24_3.csv',index=False)
    del speed_3
    gc.collect()
    speed_4 = speed_4.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    gc.enable()
    speed_4.to_csv('speed_17_24_4.csv',index=False)
    del speed_4
    gc.collect()
    speed_1 = pd.read_csv('speed_17_24_1.csv')
    speed_2 = pd.read_csv('speed_17_24_2.csv')
    speed_3 = pd.read_csv('speed_17_24_3.csv')
    speed_4 = pd.read_csv('speed_17_24_4.csv')
    speed_17_24 = pd.concat([speed_1,speed_2,speed_3,speed_4])
    for i in ['max','min','sum','mean','kurtosis','skew','max2','min2','sum2','mean2','kurtosis2','skew2','A_max','A_min','A_sum','A_mean','A_kurtosis2','A_skew2']:
        speed_17_24[i] = speed_17_24[i].astype(np.float16)
    gc.enable()
    speed_17_24 = speed_17_24.drop('speed',1)
    speed_17_24.to_csv('speed_17_24.csv',index=False)
    gc.collect()
    """
    #25-31
    """
    speed_25_31 = pd.read_csv('./data/speed_25_31.csv')
    speed_25_31 = speed_25_31.groupby(['O_LINENO','O_TERMINALNO','hour','O_DATE']).apply(deal_speed)
    for i in ['max','min','sum','mean','kurtosis','skew','max2','min2','sum2','mean2','kurtosis2','skew2','A_max','A_min','A_sum','A_mean','A_kurtosis2','A_skew2']:
        speed_25_31[i] = speed_25_31[i].astype(np.float16)
    gc.enable()
    speed_25_31 = speed_25_31.drop('speed',1)
    speed_25_31.to_csv('speed_25_31.csv',index=False)
    gc.collect()
    """
    #speed合并
    """
    speed_9_16 = pd.read_csv('speed_9_16.csv')
    speed_17_24 = pd.read_csv('speed_17_24.csv')
    speed_25_31 = pd.read_csv('speed_25_31.csv')
    speed = pd.concat([speed_9_16,speed_17_24,speed_25_31])
    speed['hour'] = speed['hour']+1
    speed.columns = ['O_LINENO','O_TERMINALNO','O_TIME','O_DATE','max','min','sum','mean','kurtosis','skew','max2','min2','sum2','mean2','kurtosis2','skew2','A_max','A_min','A_sum','A_mean','A_kurtosis','A_skew']
    print("speed处理完毕: " + str(time.strftime("%m-%d %H:%M:%S",time.localtime())))   
    speed.to_csv('speed_all.csv',index=False)
    gc.enable()
    del speed, speed_17_24, speed_25_31, speed_9_16
    gc.collect()
    """
    
    speed = pd.read_csv('./speed_all.csv')
    for i in ['max','min','sum','mean','kurtosis','skew','max2','min2','sum2','mean2','kurtosis2','skew2','A_max','A_min','A_sum','A_mean','A_kurtosis','A_skew']:
        speed[i] = speed[i].astype(np.float16)
    df_train = pd.read_csv('./data/fun2_all_9_16.csv')
    df_test = pd.read_csv('./data/fun2_all_17_24.csv')
    df_train = df_train[(df_train.O_START != -1)]
    df_train = df_train[(df_train.O_TIMESTEP>=50) & (df_train.O_TIMESTEP<=350)]
    df_test = df_test[(df_test.O_START != -1)]
    df_test = df_test[(df_test.O_TIMESTEP>=50) & (df_test.O_TIMESTEP<=350)]
    test = pd.read_csv('./predict.csv')
    #for i in ['O_LINENO','O_TERMINALNO','O_TIME','O_DATE']:
    #    df_train[i] = df_train[i].astype(int)
    #    df_test[i] = df_test[i].astype(int)
    gc.enable()
    for i in ['O_START', 'O_END', 'O_TIMESTEP','O_UP', 'O_DISTANCE']:
        df_train[i] = df_train[i].astype(np.int16)
        df_test[i] = df_test[i].astype(np.int16)
    df_train = df_train.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    #print(df_train.iloc[0,])
    df_test = df_test.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    #speed_25_31.columns = ['O_LINENO','O_TERMINALNO','O_TIME','O_DATE','max','min','sum','mean']
    test = test.merge(speed,how='left',on=['O_LINENO','O_TERMINALNO','O_TIME','O_DATE'])
    #test2 = test[['O_LINENO', 'O_TERMINALNO', 'O_START', 'O_END','O_UP', 'max', 'min', 'sum', 'mean']]
    gc.collect()
    """
    df_train_1 = df_train.iloc[0:300000,]
    #df_test_1 = df_test.iloc[0:300000,]
    pred1,model1,a = lgb_train(df_train_1,df_test,test)
    print("时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    df_train_2 = df_train.iloc[300001:,]
    #df_test_2 = df_test.iloc[300001:,]
    pred2,model2,a = lgb_train(df_train_2,df_test,test)
    print("时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    
    #pred,model,a = lgb_train(df_train,df_test,test)
    #pred2 = np.array(pred)
    #pred2[pred2<30] = 30
    #test['O_TIMESTEP'] = pred2
    #deal_predict(test)
    print("结束时间:"+str(time.strftime("%m-%d %H:%M:%S",time.localtime())))
    """
    #print(df_train.iloc[0,])
    test2 = test[['O_LINENO','O_TERMINALNO']]
    test2['exist'] = 1
    test2 = test2.drop_duplicates()
    df_train = df_train.merge(test2,how='left',on=['O_LINENO','O_TERMINALNO'])
    df_test = df_test.merge(test2,how='left',on=['O_LINENO','O_TERMINALNO'])
    df_train = df_train[df_train.exist == 1]
    df_test = df_test[df_test.exist == 1]
    df_train['A'] = df_train['O_START']-df_train['O_END']
    df_test['A'] = df_test['O_START']-df_test['O_END']
    df_train = df_train[df_train.A == -1]
    df_test = df_test[df_test.A == -1]
    
