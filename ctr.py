
import time
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
import numpy as np
import gc

def timestamp_datetime(value):
    format = '%Y-%m-%d %H:%M:%S'
    value = time.localtime(value)
    dt = time.strftime(format, value)
    return dt

def convert_data(data):
    data['time'] = data.context_timestamp.apply(timestamp_datetime)
    data['day'] = data.time.apply(lambda x: int(x[8:10]))
    data['hour'] = data.time.apply(lambda x: int(x[11:13]))
    user_query_day = data.groupby(['user_id', 'day']).size(
    ).reset_index().rename(columns={0: 'user_query_day'})
    data = pd.merge(data, user_query_day, 'left', on=['user_id', 'day'])
    user_query_day_hour = data.groupby(['user_id', 'day', 'hour']).size().reset_index().rename(
        columns={0: 'user_query_day_hour'})
    data = pd.merge(data, user_query_day_hour, 'left',
                    on=['user_id', 'day', 'hour'])
    return data

    # 单特征距离上一次点击时间差
def lasttimeDiff(data):
    for column in ['user_id', 'item_id']:
        gc.collect()
        data[column+'_lasttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_lasttime_diff']].values
        lasttime_dict = {}
        for df_list in train_data:
            if df_list[1] not in lasttime_dict:
                df_list[2] = -1
                lasttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = df_list[0] - lasttime_dict[df_list[1]]
                lasttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_lasttime_diff']] = train_data
    return data
    # 单特征距离下一次点击时间差
def nexttimeDiff(data):
    for column in ['user_id', 'item_id']:
        gc.collect()
        data[column+'_nexttime_diff'] = 0
        train_data = data[['context_timestamp', column, column+'_nexttime_diff']].values
        nexttime_dict = {}
        for df_list in train_data:
            if df_list[1] not in nexttime_dict:
                df_list[2] = -1
                nexttime_dict[df_list[1]] = df_list[0]
            else:
                df_list[2] = nexttime_dict[df_list[1]] - df_list[0]
                nexttime_dict[df_list[1]] = df_list[0]
        data[['context_timestamp', column, column+'_nexttime_diff']] = train_data

    return data

# # 同一天点击时间差 以及首次末次与当前时刻的时间差
def doTrick2(data):
    data.sort_values(['user_id', 'context_timestamp'], inplace=True)
    # user_id
    subset = ['user_id', 'day']
    temp = data.loc[:, ['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_first'] = data['context_timestamp'] - data['u_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:, ['context_timestamp', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'u_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['u_day_diffTime_last'] = data['u_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['u_day_diffTime_first', 'u_day_diffTime_last']] = -1

    # item_id
    subset = ['item_id', 'day']
    temp = data.loc[:, ['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_first'] = data['context_timestamp'] - data['i_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:, ['context_timestamp', 'item_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 'i_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['i_day_diffTime_last'] = data['i_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['i_day_diffTime_first', 'i_day_diffTime_last']] = -1
    # item_brand_id, user_id
    subset = ['item_brand_id', 'user_id', 'day']
    temp = data.loc[:, ['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset,
                                                                                                 keep='first')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_first'] = data['context_timestamp'] - data['b_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:, ['context_timestamp', 'item_brand_id', 'user_id', 'day']].drop_duplicates(subset=subset,
                                                                                                 keep='last')
    temp.rename(columns={'context_timestamp': 'b_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['b_day_diffTime_last'] = data['b_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['b_day_diffTime_first', 'b_day_diffTime_last']] = -1

    # shop_id, user_id
    subset = ['shop_id', 'user_id', 'day']
    temp = data.loc[:, ['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='first')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_first'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_first'] = data['context_timestamp'] - data['s_day_diffTime_first']
    del temp
    gc.collect()
    temp = data.loc[:, ['context_timestamp', 'shop_id', 'user_id', 'day']].drop_duplicates(subset=subset, keep='last')
    temp.rename(columns={'context_timestamp': 's_day_diffTime_last'}, inplace=True)
    data = pd.merge(data, temp, how='left', on=subset)
    data['s_day_diffTime_last'] = data['s_day_diffTime_last'] - data['context_timestamp']
    del temp
    gc.collect()
    data.loc[~data.duplicated(subset=subset, keep=False), ['s_day_diffTime_first', 's_day_diffTime_last']] = -1
    return data

# 下面是组合特征
def zuhe(data):
    for col in ['user_gender_id','user_age_level','user_occupation_id','user_star_level']:
        data[col] = data[col].apply(lambda x: 0 if x == -1 else x)

    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',
                'user_gender_id','user_age_level','user_occupation_id','user_star_level',
                'shop_review_num_level', 'shop_star_level']:
        data[col] = data[col].astype(str)

    print('item两两组合')
    data['sale_price'] = data['item_sales_level'] + data['item_price_level']
    data['sale_collect'] = data['item_sales_level'] + data['item_collected_level']
    data['price_collect'] = data['item_price_level'] + data['item_collected_level']

    print('user两两组合')
    data['gender_age'] = data['user_gender_id'] + data['user_age_level']
    data['gender_occ'] = data['user_gender_id'] + data['user_occupation_id']
    data['gender_star'] = data['user_gender_id'] + data['user_star_level']

    print('shop两两组合')
    data['review_star'] = data['shop_review_num_level'] + data['shop_star_level']


    for col in ['item_sales_level', 'item_price_level', 'item_collected_level',  'sale_price','sale_collect', 'price_collect',
                'user_gender_id', 'user_age_level', 'user_occupation_id', 'user_star_level','gender_age','gender_occ','gender_star',
                'shop_review_num_level','shop_star_level','review_star']:
        data[col] = data[col].astype(int)

    del data['review_star']

    return data


#统计特征
def item(data):
    print('一个item有多少brand,price salse collected level……')
    itemcnt = data.groupby(['item_id'], as_index=False)['instance_id'].agg({'item_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_id'], how='left')

    for col in ['item_brand_id','item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_id'], as_index=False)['instance_id'].agg({str(col) + '_item_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_id'], how='left')
        data[str(col) + '_item_prob']=data[str(col) + '_item_cnt']/data['item_cnt']
    del data['item_cnt']

    print('一个brand有多少price salse collected level……')

    itemcnt = data.groupby(['item_brand_id'], as_index=False)['instance_id'].agg({'item_brand_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_brand_id'], how='left')

    for col in ['item_city_id', 'item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_brand_id'], as_index=False)['instance_id'].agg({str(col) + '_brand_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_brand_id'], how='left')
        data[str(col) + '_brand_prob'] = data[str(col) + '_brand_cnt'] / data['item_brand_cnt']
    del data['item_brand_cnt']

    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
    del data['item_city_cnt']

    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']

    print('一个item_collected_level有多少item_pv_level')

    itemcnt = data.groupby(['item_collected_level'], as_index=False)['instance_id'].agg({'item_coll_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_collected_level'], how='left')
    for col in ['item_pv_level']:
        itemcnt = data.groupby([col, 'item_collected_level'], as_index=False)['instance_id'].agg({str(col) + '_coll_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_collected_level'], how='left')
        data[str(col) + '_coll_prob'] = data[str(col) + '_coll_cnt'] / data['item_coll_cnt']
    del data['item_coll_cnt']

    return data

def user(data):
    print('用户有多少性别')
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')

    for col in ['user_gender_id','user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob']=data[str(col) + '_user_cnt']/data['user_cnt']
    del data['user_cnt']

    print('性别的年龄段，职业有多少')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob']=data[str(col) + '_user_gender_cnt']/data['user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level对应的user_occupation_id，user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob']=data[str(col) + '_user_age_cnt']/data['user_age_cnt']
    del data['user_age_cnt']

    print('user_occupation_id对应的user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob']=data[str(col) + '_user_occ_cnt']/data['user_occ_cnt']
    del data['user_occ_cnt']

    return data

def user_item(data):
    itemcnt = data.groupby(['user_id'], as_index=False)['instance_id'].agg({'user_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_id'], how='left')
    print('一个user有多少item_id,item_brand_id……')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg({str(col)+'_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']

    print('一个user_gender有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col)+'_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']

    print('一个user_age_level有多少item_id,item_brand_id……')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col)+'_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']

    print('一个user_occupation_id有多少item_id,item_brand_id…')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col)+'_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    return data


def user_shop(data):
    print('一个user有多少shop_id,shop_review_num_level……')

    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_id'], how='left')
        data[str(col) + '_user_prob'] = data[str(col) + '_user_cnt'] / data['user_cnt']
    del data['user_cnt']

    print('一个user_gender有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob'] = data[str(col) + '_user_gender_cnt'] / data['user_gender_cnt']
    del data['user_gender_cnt']

    print('一个user_age_level有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob'] = data[str(col) + '_user_age_cnt'] / data['user_age_cnt']
    del data['user_age_cnt']

    print('一个user_occupation_id有多少shop_id,shop_review_num_level……')
    for col in ['shop_id', 'shop_review_num_level', 'shop_star_level']:
        item_shop_cnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg(
            {str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob'] = data[str(col) + '_user_occ_cnt'] / data['user_occ_cnt']
    del data['user_occ_cnt']
    return data

def shop_item(data):
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_id'], as_index=False)['instance_id'].agg({'shop_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_id'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_id'], as_index=False)['instance_id'].agg({str(col)+'_shop_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_id'], how='left')
        data[str(col) + '_shop_prob'] = data[str(col) + '_shop_cnt'] / data['shop_cnt']
    del data['shop_cnt']

    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')
    itemcnt = data.groupby(['shop_review_num_level'], as_index=False)['instance_id'].agg({'shop_rev_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['shop_review_num_level'], how='left')
    for col in ['item_id',
                'item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']:
        item_shop_cnt = data.groupby([col, 'shop_review_num_level'], as_index=False)['instance_id'].agg({str(col)+'_shop_rev_cnt': 'count'})
        data = pd.merge(data, item_shop_cnt, on=[col, 'shop_review_num_level'], how='left')
        data[str(col) + '_shop_rev_prob'] = data[str(col) + '_shop_rev_cnt'] / data['shop_rev_cnt']
    del data['shop_rev_cnt']
    return data


def doSize(data):
    add = pd.DataFrame(data.groupby(["shop_id", "day"]).item_id.nunique()).reset_index()
    add.columns = ["shop_id", "day", "shop_item_unique_day"]
    data = data.merge(add, on=["shop_id", "day"], how="left")

    user_query_day = data.groupby(['user_id', 'day']).size().reset_index().rename(columns={0: 'user_id_query_day'})
    data = pd.merge(data, user_query_day, how='left', on=['user_id', 'day'])

    data['min_10'] = data['minute'] // 10
    data['min_15'] = data['minute'] // 15
    data['min_30'] = data['minute'] // 30
    data['min_45'] = data['minute'] // 45

    # user 不同时间段点击次数
    min10_user_click = data.groupby(['user_id', 'day', 'hour', 'min_10']).size().reset_index().rename(
        columns={0: 'min10_user_click'})
    min15_user_click = data.groupby(['user_id', 'day', 'hour', 'min_15']).size().reset_index().rename(
        columns={0: 'min15_user_click'})
    min30_user_click = data.groupby(['user_id', 'day', 'hour', 'min_30']).size().reset_index().rename(
        columns={0: 'min30_user_click'})
    min45_user_click = data.groupby(['user_id', 'day', 'hour', 'min_45']).size().reset_index().rename(
        columns={0: 'min45_user_click'})

    data = pd.merge(data, min10_user_click, 'left', on=['user_id', 'day', 'hour', 'min_10'])
    data = pd.merge(data, min15_user_click, 'left', on=['user_id', 'day', 'hour', 'min_15'])
    data = pd.merge(data, min30_user_click, 'left', on=['user_id', 'day', 'hour', 'min_30'])
    data = pd.merge(data, min45_user_click, 'left', on=['user_id', 'day', 'hour', 'min_45'])

    del data['min_10']
    del data['min_15']
    del data['min_30']
    del data['min_45']

    return data



if __name__ == "__main__":
    online = False  # 这里用来标记是 线下验证 还是 在线提交
    data = pd.read_csv('round1_ijcai_18_train_20180301.txt', sep=' ')
    data.drop_duplicates(inplace=True)
    data = lasttimeDiff(data)
    data = convert_data(data)
    data = nexttimeDiff(data)
    data = doTrick2(data)
    data=doSize(data)


    data = zuhe(data)
    print('全局统计特征')
    data = item(data)
    data = user(data)
    data = user_item(data)
    data = user_shop(data)
    data = shop_item(data)


    print('data:', data.shape)
    print(data.columns)
    if online == False:
        train = data.loc[data.day < 24]  # 18,19,20,21,22,23,24
        test = data.loc[data.day == 24]  # 暂时先使用第24天作为验证集
    elif online == True:
        train = data.copy()
        test = pd.read_csv('round1_ijcai_18_test_a_20180301.txt', sep=' ')
        test = convert_data(test)

    features = ['item_id', 'item_brand_id', 'item_city_id', 'item_price_level', 'item_sales_level',
                'item_collected_level', 'item_pv_level', 'user_gender_id', 'user_occupation_id',
                'user_age_level', 'user_star_level', 'user_query_day', 'user_query_day_hour',
                'context_page_id', 'hour', 'shop_id', 'shop_review_num_level', 'shop_star_level',
                'shop_review_positive_rate', 'shop_score_service', 'shop_score_delivery', 'shop_score_description','user_id_lasttime_diff',
       'item_id_lasttime_diff','user_id_nexttime_diff',
       'item_id_nexttime_diff', 'u_day_diffTime_first', 'u_day_diffTime_last', 'i_day_diffTime_first',
       'i_day_diffTime_last', 'b_day_diffTime_first', 'b_day_diffTime_last',
       's_day_diffTime_first', 's_day_diffTime_last',]
    print(type(features))
    target = ['is_trade']
    if online == False:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target], feature_name=features,
                categorical_feature=['user_gender_id', ])
        test['lgb_predict'] = clf.predict_proba(test[features], )[:, 1]

        print(log_loss(test[target], test['lgb_predict']))
    else:
        clf = lgb.LGBMClassifier(num_leaves=63, max_depth=7, n_estimators=80, n_jobs=20)
        clf.fit(train[features], train[target],categorical_feature=['user_gender_id', ])
        test['predicted_score'] = clf.predict_proba(test[features])[:, 1]
        test[['instance_id', 'predicted_score']].to_csv('baseline1.csv', index=False, sep=' ')  # 提交结果
