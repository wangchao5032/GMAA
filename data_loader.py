# -*- coding: utf-8 -*-
import pandas as pd
import utils
import os

def LoadCSV(name, chunk):  
    reader = pd.read_csv(name, iterator=True, chunksize=chunk)
    tmp_df = pd.DataFrame()
    for chunk in reader:
        tmp_df = pd.concat([tmp_df, chunk])
        print(f'read file{name}, length is {len(tmp_df)}')
    return tmp_df

def GetDistBetweenMatrix(files, need, data1, data2):
    dist, dist_sort, dist_sort_id = [], [], []
    # 先判断是否保存过一系列文件（默认有一个就有一系列）
    if os.path.exists(files[0]):  # 如果存在，直接读取
        if need[0] == 1:
            dist = LoadCSV(files[0], 5000).values
        if need[1] == 1:
            dist_sort = LoadCSV(files[1], 5000).values
        if need[2] == 1:
            dist_sort_id = LoadCSV(files[2], 5000).values
    else:  # 如果不存在，计算
        dist, dist_sort, dist_sort_id = CalDist(data1, data2)
        # 保存
        dist_df = pd.DataFrame(dist,index = None)
        dist_sort_df = pd.DataFrame(dist_sort,index = None)
        dist_sort_id_df = pd.DataFrame(dist_sort_id, index = None)
        
        dist_df.to_csv(files[0] ,encoding='utf_8_sig', index=False)
        print(f'save {files[0]}')
        dist_sort_df.to_csv(files[1] ,encoding='utf_8_sig', index=False)
        print(f'save {files[1]}')
        dist_sort_id_df.to_csv(files[2] ,encoding='utf_8_sig', index=False)
        print(f'save {files[2]}')
    return dist, dist_sort, dist_sort_id

# 计算两个矩阵之间的距离，排序  
def CalDist(data1, data2):
    cal = utils.CalDisMatrix(data1, data2) 
    # 计算距离矩阵
    dist = cal.CalDistance_LonLat()  # 输入经纬度，输出的是km
    # 对距离进行排序
    dist_sort, dist_sort_id = cal.GetSort()
    return dist, dist_sort, dist_sort_id



   