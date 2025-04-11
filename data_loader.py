# -*- coding: utf-8 -*-
import time

import numpy as np
import pandas as pd
import utils
import os
import sys

def GetDisMatrix_Baidu(s_begin, s_end, d_begin, d_end):
    # 读取数据
    supply_ori = pd.read_csv('./data/hz/Health.csv', engine='python', encoding='utf-8-sig')  # 医院数据
    demand_ori = pd.read_csv('./data/hz/Area.csv', engine='python', encoding='utf-8-sig')  # 小区数据
    
    all_supply_spatial = supply_ori[['lat_bd','lon_bd']].values[s_begin: s_end] 
    all_demand_spatial = demand_ori[['lat_bd','lon_bd']].values[d_begin: d_end]
    supply_num = len(all_supply_spatial)
    demand_num = len(all_demand_spatial)

    print(f'tot {supply_num} hospitals and {demand_num} areas')
    
    # 因为百度api限制了每次只能计算50个，所以每50个小区数据计算一次
    supply_step = 1
    demand_step = 50
    dist_matrix = np.zeros((demand_num, supply_num))
    duration_matrix = np.zeros((demand_num, supply_num))

    # 计算距离矩阵
    for i in range(0, len(all_supply_spatial), supply_step):
        supply_spatial = all_supply_spatial[i: i+supply_step]
        for j in range(0, len(all_demand_spatial), demand_step):
            demand_spatial = all_demand_spatial[j: j+demand_step]
            print(f'calc distance of hos_i [{i}, {i+len(supply_spatial)}) to area_j [{j}, {j+len(demand_spatial)})')
            
            cal = utils.CalDisMatrix_Baidu(demand_spatial, supply_spatial)
            dist, duration = cal.CalDistance()
            if dist is None:
                break
            dist_matrix[j:j+len(demand_spatial), i:i+len(supply_spatial)] = dist
            duration_matrix[j:j+len(demand_spatial), i:i+len(supply_spatial)] = dist
            
        if i % 5 == 0:
            print(f'sleep ...')
            time.sleep(2.0)

    dist_sort, dist_sortid = utils.GetSort(dist_matrix)  
    duration_sort, duration_sortid = utils.GetSort(duration_matrix)  

    # 保存距离矩阵
    dist_df = (pd.DataFrame(dist_matrix,index = None)) * 0.001  
    dist_sort_df = (pd.DataFrame(dist_sort,index = None)) * 0.001 
    dist_sortid_df = pd.DataFrame(dist_sortid,index = None)
    utils.SaveCSV(dist_df, f'./data/hz/dist/Dist_{d_begin:05d}_{d_end:05d}.csv')
    utils.SaveCSV(dist_sort_df, f'./data/hz/dist/Dist_sort_{d_begin:05d}_{d_end:05d}.csv')
    utils.SaveCSV(dist_sortid_df, f'./data/hz/dist/Dist_sortid_{d_begin:05d}_{d_end:05d}.csv')
    
    # 保存时间矩阵
    duration_df = (pd.DataFrame(duration_matrix,index = None)) 
    duration_sort_df = (pd.DataFrame(duration_sort,index = None)) 
    duration_sortid_df = pd.DataFrame(duration_sortid,index = None)
    utils.SaveCSV(duration_df, f'./data/hz/dist/Duration_{d_begin:05d}_{d_end:05d}.csv')
    utils.SaveCSV(duration_sort_df, f'./data/hz/dist/Duration_sort_{d_begin:05d}_{d_end:05d}.csv')
    utils.SaveCSV(duration_sortid_df, f'./data/hz/dist/Duration_sortid_{d_begin:05d}_{d_end:05d}.csv')
    
    return

def GetDisMatrix():
     # 读取小区数据
     demand_ori = pd.read_csv('./data/hz/Area.csv', engine='python', encoding='utf-8-sig')[:10]
     demand_spatial = demand_ori[['lon_gc','lat_gc']] 
     
     # 计算小区之间的距离矩阵
     cal = utils.CalDisMatrix(demand_spatial.values, demand_spatial.values)
     area_dist = cal.CalDistance_LonLat()
     area_dist_sort, area_dist_sortid = cal.GetSort()  
     
     # 保存
     area_dist_df = pd.DataFrame(area_dist,index = None)
     area_dist_sort_df = pd.DataFrame(area_dist_sort,index = None)
     area_dist_sortid_df = pd.DataFrame(area_dist_sortid,index = None)

     utils.SaveCSV(area_dist_df, './data/hz/dist/Area_dist_gc.csv', )
     utils.SaveCSV(area_dist_sort_df, './data/hz/dist/Area_dist_sort_gc.csv')
     utils.SaveCSV(area_dist_sortid_df, './data/hz/dist/Area_dist_sortid_gc.csv')
     
if __name__ == "__main__":    
    GetDisMatrix_Baidu(0, 140, 150, 150)
    sys.exit()


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



   