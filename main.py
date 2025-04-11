# %% -*- coding: utf-8 -*-
import sys
sys.path.append("tool")
import random
import os
import numpy as np
import utils
import data_loader as dl
import pandas as pd
import model
from model import gp
import pickle

if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)

    demand_ori = pd.read_csv('sample_data/Area.csv', encoding='utf-8-sig')
    demand_spatial = demand_ori[['lon_gc','lat_gc']]
    
    supply_ori = pd.read_csv('sample_data/Health.csv', encoding='utf-8-sig')
    supply_spatial = supply_ori[['lon_gc','lat_gc']]
    
    files = ['sample_data/Area_dist.csv', 'sample_data/Area_dist_sort.csv', 'sample_data/Area_dist_sortid.csv']
    need = [0,1,1]
    _, gp.area_dist_sort, gp.area_dist_sort_id = dl.GetDistBetweenMatrix(files, need, demand_spatial.values, demand_spatial.values)
   
    files = ['sample_data/Area_Hos_dist.csv', 'sample_data/Area_Hos_dist_sort.csv', 'sample_data/Area_Hos_dist_sortid.csv']
    need = [1,0,1]
    gp.distance_matrix, _, gp.distance_matrix_id = dl.GetDistBetweenMatrix(files, need, demand_spatial.values, supply_spatial.values)
    
    gp.area_num = demand_ori.shape[0]
    gp.hos_num = supply_ori.shape[0]
    
    for i in range(gp.area_num):
        if i % 1000 == 0: print(f'init res {i}')
        loc = demand_ori.loc[i]
        x, y, district = loc['lon_gc'], loc['lat_gc'], loc['hz_name']
        area = model.Area(i, x, y, district)
        gp.areas.append(area)
    
    gp.res_num = len(gp.residents)
    res_of_interest = model.SetBelief(gp.select_areas)  
    gp.InitRecordRes(res_of_interest)
    
    for i in range(gp.area_num):
        if i % 1000 == 0: print(f'init neighbour of res {i}')
        area = gp.areas[i]
        area.FindNeighborsArea()

    for i in range(gp.hos_num):
        if i % 50 == 0: print(f'init hos {i}')
        loc = supply_ori.loc[i]  
        x, y, level, beds = loc['lon_gc'], loc['lat_gc'], loc['level'], loc['beds']
        hos = model.Hospital(i, x, y, level, beds)
        gp.hospitals.append(hos)
    gp.good_hospital_ids = [i for i in range(gp.hos_num) if gp.hospitals[i].level == 6]
    print(f'all init done ...')

    print('free unused variables')
    gp.area_dist_sort_id = None
    gp.area_dist_sort = None
    demand_ori = None
    supply_ori = None
    demand_spatial = None
    supply_spatial = None

    print(f'start iteration ...')
    for p in range(gp.iteration):
        print(f'iter {p} ...')
        
        # step 1
        for i in range(gp.res_num):
            res = gp.residents[i]
            res.FindHospital(p)

        # step 2
        for i in range(gp.res_num):
            res = gp.residents[i]
            res.UpdateHospital(res.hos_pro_tmp)  
            res.utility = res.CalUtilityBatch(res.hos_pro)

        # step 3
        for i in range(gp.hos_num):
            hos = gp.hospitals[i]
            hos.UpdateCurrntNum()

        # step 4
        acc = model.CalAccess()
        gp.access.append(acc)

        # step 5
        tmp = model.GetAreaMostlikelyHos()   
        gp.area_mostlikely_hos.append(tmp)
        tmp = model.GetAreaAvgUti()
        gp.area_avg_uti.append(tmp)
        tmp = model.GetHosPop()
        gp.hos_pop.append(tmp)

        for i in range(len(gp.res_record)):
            tmp = model.GetResHosProByID(gp.res_record[i])
            gp.res_hos_pro[i].append(tmp)
    
        if not os.path.exists('results/iteration'):
            os.mkdir('results/iteration')
        pkl_filename = f'results/iteration/iter-{p}.pkl'
        with open(pkl_filename, 'wb') as f:
            pickle.dump([gp.residents, gp.hospitals], f)

    print('The iteration end and the results start to be saved...')

    print('Saving the average utility value of the community...')
    util_list_df = pd.DataFrame(data = gp.area_avg_uti,index = None)
    filename = 'results/avgUility.png'
    utils.DrawAreaAvgUtil(util_list_df,filename)
    print('The average utility value of the community is saved successfully!')

    print('Saving the number of residents selected in all hospitals...')
    hos_pop_df = pd.DataFrame(data = gp.hos_pop,index = None)
    filename = 'results/avgHosPop.png'
    utils.DrawHosPop(gp.hospitals, hos_pop_df,filename)
    filename_csv = 'results/hos_result.csv'
    utils.SaveCSV(hos_pop_df.T, filename_csv)
    print('The number of residents selected in all hospitals is saved successfully!')

    print('Saving policy changes for target residents...')
    for i in range(len(gp.res_hos_pro)):
        d = pd.DataFrame(data=gp.res_hos_pro[i], index=None)
        filename = f'results/res_hos_pro_{gp.res_record[i]}.csv'
        utils.SaveCSV(d, filename)
    print('The target resident\'s strategy change was saved successfully!')

    print('Saving accessibility...')
    acc_list_df = pd.DataFrame(data=gp.access, index=None)
    hos_select_list_df = pd.DataFrame(data=gp.area_mostlikely_hos, index=None)
    demand_ori = pd.read_csv('sample_data/Area.csv', engine='python', encoding='utf-8-sig')
    data = pd.concat([demand_ori, acc_list_df.T, hos_select_list_df.T, util_list_df.T],
                     axis=1)
    utils.SaveResult(data, 'results/result.csv', gp.iteration)
    
    print('Accessibility saved successfully!')

    sys.exit()
