# -*- coding: utf-8 -*-
import numpy as np
from sklearn import preprocessing
import random
import utils
import math


class GlobalParams(object):
    # 一些全局变量
    Hospital_Level = {'Grade1B': 1, 'Grade1A': 2, 'Grade2B':3, 'Grade2A':4, 'Grade3B':5, 'Grade3A':6}
    Hospital_Cap =  {1:1500, 2:1600, 3:1700, 4:1800, 5:1900, 6:2000}
    Area_Pop = {'Shangcheng':13, 'Gongshu':12, 'Xiaoshan':23, 'Xihu':15, 'Yuhang':22, 'Linpin':20, 'Linan':16, 'Fuyang':22, 'Tonglu':14, 'Qiantang':26, 'Binjiang':19, 'Jiande':21, 'Chunan':23}
    
    belief_score =[0.01, 1, 100]
    belief_pop = [0.03, 0.956, 0.014]
    select_areas = [0]
    good_hospital_ids = []
    
    area_dist_sort_id = []  
    area_dist_sort = [] 

    distance_matrix = []
    distance_matrix_id = []  
    k = 50
    
    areas = []  
    residents = []  
    hospitals = []  
    
    area_num = -1 
    hos_num = -1  
    res_num = -1  
    
    area_mostlikely_hos = []  
    area_avg_uti = []  
    hos_pop = []  
    res_uti = []  
    
    policy_dimension = 20 
    policy_cons_num = 5  

    access = []  
    thre_dist = 15  
    thre_dist_nei = 5  
   
    iteration = 500  
    iteration_explore = 2  
    
    # 默认是5,1,5
    w_quality  = 5  
    w_dist = 1  
    w_cong = 5  
    
    p_explore_init = 0.9  
    decay_rate = 0.01  

    res_hos_pro = []  
    res_record = []

    def InitRecordRes(self, res_IDs):
        self.res_record = res_IDs
        for i in range(len(self.res_record)):
            self.res_hos_pro.append([])
        print(f"record res_IDs {res_IDs}")
        
gp = GlobalParams()

#%% 医院智能体
class Hospital:  
    def __init__(self, ID, x, y, level):
        self.ID  = ID  
        self.x = x  
        self.y = y  
        self.level = gp.Hospital_Level[level]  
        
        self.quality = self.level  
        self.cap = gp.Hospital_Cap[self.level]
        self.current_num = 0 
 
    def UpdateCurrntNum(self): 
        self.current_num = sum([res.hos_pro[self.ID] for res in gp.residents])
        return self.current_num

    
 #%% 居民智能体       
class Resident: 
    def __init__(self,ID, area_ID, belief):
        self.ID = ID  
        self.area_ID = area_ID  
        self.belief = belief  
        self.hos_pro = []  
        self.hos_pro_tmp = []  
        self.hos_pro_max = -1  
        self.utility = 0  
        
    def UpdateHospital(self, hos_select):  
        self.hos_pro = hos_select
        max_hos = np.argmax(hos_select) 
        self.hos_pro_max = max_hos
        return 

    def UpdateHospitalTemp(self, hos_select):  
        self.hos_pro_tmp = hos_select
        return

    def CalUtilityBatch(self, hos_pro):  
        hos_attrs = np.array([[hos.quality, hos.cap, hos.current_num] for hos in gp.hospitals])
        dists = gp.distance_matrix[self.area_ID]
        sc1 = gp.w_quality * (self.belief * hos_attrs[:, 0])
        sc2 = -1 * gp.w_dist * dists
        sc3 = -1 * gp.w_cong * hos_attrs[:, 2] / hos_attrs[:, 1]
        utility = np.sum(np.array(hos_pro) * (sc1 + sc2 + sc3))
        return utility

    def FindHospital(self, q):  
        if q < gp.iteration_explore:  
            self.FindHospital_Explore(rand=True)
            return
        # 计算本轮的随机概率
        p = gp.p_explore_init * math.pow(1-gp.decay_rate, q//2)
        # 默认是p = gp.p_explore_init * math.pow(1-gp.decay_rate, q)
        # p的计算参考论文《REWARD UNCERTAINTY FOR EXPLORATION IN PREFERENCE-BASED REINFORCEMENT LEARNING》公式5
        # 生成随机数，判断是否需要探索
        rand_num = random.random()  
        if rand_num <= p:  
            self.FindHospital_Explore(rand=False)
            return
        else:  # 和邻居学习
            # 1.随机选择一个邻居
            area = gp.areas[self.area_ID]
            area_neis = area.neighbours  
            area_nei = random.sample(area_neis, 1)[0]  
            area_nei_res = gp.areas[area_nei].res_IDs
            res_neighbor = gp.residents[random.sample(area_nei_res, 1)[0]]  
            # 2.以概率prob_learn和邻居学习
            if self.belief!=res_neighbor.belief: 
                # 计算邻居和自己的效用差方式1:将邻居的策略应用在自己的健康信念上，重新计算
                new_utility = self.CalUtilityBatch(res_neighbor.hos_pro)
            else: 
                # 计算邻居和自己的效用差方式2:直接比较二者效用
                new_utility = res_neighbor.utility
            diff = (self.utility - new_utility) / 0.1
            diff = 709 if diff > 709 else diff # 为了避免exp函数报错（Math Range Error），设置上限和下限的值
            prob_learn = 1 / (1 + math.exp(diff))  
            # prob_learn的计算参考论文《Strategic use of payoff information in k-hop evolutionary Best-shot networked public goods game》的公式5\
            rand_num = random.random()  
            if rand_num <= prob_learn:  
                self.FindHospital_Learn(res_neighbor)
        return
           
    def FindHospital_Learn(self, neighbor):  
        # 临时记录选择的医院
        self.UpdateHospitalTemp(neighbor.hos_pro)
        return
    
    def FindHospital_Explore_Score(self):  
        scores = []  
        # 1.计算每一个医院对居民的吸引力分数
        for i in range(gp.hos_num): 
            area_ID = self.area_ID  
            dist = gp.distance_matrix[area_ID][i]  
            hos = gp.hospitals[i]
            score = CalScore2Resident(hos, self, dist)
            scores.append(score)
        
        # 2.计算选择所有医院的概率
        # 先把分数缩放到0和1之间
        tmp = np.array(scores).reshape(-1, 1)
        scaler = preprocessing.MinMaxScaler()
        tmp_norm = scaler.fit_transform(tmp).reshape(-1).tolist()
        # 让分数之和为1
        tmp_norm_sum = np.sum(tmp_norm)
        tmp_norm = tmp_norm / tmp_norm_sum
        # 3.临时记录选择的医院
        self.UpdateHospitalTemp(tmp_norm)
        return

    def FindHospital_Explore(self, rand):  
        # 1.随机生成前往各个医院的概率
        if rand==True:  
            prob = np.random.rand(gp.hos_num)
            prob_c = prob / np.sum(prob)  
        else:
            area = gp.areas[self.area_ID]
            if self.ID - area.res_IDs[0] < gp.policy_cons_num: 
                random_idx = np.where(area.policy_constraint == 1)
            elif self.ID - area.res_IDs[0] < gp.policy_cons_num * 2: 
                random_idx = np.random.choice(gp.good_hospital_ids, size=gp.policy_dimension, replace=False)
            else: 
                random_idx = np.random.choice(gp.hos_num, size=gp.policy_dimension, replace=False)
            prob_c = np.zeros(gp.hos_num)
            prob_c[random_idx] = np.random.rand(gp.policy_dimension)
            prob_c = prob_c / np.sum(prob_c)
            prob_c = self.hos_pro * 0.8 + prob_c * 0.2
        # 3.更新选择的医院
        self.UpdateHospitalTemp(prob_c)
        return

#%% 地区智能体       
class Area: # 区域智能体
    def __init__(self, ID, x, y, district):  
       self.ID = ID  
       self.x = x  
       self.y = y  
       self.district = district  
       
       self.pop = gp.Area_Pop[self.district]  
       self.res_IDs = []  
       self.avg_uti = 0  
       self.neighbours = None
       
       # 初始化居民个体，健康信念是gp.belief_score的中间值
       for j in range(self.pop):
           gp.res_num = gp.res_num + 1  
           self.res_IDs.append(gp.res_num)
           res = Resident(gp.res_num, self.ID, gp.belief_score[1])
           gp.residents.append(res)
              
       # 初始化小区中居民的概率将来乘以的策略约束
       self.policy_constraint = self.GetPolicyConstraint()  
        
       
    def GetPolicyConstraint(self):
        distid = gp.distance_matrix_id[self.ID]  
        distid_constraint = distid[:gp.policy_dimension]  
        tmp = np.array([0] * gp.hos_num)  
        for c in distid_constraint:  
            tmp[c] = 1
        return tmp

    # 寻找最近的小区里的居民作为邻居
    def FindNeighbors(self):
        neis = []
        distid = gp.area_dist_sort_id[self.ID]  
        distid_k = distid[:gp.k]  
        for i in range(len(distid_k)):
           neis.append(gp.areas[distid_k[i]].res_IDs)
        neis = list(np.concatenate(neis))  
        # 赋值给自己小区内的居民
        for r in self.res_IDs:
            tmp = neis.copy()
            tmp.remove(r)  
            gp.residents[r].neighbors = tmp

    # 寻找最近的小区作为邻居
    def FindNeighborsArea(self):
        distid = gp.area_dist_sort_id[self.ID]   
        dists = gp.area_dist_sort[self.ID]  
        dists_k = dists[:gp.k]   
        valid_dists = (dists_k < gp.thre_dist_nei)   
        new_k = np.sum(valid_dists)  
        if (new_k==0):
            new_k = 5
        self.neighbours = list(distid[:new_k])

    def GetMostLikelyHos(self):  
        choose_hos = np.zeros(len(gp.hospitals)) 
        for res_ID in self.res_IDs:  
            choose_hos += gp.residents[res_ID].hos_pro
        max_pro, max_hos = np.max(choose_hos), np.argmax(choose_hos)  
        return max_hos
    
    def CalAvgUtility(self):  
        result = 0 
        for res_ID in self.res_IDs:
            result += gp.residents[res_ID].utility
        self.avg_uti = result / len(self.res_IDs)
        return self.avg_uti
        
#%% 一些涉及多个智能体通用方法（部分类似的等结束后可以重构下）
def SetBelief(select_areas):
    initialized_low, initialized_medium, initialized_high = SetPOIBelief(select_areas)
    SetGlobalBelief(initialized_low, initialized_medium, initialized_high)
    return initialized_low + initialized_medium + initialized_high

def SetGlobalBelief(initialized_low, initialized_medium, initialized_high):
    # 计算需要赋值的居民数
    belief_pop_num = gp.res_num * np.array(gp.belief_pop)  
    belief_pop_num = list(map(round,belief_pop_num))  
    belief_pop_num[0] -= len(initialized_low)
    belief_pop_num[2] -= len(initialized_high)
    belief_pop_num[1] = sum(belief_pop_num) - belief_pop_num[0] - belief_pop_num[2]
    
    # 初始化健康信念列表
    b_list = [gp.belief_score[1]] * gp.res_num  
    for lr in initialized_low:
        b_list[lr] = gp.belief_score[0]
    for hr in initialized_high:
        b_list[hr] = gp.belief_score[2]
        
    # 随机选择未初始化的居民，设置健康信念
    initialized_res = initialized_high + initialized_medium + initialized_low
    sample = [r for r in range(gp.res_num) if r not in initialized_res]  
    b_list_low = random.sample(sample, belief_pop_num[0])  
    sample_new = [i for i in sample if i not in b_list_low]  
    b_list_high = random.sample(sample_new, belief_pop_num[2])
    for v in b_list_low:
        b_list[v] = gp.belief_score[0]
    for v in b_list_high:
        b_list[v] = gp.belief_score[2]
    # 为居民的健康信念赋值
    for index, element in enumerate(b_list):
        if element != gp.belief_score[1]:
            gp.residents[index].belief = element
    # print(f"random set low:{b_list_low}")
    # print(f"random set high:{b_list_high}")

def SetPOIBelief(select_areas):
    initialized_low = []
    initialized_medium = []
    initialized_high = []
    # 遍历selectAreas，给小区中的前两个居民分别设置为高、低信念
    for a in select_areas:
        res_ids = gp.areas[a].res_IDs  
        gp.residents[res_ids[0]].belief = gp.belief_score[2]  
        gp.residents[res_ids[1]].belief = gp.belief_score[1]  
        gp.residents[res_ids[2]].belief = gp.belief_score[0]  
        initialized_high.append(res_ids[0])  
        initialized_medium.append(res_ids[1])  
        initialized_medium.append(res_ids[3])  
        initialized_low.append(res_ids[2])  
    return initialized_low, initialized_medium, initialized_high
    
def GetAreaMostlikelyHos():  
    return [a.GetMostLikelyHos() for a in gp.areas]

def GetAreaAvgUti():  
    return [a.CalAvgUtility() for a in gp.areas]

def GetHosPop():  
    return [h.current_num for h in gp.hospitals]
 
def GetResUtil():  
    return [res.utility for res in gp.residents]

def GetResHosPro():  
    return [res.hos_pro for res in gp.residents]

def GetResHosProByID(id):  
    return (gp.residents[id]).hos_pro

def CalScore2Resident(hos, res, dist):  
    sc1 = (hos.quality**2) * res.belief
    sc2 = dist
    sc3 = hos.current_num / hos.cap
    score = gp.w_quality * sc1 - gp.w_dist * sc2 - gp.w_cong * sc3
    return score

def CalAccess():
    access_list = []  
    supply_demand_ratio_list = [0] * gp.hos_num  
    # 计算每一个医院的供需比
    for j in range(gp.hos_num): 
        hos = gp.hospitals[j]
        supply = hos.cap
        demand = hos.current_num 
        # 计算供需比
        supply_demand_ratio = supply/(demand+1e-6)  
        supply_demand_ratio_list[j] = supply_demand_ratio
    
    # 计算每一个区域的可达性
    for i in range(gp.area_num): 
        dist_list = gp.distance_matrix[i] 
        scores = utils.gaussian_decay(np.array(dist_list), gp.thre_dist)
        result = np.sum(scores * supply_demand_ratio_list)
        access_list.append(result)
    return access_list

def Cal2SFCA(thre_dist):  
    access_list = []  
    supply_demand_ratio_list = []  
    # 计算每一个医院的供需比
    for j in range(gp.hos_num): 
        hos = gp.hospitals[j]
        supply = hos.cap
        dist_j_list = gp.distance_matrix[:,j]  
        demand = np.sum(dist_j_list <= thre_dist)  
        demand = demand * gp.pop  
        # 计算供需比
        supply_demand_ratio = supply / (demand + 1e-6)
        supply_demand_ratio_list.append(supply_demand_ratio)
    
    # 计算每一个区域的可达性
    for i in range(gp.area_num): 
        result = 0
        dist_list = gp.distance_matrix[i] 
        for j in range(gp.hos_num):
            hos = gp.hospitals[j]
            dist_i_j = dist_list[j] 
            if dist_i_j <= thre_dist:
                result += supply_demand_ratio_list[j]  
        access_list.append(result)

    return access_list