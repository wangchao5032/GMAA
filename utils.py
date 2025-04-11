# -*- coding: utf-8 -*-
import time

import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging
import matplotlib.pyplot as plt
import math
import random

class CoordinateSystemConversion:
    def __init__(self):
        self.x_pi = 3.14159265358979324 * 3000.0 / 180.0
        self.pi = 3.1415926535897932384626  
        self.a = 6378245.0  
        self.ee = 0.00669342162296594323  
        
    def gcj02_to_bd09(self, lng, lat):
        z = math.sqrt(lng * lng + lat * lat) + 0.00002 * math.sin(lat * self.x_pi)
        theta = math.atan2(lat, lng) + 0.000003 * math.cos(lng * self.x_pi)
        bd_lng = z * math.cos(theta) + 0.0065
        bd_lat = z * math.sin(theta) + 0.006
        return [bd_lng, bd_lat]


    def bd09_to_gcj02(self, bd_lon, bd_lat):
        x = bd_lon - 0.0065
        y = bd_lat - 0.006
        z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * self.x_pi)
        theta = math.atan2(y, x) - 0.000003 * math.cos(x * self.x_pi)
        gg_lng = z * math.cos(theta)
        gg_lat = z * math.sin(theta)
        return [gg_lng, gg_lat]


    def wgs84_to_gcj02(self, lng, lat):
        if self.out_of_china(lng, lat):  
            return lng, lat
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [mglng, mglat]


    def gcj02_to_wgs84(self, lng, lat):
        if self.out_of_china(lng, lat):
            return lng, lat
        dlat = self._transformlat(lng - 105.0, lat - 35.0)
        dlng = self._transformlng(lng - 105.0, lat - 35.0)
        radlat = lat / 180.0 * self.pi
        magic = math.sin(radlat)
        magic = 1 - self.ee * magic * magic
        sqrtmagic = math.sqrt(magic)
        dlat = (dlat * 180.0) / ((self.a * (1 - self.ee)) / (magic * sqrtmagic) * self.pi)
        dlng = (dlng * 180.0) / (self.a / sqrtmagic * math.cos(radlat) * self.pi)
        mglat = lat + dlat
        mglng = lng + dlng
        return [lng * 2 - mglng, lat * 2 - mglat]

    def bd09_to_wgs84(self, bd_lon, bd_lat):
        lon, lat = self.bd09_to_gcj02(bd_lon, bd_lat)
        return self.gcj02_to_wgs84(lon, lat)

    def wgs84_to_bd09(self, lon, lat):
        lon, lat = self.wgs84_to_gcj02(lon, lat)
        return self.gcj02_to_bd09(lon, lat)

    def _transformlat(self, lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lat * self.pi) + 40.0 *
                math.sin(lat / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(lat / 12.0 * self.pi) + 320 *
                math.sin(lat * self.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transformlng(self, lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * math.sqrt(math.fabs(lng))
        ret += (20.0 * math.sin(6.0 * lng * self.pi) + 20.0 *
                math.sin(2.0 * lng * self.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(lng * self.pi) + 40.0 *
                math.sin(lng / 3.0 * self.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(lng / 12.0 * self.pi) + 300.0 *
                math.sin(lng / 30.0 * self.pi)) * 2.0 / 3.0
        return ret

    def out_of_china(self, lng, lat):
        return not (lng > 73.66 and lng < 135.05 and lat > 3.86 and lat < 53.55)

class CalDisMatrix:
    def __init__(self, X, Y):
        self.X = X  # demand
        self.Y = Y  # supply
        self.distance_matrix = None 
        
    # 计算距离矩阵
    def CalDistance(self):
        self.distance_matrix = np.reshape(np.sum(self.X**2,axis=1),(self.X.shape[0],1))+ np.sum(self.Y**2,axis=1)-2*self.X.dot(self.Y.T)
        self.distance_matrix = np.sqrt(self.distance_matrix)
        return self.distance_matrix
   
    # 计算经纬度距离矩阵
    def CalDistance_LonLat(self):
        self.distance_matrix = np.zeros([self.X.shape[0], self.Y.shape[0]])
        for i in range(self.X.shape[0]):  # demand
            if (i % 1000 == 0):
                print(f'cal_{i}...')
            for j in range(self.Y.shape[0]):  # supply
                d = self.HaversineDist(self.X[i][0], self.X[i][1], self.Y[j][0], self.Y[j][1])
                self.distance_matrix[i][j] = d
                
        return self.distance_matrix
    
    # 返回距离矩阵的排序结果
    def GetSort(self):
        sort = []
        sort_id = []
        for i in range(self.distance_matrix.shape[0]):
            if (i % 1000 == 0):
                print(f'sort_{i}...')
            d = self.distance_matrix[i]
            sort.append(np.sort(d))           
            sort_id.append(np.argsort(d))
        return sort, sort_id
            
    
    # Haversine公式
    def HaversineDist(self, x1, y1, x2, y2):  
        # 将坐标转换为弧度
        lat1 = math.radians(x1)
        lon1 = math.radians(y1)
        lat2 = math.radians(x2)
        lon2 = math.radians(y2)
    
        # 计算纬度和经度的差异
        dlat = lat2 - lat1
        dlon = lon2 - lon1
    
        # 应用Haversine公式
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
        # 计算两个坐标间的距离
        distance = 6371 * c
    
        return distance

def gaussian_decay(distance_array, sigma):
    return np.exp(-(distance_array**2 / (2.0 * sigma**2)))

# 通过百度地图计算距离矩阵
class CalDisMatrix_Baidu:
    def __init__(self, X, Y):
        self.X = X  # demand
        self.Y = Y  # demand
        self.distance_matrix = None
        self.ak = ''
        self.url = "https://api.map.baidu.com/routematrix/v2/driving"
        
    def CalDistance(self):
        origins =  '|'.join([f"{lat},{lon}" for lat, lon in self.X])
        destinations = '|'.join([f"{lat},{lon}" for lat, lon in self.Y])
        tactics = 11

        params = {
            'ak': self.ak,
            'origins': origins,
            'destinations': destinations,
            'tactics': tactics
        }

        source_num = len(self.X)
        target_num = len(self.Y)

        # 调用百度api，最多尝试10次
        for i in range(10):
            res = requests.get(url=self.url, params=params)
            json_res = json.loads(res.text)
            if json_res.get('status', -1) == 0: break
            print(f'request.get {i} failed, msg=[{json_res["message"]}]sleep and try again')
            time.sleep(1.5)     
            if i == 9: 
                return None, None
        distance_values = [item['distance']['value'] for item in json_res['result']]
        duration_values = [item['duration']['value'] for item in json_res['result']]
        # print(distance_values)
        # print(duration_values)

        np_distance = np.array(distance_values).reshape(source_num,target_num)
        np_duration = np.array(duration_values).reshape(source_num,target_num)
        # print(np_distance)
        # print(np_duration)
        self.distance_matrix = np_distance
        return np_distance, np_duration

# 返回距离矩阵的排序结果
def GetSort(distance_matrix):
    sort = []
    sort_id = []
    for i in range(distance_matrix.shape[0]):
        if (i % 1000 == 0):
            print(f'sort_{i}...')
        d = distance_matrix[i]
        sort.append(np.sort(d))
        sort_id.append(np.argsort(d))
    return sort, sort_id
    
#%% 插值  
# 克里金插值
class SpatialInterpolation:
    def __init__(self,x, y, z):
        self.x = x  
        self.y = y  
        self.z = z  
        minx, maxx = np.min(self.x), np.max(self.x)
        miny, maxy = np.min(self.y), np.max(self.y)
        self.xx = np.linspace(minx, maxx, 100)  
        self.yy = np.linspace(miny, maxy, 100)
        
    def Krig(self):
        OK = OrdinaryKriging(
            np.array(self.x),
            np.array(self.y),
            self.z,
            variogram_model = "linear",
            verbose = False,
            enable_plotting = False,
            coordinates_type = "euclidean",
        )

        acc_krig, sigma = OK.execute("grid", self.xx, self.yy)

        return acc_krig

def CalGini(x, w=None):
    # The rest of the code requires numpy arrays.
    x = np.asarray(x)
    if w is not None:
        w = np.asarray(w)
        sorted_indices = np.argsort(x)
        sorted_x = x[sorted_indices]
        sorted_w = w[sorted_indices]
        # Force float dtype to avoid overflows
        cumw = np.cumsum(sorted_w, dtype=float)
        cumxw = np.cumsum(sorted_x * sorted_w, dtype=float)
        return (np.sum(cumxw[1:] * cumw[:-1] - cumxw[:-1] * cumw[1:]) /
                (cumxw[-1] * cumw[-1]))
    else:
        sorted_x = np.sort(x)
        n = len(x)
        cumx = np.cumsum(sorted_x, dtype=float)
        # The above formula, with all weights equal to 1 simplifies to:
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n

def Cosine_similarity(A, B):
    dot = np.dot(A, B.T)
    norm_A = np.linalg.norm(A, axis=1).reshape(-1,1)
    norm_B = np.linalg.norm(B, axis=1).reshape(1,-1)
    similarity = dot / (norm_A * norm_B)
    return similarity

def Corrcoef(A,B):
    return np.corrcoef(A.flatten(), B.flatten())[0,1]

#%% 绘图
def DrawAccess(index, supply_spatial, demand_spatial, areas, residents, hospitals, acc, filename):  # 绘制可达性结果
    # 创建一个图像    
    plt.figure(index, figsize=(20, 20))
    # 设置颜色表
    color_table = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'magenta']
    color_num = len(color_table)
    # 绘制医院的位置为五角星
    plt.plot(supply_spatial.to_numpy()[:, 0], supply_spatial.to_numpy()[:, 1], 'b*', ms=40)
    # 绘制居民
    hos_num = len(hospitals)  
    for i in range(hos_num):
        hos = hospitals[i]
        # 寻找最喜欢的医院是i的居民
        res_poss = np.array([[areas[res.area_ID].x, areas[res.area_ID].y] for res in residents if res.hos_pro_max == i])  
        if res_poss.shape[0] == 0:  # 如果没有居民最想去的医院是i
            print(f' - No res choose hos {i}, skip')
            continue
        # 如果有居民最想去的医院是i，则绘制居民所在区域的位置
        plt.plot(res_poss[:, 0], res_poss[:, 1], '.', c=color_table[i % color_num], alpha=0.7,markersize='20', label=f'hos{i}') 
        # 标记医院的级别
        plt.text(hos.x, hos.y, f'H-{i}\nlv={hos.level:.0f}', fontsize=20)
 
    # 克里金插值
    demand_x_list, demand_y_list = list(demand_spatial['经度']),  list(demand_spatial['纬度'])
    si = SpatialInterpolation(demand_x_list, demand_y_list, acc)
    acc_krig = si.Krig()
    plt.pcolormesh(si.xx, si.yy, acc_krig)  # 使用非规则矩形网格创建伪彩色绘图

    # 保存图像
    plt.title(f'iter {index}',fontsize=30)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.axis('equal')
    plt.grid()
    plt.legend(fontsize=20)
    plt.savefig(filename)
    return

def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color ="#"+''.join([random.choice(colorArr) for i in range(6)])
    return color

def DrawAreaAvgUtil(util_list_df, filename):
    util_list_df = util_list_df[1:util_list_df.shape[0]] # 删除第一行，即第一次迭代
    data = util_list_df.T
    plt.figure(figsize=(20, 20))
    for i in range(len(data)):
        if (i % 300 == 0):
            plt.plot(util_list_df[i],'o:', c = randomcolor(), label=f'area{i}')
    
    # plt.legend(fontsize=20)
    plt.title('Average utility value of the community',fontsize=20)
    plt.legend(fontsize=10)
    plt.savefig(filename)
    return 

def DrawHosPop(hospitals, hos_pop_df, filename):  # 绘制所有医院被选择的居民数变化
    hos_pop_df = hos_pop_df[1:hos_pop_df.shape[0]] # 删除第一行，即第一次迭代
    data = hos_pop_df.T
    plt.figure(figsize=(20, 20))
    # for i in range(len(data)):
    #     plt.plot(hos_pop_df[i],'ob:', c = randomcolor(), label=f'hos{i}-level:{hospitals[i].level}')
        
    # color_table = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'magenta']
    
    for i in range(len(data)):
        if (i % 5 == 0):
            # 先绘制原始数据
            plt.plot(hos_pop_df[i],'o:', c = randomcolor(), label=f'hos{i}-level:{hospitals[i].level}', linewidth=3)
            # plt.plot(hos_pop_df[i],'ob:', c = color_table[i], label=f'hos{i}-level:{hospitals[i].level}', linewidth=3)
            # 绘制拟合的直线
            # y = np.array(hos_pop_df[i])
            # x = np.array(list(range(0,len(y))))  # 新建一个0-11的序列
            # p = np.polyfit(x, y, 1)  # 最小二乘法拟合直线
            # y_fit = np.polyval(p, x)
            # plt.plot(x, y_fit, color=color_table[i])

    plt.title('Number of residents selected by hospital',fontsize=20)
    # plt.title('Number of residents selected by hospital(excluding iteration 0)',fontsize=20)
    plt.legend(fontsize=10, loc="upper right")
    plt.savefig(filename)
    return 

def DrawResUtils(res_uti_df, filename):  # 绘制所有居民的效用
    res_uti_df = res_uti_df[1:res_uti_df.shape[0]] # 删除第一行，即第一次迭代
    data = res_uti_df.T
    plt.figure(figsize=(20, 20))
    for i in range(len(data)):
        if (i % 2500 == 0):
            plt.plot(res_uti_df[i],'o:', c = randomcolor(), label=f'res{i}')
    
    plt.title('Utility of all residents',fontsize=20)
    plt.legend(fontsize=10)
    plt.savefig(filename)
    return


def DrawResHosPo(hospitals, index, res_hos_pro, filename):  # 绘制第index个居民的策略（选择各个医院的概率）变化
    data = []  # 保存第index个居民在所有迭代的策略
    for i in range(len(res_hos_pro)):  # 遍历每一次迭代
        p = res_hos_pro[i][index]  # 第index个居民在第i轮迭代的策略
        data.append(p)
    data_df = pd.DataFrame(data = data,index = None)
    
    # 绘图
    plt.figure(figsize=(20, 20))
    color_table = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'magenta']
    for i in range(len(data[0])):  # 绘制第index个居民选择第j个医院的策略变化
        plt.plot(data_df[i],'ob:', c = color_table[i], label=f'hos{i}-level:{hospitals[i].level}')
        
    plt.title(f'The strategy change of the the {index}-th resident',fontsize=20)
    plt.legend(fontsize=10, loc="upper right")
    plt.savefig(filename)
    return


#%% 保存结果
def SaveResult(data, name, iteration):  
    bias = data.columns.values.shape[0] - iteration * 3
    for i in range(iteration):
        data.columns.values[bias+i] = f'acc_{i}'  
        data.columns.values[bias+i+iteration] = f'h_{i}'  
        data.columns.values[bias+i+iteration*2] = f'u_{i}'  
    # 保存为csv格式
    data.to_csv(name ,encoding='utf_8_sig', index=False)
    return

def SaveCSV(data, name):  
    # 保存为csv格式
    data.to_csv(name ,encoding='utf_8_sig', index=False)
    return

def SaveResult2SFCA(data, name):  
    data.columns.values[7] = 'acc'  
    data.to_csv(name ,encoding='utf_8_sig', index=False)
    return
    


















