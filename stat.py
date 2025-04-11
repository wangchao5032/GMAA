# %%
import pickle
import numpy as np
# with open('./data/hz/picture/result.pkl', 'rb') as f:
    
def stat(res, hospitals, k):
    lvl_cnts = np.zeros(4)
    for r in res:
        top_hos_ids = np.argsort(r.hos_pro, )[-k:][::-1]
        lvls = np.array([hospitals[hi].level for hi in top_hos_ids])
        lvl_cnt, bins = np.histogram(lvls, bins=[3,4,5,6,7])
        lvl_cnts += lvl_cnt
    lvl_cnts /= len(res)
    return lvl_cnts[::-1]

def stat_one_iteration(ks, filename):
    f = open(filename, 'rb')
    residents, hospitals = pickle.load(f)
    f.close()

    print(filename)
    level_counts = {}
    for k in ks:
        print(f'* top {k}')
        hres = [r for r in residents if r.belief == 100]
        hlvl_cnts = stat(hres, hospitals,  k)
        print(f'tot {len(hres)} high_belief, lv6-lv1: ', ' - '.join([f'{n:.2f}' for n in hlvl_cnts]))
    
        mres = [r for r in residents if r.belief == 1]
        mres_sample_rate = 1
        mres = mres[::mres_sample_rate]
        mlvl_cnts = stat(mres, hospitals, k)
        print(f'tot {len(mres)} med_belief, lv6-lv1: ', ' - '.join([f'{n:.2f}' for n in mlvl_cnts]))
    
        lres = [r for r in residents if r.belief == 0.01]
        llvl_cnts = stat(lres, hospitals, k)
        print(f'tot {len(lres)} low_belief, lv6-lv1: ', ' - '.join([f'{n:.2f}' for n in llvl_cnts]))
        
        print('')
        level_counts[k] = [hlvl_cnts, mlvl_cnts, llvl_cnts]
    return level_counts
        
topks = [20, 10, 5, 3, 1]
dirname = r'D:\code\access\data\实验\实验-增加健康信念异质性\分数为0.01-1-100-随机1.0-探索保留前20-500-统计\results'
tot_level_counts = {}
for i in range(0, 500, 1):
    filename = rf'{dirname}\iter-{i}.pkl'

    lvl_cnts = stat_one_iteration(topks, filename)
        
    tot_level_counts[f'iter-{i}'] = lvl_cnts


with open('final_results.pkl', 'wb') as f:
    pickle.dump(tot_level_counts, f)


# %%
# import matplotlib.pyplot as plt

# top_10_cnts = np.vstack([lc[10][0] for lc in tot_level_counts])
# plt.plot(top_10_cnts, label=['lv6', 'lv5', 'lv4', 'lv3'])

# plt.legend()
# plt.ylim(0, 10)

