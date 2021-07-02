from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.vendor.hv import HyperVolume
import random
import numpy as np


def randomcolor():
    colorArr = ['1','2','3','4','5','6','7','8','9','A','B','C','D','E','F']
    color = ""
    for i in range(6):
        color += colorArr[random.randint(0,14)]
    return "#"+color

def median_std(accs):
    median_acc=np.median(accs)
    s=0
    for acc in accs:
        s=s+(acc-median_acc)**2
    s=np.sqrt(s/len(accs))
    return s



path='./NIPSDATA/Logs_ML_MSE/'
name='ML.csv'


if name=='OptNet.csv' or name=='SARL.csv':
    idx1=2###########2
    idx2=3###########3
else:
    idx1=1###########1
    idx2=4###########4


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]

all_results=[]

#
for iter_idx in range(0,5):
    accs_s = []
    accs_t = []
    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')

        acc_s = float(data[idx1])
        acc_t = float(data[idx2])
        accs_s.append((100-acc_s)*100/50.03)
        accs_t.append(acc_t)

    referencePoint = [0, 0]
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1] * -1


    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1)
    all_results.append(result1)

print('SGDA Median and std: %.4f ± %.4f' %(np.median(all_results), median_std(all_results)))




path='./NIPSDATA/Logs_Extra_MSE/'
name='Extra.csv'

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=2###########2
    idx2=3###########3
else:
    idx1=1###########1
    idx2=4###########4


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]
all_results=[]

for iter_idx in range(0,5):

    accs_s = []
    accs_t = []
    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')

        acc_s = float(data[idx1])
        acc_t = float(data[idx2])
        accs_s.append((100-acc_s)*100/50.03)
        accs_t.append(acc_t)

    referencePoint = [0, 0]
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1] * -1


    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1)
    all_results.append(result1)

print('Extra SGDA Median and std: %.4f ± %.4f' %(np.median(all_results), median_std(all_results)))



path='./NIPSDATA/Logs_OptNet_MSE/'
name='OptNet.csv'

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=2###########2
    idx2=3###########3
else:
    idx1=1###########1
    idx2=4###########4


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]

all_results=[]

for iter_idx in range(0,5):

    accs_s = []
    accs_t = []
    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')

        acc_s = float(data[idx1])
        acc_t = float(data[idx2])
        accs_s.append((100-acc_s)*100/50.03)
        accs_t.append(acc_t)

    referencePoint = [0, 0]
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1] * -1


    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1)
    all_results.append(result1)

print('OptNet Median and std: %.4f ± %.4f' %(np.median(all_results), median_std(all_results)))