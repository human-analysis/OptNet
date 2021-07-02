import matplotlib.pyplot as plt
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
    median_acc=np.mean(accs)
    s=0
    for acc in accs:
        s=s+(acc-median_acc)**2
    s=np.sqrt(s/len(accs))
    return s



path='./NIPSDATA/gaussian_new/Log_SGDA/'
name='ML.csv'

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=0###########2
    idx2=1###########3
else:
    idx1=0###########1
    idx2=2###########4


lambd = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99]

all_results=[]

for iter_idx in range(0,5):
    idxx=iter_idx
    # if iter_idx==1:
    #     continue
    accs_s = []
    accs_t = []
    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')

        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append((acc_s)) ############100
        accs_t.append(acc_t)

    referencePoint = [0, 1]  ###########100 100
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1]
    color = randomcolor()
    plt.plot(data[:, 0], data[:, 1], 'x', color=color)


    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    plt.plot(points2[:, 0], points2[:, 1], 'x-', label='SGDA-ARL iter' + str(idxx),color=color)
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1*4)
    all_results.append(result1*4)

print('SGDA Median and std: %.4f ± %.4f' %(np.mean(all_results), median_std(all_results)))


path='./NIPSDATA/gaussian_new/Log_Extra/'
name='Extra.csv'

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=0###########2
    idx2=1###########3
else:
    idx1=0###########1
    idx2=2###########4


lambd = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99]

all_results=[]

for iter_idx in range(0,6):
    if iter_idx==0:
        idxx=0
    else:
        idxx=iter_idx-1
    if iter_idx==1:
        continue
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
        accs_s.append((acc_s))
        accs_t.append(acc_t)

    referencePoint = [0, 1]
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1]
    color=randomcolor()
    plt.plot(data[:,0],data[:,1],'.',color=color)
    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    plt.plot(points2[:, 0], points2[:, 1], '.-', label='ExtraSGDA-ARL iter' + str(idxx),color=color)
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1*4)
    all_results.append(result1*4)

print('Extra SGDA Median and std: %.4f ± %.4f' %(np.mean(all_results), median_std(all_results)))






path='./NIPSDATA/gaussian_new/Log_OptNet/'
name='OptNet.csv'
plt.title('OptNet')

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=0###########2
    idx2=1###########3
else:
    idx1=0###########1
    idx2=2###########4


lambd =[0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.535, 0.537, 0.538, 0.539]
all_results=[]

#
for iter_idx in range(0,5):
    idxx=iter_idx
    # if iter_idx==1:
    #     continue
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
        accs_s.append((acc_s))
        accs_t.append(acc_t)
    accs_s.append(0.2499)
    accs_t.append(0.1443)



    referencePoint = [0, 1]
    data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
    data_n = data.copy()
    data_n[:, 0] = data[:, 0] * -1
    data_n[:, 1] = data[:, 1]

    color = randomcolor()
    plt.plot(data[:, 0], data[:, 1], '^', color=color)


    hyperVolume = HyperVolume(referencePoint)
    front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

    points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
    plt.plot(points2[:, 0], points2[:, 1], '^-', label='OptNet-ARL iter' + str(idxx),color=color)
    points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
    result1 = hyperVolume.compute(data_n)
    print(result1*4)
    all_results.append(result1*4)

print('OptNet Median and std: %.4f ± %.4f' %(np.mean(all_results), median_std(all_results)))


path='./NIPSDATA/gaussian_new/Log_SARL/'
name='SARL.csv'
plt.title('SARL')

if name=='OptNet.csv' or name=='SARL.csv':
    idx1=0###########2
    idx2=1###########3
else:
    idx1=0###########1
    idx2=2###########4

lambd =[0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.535, 0.537, 0.538, 0.539, 0.55]

all_results=[]
#
#
#
accs_s = []
accs_t = []
for la in lambd:
    # if iter_idx==1:
    #     continue


    path_iter = 'TestLogger_%.6f.txt' % ( la)
    with open(path + path_iter, 'r') as fp:
        lines = fp.readlines()
        lastline = lines[-1]
    data = lastline.split('\t')

    acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
    acc_t = float(data[idx2])
    accs_s.append((acc_s)) ############100
    accs_t.append(acc_t)

accs_s.append(0.25) ############100
accs_t.append(0.1)

referencePoint = [0, 1]  ###########100 100
data=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n = data.copy()
data_n[:, 0] = data[:, 0] * -1
data_n[:, 1] = data[:, 1]
color=randomcolor()
# plt.plot(data[:,0],data[:,1],'x',color=color)
hyperVolume = HyperVolume(referencePoint)
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)

points2 = np.array(sorted(data[front], key=lambda x: (x[1], x[0])))
# plt.plot(points2[:, 0], points2[:, 1], 'x-', label='OptNet-ARL iter' + str(idxx),color=color)
points1 = np.array(sorted(data_n[front], key=lambda x: (x[1], x[0])))
result1 = hyperVolume.compute(data_n)
print(result1*4)
all_results.append(result1*4)

print('SARL Median and std: %.4f ± %.4f' %(np.mean(all_results), median_std(all_results)))
plt.ylim(ymax=1)
plt.legend()
plt.show()