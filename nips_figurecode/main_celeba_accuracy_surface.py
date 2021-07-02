import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

import numpy as np

f = plt.figure()
ax = f.add_subplot(111, facecolor="whitesmoke")

x1=ax.scatter(50.53,100,s=70,color='saddlebrown',marker='<')#,label='Ideal Trade-Off Solution (HV: 1.00 )')

def median_std(accs):
    median_acc=np.median(accs)
    s=0
    for acc in accs:
        s=s+(acc-median_acc)**2
    s=np.sqrt(s/len(accs))
    return s
## ML ##

# path='./NIPSDATA/Logs_ML_MSE/'
path='./NIPSDATA/Logs_ML_MSE/'
name='ML.csv'

if name=='OptNet.csv':
    idx1=2###########2 0
    idx2=3###########3 1
else:
    idx1=1###########1 0
    idx2=4###########4 3


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]
accs_s = []
accs_t = []
for iter_idx in range(0,5):

    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')
        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append(100-acc_s)################100
        accs_t.append(acc_t)


data1=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data1.copy()
data_n[:,0]=data1[:,0]*-1
data_n[:,1]=data1[:,1]*-1
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points1 = np.array(sorted(data1[front], key=lambda x: (x[1], x[0])))
data_n=data1.copy()

front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points11 = np.array(sorted(data1[front_l], key=lambda x: (x[1], x[0])))



## Extra ML ##

path='./NIPSDATA/Logs_Extra_MSE/'
name='Extra.csv'
if name=='OptNet.csv':
    idx1=2###########2 0
    idx2=3###########3 1
else:
    idx1The=1###########1 0
    idx2=4###########4 3


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]
accs_s = []
accs_t = []
for iter_idx in range(0,5):

    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')
        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append(100-acc_s)############100
        accs_t.append(acc_t)

data2=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data2.copy()
data_n[:,0]=data2[:,0]*-1
data_n[:,1]=data2[:,1]*-1
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points2 = np.array(sorted(data2[front], key=lambda x: (x[1], x[0])))
data_n=data2.copy()

front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points22 = np.array(sorted(data2[front_l], key=lambda x: (x[1], x[0])))

## OptNet ##
path='./NIPSDATA/Logs_OptNet_MSE/'
name='OptNet.csv'

if name=='OptNet.csv':
    idx1=2###########2 0
    idx2=3###########3 1
else:
    idx1The=1###########1 0
    idx2=4###########4 3


lambd = [0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.91, 0.92, 0.93]
accs_s = []
accs_t = []
for iter_idx in range(0,5):

    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')
        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append(100-acc_s) ################ 100
        accs_t.append(acc_t)


data3=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data3.copy()
data_n[:,0]=data3[:,0]*-1
data_n[:,1]=data3[:,1]*-1
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points3 = np.array(sorted(data3[front], key=lambda x: (x[1], x[0])))
data_n=data3.copy()

front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points33 = np.array(sorted(data3[front_l], key=lambda x: (x[1], x[0])))


x2,=ax.plot(points1[:,0],points1[:,1],color='red',label='SGDA-ARL (HV: 0.30 ± 0.31)', alpha=.75)#,marker='o')
ax.plot(points11[:,0],points11[:,1],color='red', alpha=.75)
ax.fill(
    np.append(points1[:,0],points11[:,0][::-1]),np.append(points1[:,1],points11[:,1][::-1]), color='red', alpha=.25)


x3,=ax.plot(points2[:,0],points2[:,1],color='blue',label='ExtraSGDA-ARL (HV: 0.56 ± 0.18)', alpha=.75)#,marker='+')
ax.plot(points22[:,0],points22[:,1],color='blue', alpha=.75)#,marker='+')
ax.fill(
    np.append(points2[:,0],points22[:,0][::-1]),np.append(points2[:,1],points22[:,1][::-1]), color='blue', alpha=.25)



x4,=ax.plot(points3[:,0],points3[:,1],color='green',label='OptNet-ARL (HV: 0.58 ± 0.06)', alpha=.75)#,marker='x')
ax.plot(points33[:,0],points33[:,1],color='green', alpha=.75)#,marker='x')
ax.fill(
    np.append(points3[:,0],points33[:,0][::-1]),np.append(points3[:,1],points33[:,1][::-1]), color='green', alpha=.25)


plt.yticks(fontsize=14)
plt.xticks(fontsize=14)

plt.ylabel('Target Accuracy (%) ',fontsize=15)
plt.xlabel('100% - Adversary Accuracy (%)',fontsize=15)

le=ax.legend([x1,x2,x3,x4],['Ideal Trade-Off Solution (HV: 1.00 )','SGDA-ARL (HV: 0.30 ± 0.31)','ExtraSGDA-ARL (HV: 0.56 ± 0.18)','OptNet-ARL (HV: 0.58 ± 0.06)'],fontsize=12)

le.get_frame().set_edgecolor("black")
plt.grid(linestyle=":")

plt.xlim(xmin=5,xmax=51.5)
plt.ylim(ymin=37,ymax=101.5)

plt.savefig("Accuracy_new.pdf")
plt.savefig("Accuracy_new.png")

plt.show()

