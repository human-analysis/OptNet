
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
import numpy as np

f = plt.figure()
ax = f.add_subplot(111, facecolor="whitesmoke")

x1=ax.scatter(0.25,0,s=70,color='saddlebrown',marker='<')#,label='Ideal Trade-Off Solution (HV: 1.00 )')

def median_std(accs):
    median_acc=np.median(accs)
    s=0
    for acc in accs:
        s=s+(acc-median_acc)**2
    s=np.sqrt(s/len(accs))
    return s

path='./NIPSDATA/gaussian_new/Log_SGDA/'
name='ML.csv'

if name=='OptNet.csv':
    idx1=0##########2 0
    idx2=1###########3 1
else:
    idx1=0###########1 0
    idx2=2###########4 3


lambd =[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99]
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
        accs_s.append(acc_s)################100
        accs_t.append(acc_t)


data1=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data1.copy()
data_n[:,0]=data1[:,0]*-1
data_n[:,1]=data1[:,1]
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points1 = np.array(sorted(data1[front], key=lambda x: (x[1], x[0])))
data_n=data1.copy()
data_n[:,0]=data1[:,0]
data_n[:,1]=data1[:,1]*-1

front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points11 = np.array(sorted(data1[front_l], key=lambda x: (x[1], x[0])))



path='./NIPSDATA/gaussian_new/Log_Extra/'
name='Extra.csv'
if name=='OptNet.csv':
    idx1=0###########2 0
    idx2=1###########3 1
else:
    idx1=0###########1 0
    idx2=2###########4 3


lambd =[0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 0.99]
accs_s = []
accs_t = []
for iter_idx in range(0,6):
    if iter_idx==1:
        continue

    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')
        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append(acc_s)############100
        accs_t.append(acc_t)

data2=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data2.copy()
data_n[:,0]=data2[:,0]*-1
data_n[:,1]=data2[:,1]
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points2 = np.array(sorted(data2[front], key=lambda x: (x[1], x[0])))
data_n=data2.copy()
data_n[:,0]=data2[:,0]
data_n[:,1]=data2[:,1]*-1
front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points22 = np.array(sorted(data2[front_l], key=lambda x: (x[1], x[0])))


path='./NIPSDATA/gaussian_new/Log_OptNet/'
name='OptNet.csv'

if name=='OptNet.csv':
    idx1=0###########2 0
    idx2=1###########3 1
else:
    idx1=0###########1 0
    idx2=2###########4 3


lambd =[0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.535, 0.537, 0.538, 0.539]
accs_s = []
accs_t = []
for iter_idx in range(0,5):
    # if iter_idx==1:
    #     continue
    for la in lambd:
        path_iter = 'TestLogger_%d_%.4f.txt' % (iter_idx, la)
        with open(path + path_iter, 'r') as fp:
            lines = fp.readlines()
            lastline = lines[-1]
        data = lastline.split('\t')
        acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
        acc_t = float(data[idx2])
        accs_s.append(acc_s) ################ 100
        accs_t.append(acc_t)


data3=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data3.copy()
data_n[:,0]=data3[:,0]*-1
data_n[:,1]=data3[:,1]
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points3 = np.array(sorted(data3[front], key=lambda x: (x[1], x[0])))
data_n=data3.copy()
data_n[:,0]=data3[:,0]
data_n[:,1]=data3[:,1]*-1
front_l=NonDominatedSorting().do(data_n, only_non_dominated_front=True)
points33 = np.array(sorted(data3[front_l], key=lambda x: (x[1], x[0])))


########### SARL ###############
#
path='./NIPSDATA/gaussian_new/Log_SARL/'
name='SARL.csv'
if name=='OptNet.csv' or name=='SARL.csv':
    idx1=0###########2 0
    idx2=1###########3 1
else:
    idx1The=0###########1 0
    idx2=3###########4 3


lambd = [0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.51, 0.52, 0.53, 0.535, 0.537, 0.538, 0.539, 0.55]
accs_s = []
accs_t = []


for la in lambd:
    path_iter = 'TestLogger_%.6f.txt' % ( la)
    with open(path + path_iter, 'r') as fp:
        lines = fp.readlines()
        lastline = lines[-1]
    data = lastline.split('\t')
    acc_s = float(data[idx1])  ## oneshot 2 3   ML 1 4
    acc_t = float(data[idx2])
    accs_s.append(acc_s) ################ 100
    accs_t.append(acc_t)

accs_s.append(0.25)
accs_t.append(0.1)
data4=np.hstack((np.array(accs_s)[:,np.newaxis],np.array(accs_t)[:,np.newaxis]))
data_n=data4.copy()
data_n[:,0]=data4[:,0]*-1
data_n[:,1]=data4[:,1]
front = NonDominatedSorting().do(data_n, only_non_dominated_front=True)
data4 = np.array(sorted(data4[front], key=lambda x: (x[1], x[0])))




x2,=ax.plot(points1[:,0],points1[:,1],color='red', alpha=.75)
ax.plot(points11[:,0],points11[:,1],color='red', alpha=.75)#,marker='o')
ax.fill(
    np.append(points1[:,0],points11[:,0][::-1]),np.append(points1[:,1],points11[:,1][::-1]), color='red', alpha=.25)

# ax.plot(points2[:,0],points2[:,1],color='blue',label='ExtraSGDA-ARL (0.1956 ± 0.0525)', alpha=.75,marker='x')
x3,=ax.plot(points2[:,0],points2[:,1],color='blue', alpha=.75)#,marker='+')
ax.plot(points22[:,0],points22[:,1],color='blue', alpha=.75)#,marker='+')
ax.fill(
    np.append(points2[:,0],points22[:,0][::-1]),np.append(points2[:,1],points22[:,1][::-1]), color='blue', alpha=.25)

x5,=ax.plot(data4[:,0],data4[:,1],color='black',linewidth=3,linestyle=':',label='SARL', alpha=.75)

x4,=ax.plot(points3[:,0],points3[:,1],color='green', alpha=.75)#,marker='x')
ax.plot(points33[:,0],points33[:,1],color='green', alpha=.75)#,marker='x')
ax.fill(
    np.append(points3[:,0],points33[:,0][::-1]),np.append(points3[:,1],points33[:,1][::-1]), color='green', alpha=.25)




plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.ylabel('Target MSE',fontsize=15)
plt.xlabel('Adversary MSE',fontsize=15)

le=ax.legend([x1,x2,x3,x5,x4],['Ideal Trade-Off Solution (HV: 1.000 )','SGDA-ARL (HV: 0.957 ± 0.010)','ExtraSGDA-ARL (HV: 0.955 ± 0.009)','SARL (HV: 0.988 )','OptNet-ARL (HV: 0.966 ± 0.002)'],fontsize=12)

le.get_frame().set_edgecolor("black")
plt.grid(linestyle=":")

plt.xlim(xmax=0.26)
plt.ylim(ymax=0.45)

plt.savefig("MSE_gaussian.pdf")
plt.savefig("MSE_gaussian.png")

plt.show()

