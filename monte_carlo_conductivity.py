import numpy as np
import math
import random
import numba as nb
from matplotlib import pyplot as plt
from scipy import optimize


N=0
dot=np.zeros((10000,2))
a,b=input('AMC-?-?').split()
with open('AMC-%s-%s merged_coor.txt'%(a,b),'r') as file:
    for line in file.readlines():
        a,b=line.split()
        a,b=float(a)*100/1024,float(b)*100/1024
        dot[N][0],dot[N][1]=a,b
        N+=1

def f(x,A,B):
    return A*x+B

@nb.jit(nopython=True)
def dist(i,j):
    return math.sqrt((dot[i][0]-dot[j][0])**2+(dot[i][1]-dot[j][1])**2)

@nb.jit(nopython=True)
def carlo(i): # one step of monte-carlo
    events=np.zeros(N)
    for j in range(N):
        events[j]=math.e**(-1*dist(i,j)) #j:hop from i to j; events[j]: relative probability
    
    # renormalize
    total=sum(events)
    for j in range(N):
        events[j]=events[j]/total
    sample=random.random() # from 0 to 1
    for j in range(N):
        sample=sample-events[j]
        if sample<=0:
            return j
    
    return N-1

@nb.jit(nopython=True)
def hopping(walkers,maxiter,unit_time):
    datas=np.zeros(int(maxiter/unit_time))
    for m in range(walkers):   
        print(m)
        cnt=0
        site=random.randint(0,N-1)
        temp_site=site
        for l in range(maxiter): # maxiter:maxtime
            temp_site=carlo(temp_site)
            if l%unit_time==0:
                temp_dist=dist(temp_site,site)
                datas[cnt]+=temp_dist**2
                cnt+=1

    for j in range(len(datas)):
        datas[j]=datas[j]/walkers

    return datas


unit=1000
maxtime_iter=100000
data=hopping(20,maxtime_iter,unit)

plt.scatter(range(unit,maxtime_iter+unit,unit),data)

A,B=optimize.curve_fit(f,range(unit,maxtime_iter+unit,unit),data)[0]
newy=[f(x,A,B) for x in range(unit,maxtime_iter+unit,int(unit/100))]
# 1s= unit=100
plt.xlabel('t')
plt.ylabel(r'$<r^2>$')
plt.plot(range(unit,maxtime_iter+unit,int(unit/100)),newy,c='red')
print('Conductivity: %.3f'%A)
plt.show()

