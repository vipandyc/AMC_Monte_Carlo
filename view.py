import numpy as np
import math
import matplotlib.pyplot as plt
import numba as nb
a,b=input('AMC-? -figure?').split()

pos = np.load('AMC-%s/%s/carbon_pos sub.npy'%(a,b))
lattice = np.zeros((pos.shape[0],pos.shape[1]))
lattice[:,0], lattice[:,1] = pos[:,1], pos[:,0]


@nb.jit(nopython=True)
def distance(a,b):
    return math.sqrt((a[0]-b[0])**2+(a[1]-b[1])**2)

@nb.jit(nopython=True)
def cos_angle(atom,atom1,atom2):
    vector1=(atom1[0]-atom[0],atom1[1]-atom[1])
    vector2=(atom2[0]-atom[0],atom2[1]-atom[1])
    product=vector1[0]*vector2[0]+vector1[1]*vector2[1]
    mode=math.sqrt(vector1[0]**2+vector1[1]**2)*math.sqrt(vector2[0]**2+vector2[1]**2)
    return product/mode

#heihei ~ waigualailou~
@nb.jit(nopython=True)
def judgement(mat):
    is_crystal=np.zeros(pos.shape[0])
    # if you use numba, do not change global variables outside the function...

    for i in range(len(mat)):
        neighbours=np.zeros((pos.shape[0],2),dtype=np.int64)
        cnt=0
        for j in range(len(mat)):
            neighbours[cnt][0],neighbours[cnt][1]=j,distance(mat[i],mat[j])
            cnt+=1

        neighbours=sorted(neighbours,key=lambda x:x[1])[1:4]

        if neighbours[2][1]>neighbours[1][1]*1.5: # on the border
            cos=cos_angle(mat[i],mat[neighbours[0][0]],mat[neighbours[1][0]])
            if abs(cos+0.5)<0.15:
                is_crystal[i],is_crystal[neighbours[0][0]],is_crystal[neighbours[1][0]]=1,1,1
            continue

        cos1=cos_angle(mat[i],mat[neighbours[0][0]],mat[neighbours[1][0]])
        cos2=cos_angle(mat[i],mat[neighbours[0][0]],mat[neighbours[2][0]])
        cos3=cos_angle(mat[i],mat[neighbours[1][0]],mat[neighbours[2][0]])
        if abs(cos1+0.5)<0.2 and abs(cos2+0.5)<0.2 and abs(cos3+0.5)<0.2:
            is_crystal[i],is_crystal[neighbours[1][0]]=1,1

    return is_crystal

result=judgement(lattice)
print('Done analyzing, now drawing...')
score=0
with open('AMC-%s-%s crystal_coor.txt'%(a,b),'w') as file:
    for i in range(len(lattice)):
        if result[i]==1: 
            score+=1
            file.writelines('%.3f %.3f\n'%(lattice[i][0],lattice[i][1]))
            plt.scatter(lattice[i][0],lattice[i][1],s=1,c='red')
        else:
            plt.scatter(lattice[i][0],lattice[i][1],s=0.5,c='blue')
            continue

print('score=%d/%d, now saving figure..'%(score,pos.shape[0]))
plt.savefig('AMC-%s-%s merged_plot.jpg'%(a,b))
