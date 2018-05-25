import numpy as np
from decimal import *
from itertools import chain, combinations

def factoriel(n):
    res = 1
    for i in range(n+1):
        if i==0:continue
        res *= i
    return res
def read_dataset(path):
    f1  = open(path,"r")
    list = []
    for l in f1:
        list.append(l.split())
    list= np.array(list)
    list = np.delete(list, 0, axis=0)
    list = np.delete(list, 0, axis=1)

    list= list.astype(float)
    return list
def DifraggerDS (i,pi_vec,Dataset1):
    Maghadir = {} ; temp = [] ; Nij = [] ; t = 0
    Maghadir[i] = np.unique(Dataset1[:, i])
    if len(pi_vec)==0:
        for k in Maghadir[i]:
            temp.append(Dataset1[Dataset1[:,i]==k,:])
            t += len(Dataset1[Dataset1[:,i]==k,:])
        Nij.append(t)
        return temp, Nij
    else:
        Dataset1, Nij = DifraggerDS(pi_vec[0],pi_vec[1:],Dataset1)
        Nij = []
        for ds in Dataset1:
            for k in Maghadir[i]:
                temp.append(ds[ds[:, i] == k, :])
                t += len(ds[ds[:, i] == k, :])
            Nij.append(t) ; t=0
    return temp,Nij
def Equation (i,Parent_vec,Dataset1):
    res = Decimal(1) ; ri =len(np.unique(Dataset1[:, i]))
    Dataset1 , Nij1 = DifraggerDS(i,Parent_vec,Dataset1)
    for n in Nij1:
        res = res* Decimal((factoriel(ri-1))/factoriel(n+ri-1))
    for m in Dataset1:
        if len(m) != 0 : res *= factoriel(len(m))
    return res
def GetAllPosibleParent(iterable):
    xs = list(iterable)
    return list(chain.from_iterable(combinations(xs,n) for n in range(len(xs)+1)))
def K2(Dataset1,flag1):
    Mat = np.zeros([Dataset1.shape[1],Dataset1.shape[1]])
    for i in range(Dataset1.shape[1]):
        if not flag1: allPosParents = GetAllPosibleParent(np.arange(0,i))
        else:allPosParents = GetAllPosibleParent(np.delete(np.arange(0, Dataset1.shape[1]),i))
        MaxVal = -float("inf")
        for a in allPosParents:
            print(str(i)+ '\t  => '+ str(a)+ '\t\t\t => '+ str(Equation(i,a,Dataset1)))
            val = Equation(i,a,Dataset1)
            if val > MaxVal :
                MaxVal=val
                BestParent = a
        for p in BestParent :
            Mat[i][p] = 1
    return Mat
def CV (K , Dataset):
    np.random.shuffle(Dataset)
    for i in range(K-1):
        TrainDS = [] ; TestDS =[]
        startrow = (i/K)*len(Dataset) ; endrow = ((i+1)/K)*len(Dataset)
        for i,row in enumerate(Dataset) :
            if startrow <= i and i<endrow:
               TestDS.append(row)
            else:
                TrainDS.append(row)
        Mat = K2(np.array(TrainDS),False)
        print(Mat)


def BayesianNet_Test(Dataset,Mat):
    P = {}
    for i in range(len(Mat)):
        print()

    print()






Dataset = read_dataset("data1.txt")
Dataset2 = read_dataset("data2.txt")



#m1=K2(Dataset,False)
CV(5,Dataset2)

print('========== Matrice Mojaverat ==============')
#print(m1)



