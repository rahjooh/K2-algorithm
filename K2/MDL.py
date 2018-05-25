import numpy as np
from decimal import *
from itertools import chain, combinations
import math

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
            #print(str(i)+ '\t  => '+ str(a)+ '\t\t\t => '+ str(Equation(i,a,Dataset1)))
            N =Mat.shape[0]
            B=0
            for r1 in range(Mat.shape[0]):
                for r2 in range(Mat.shape[1]):
                    if Mat[r1][r2] == 1 : B+=1
            val = Equation(i, a, Dataset1) - Decimal(1/2)*Decimal(math.log(N))*B
            if val > MaxVal :
                MaxVal=val
                BestParent = a
        for p in BestParent :
            Mat[i][p] = 1
    return Mat
def CV (K , Dataset):
    np.random.shuffle(Dataset)
    TP = 0;    TN = 0;    FP = 0;    FN = 0
    for i in range(K-1):
        TrainDS = [] ; TestDS =[]
        startrow = (i/K)*len(Dataset) ; endrow = ((i+1)/K)*len(Dataset)
        for i,row in enumerate(Dataset) :
            if startrow <= i and i<endrow:
               TestDS.append(row)
            else:
                TrainDS.append(row)
        TP1 ,TN1,FP1,FN1 = BayesianNet_Test(TrainDS,TestDS,K2(Dataset,False))
        TP += TP1 ; TN+=TN1 ; FP+=FP1 ; FN+=FN1
    TP = TP / K ; TN = TN / K ; FP = FP  / K ; FN = FN / K

    # 0.7574074074074075
    # Precison = 0.8673647469458987
    # Recall = 0.7276720351390922
    # F1
    # SCORE = 0.7914012738853504
    print('==========================================================================================')
    print('================================== Miangin  ==============================================')
    print('==========================================================================================')
    print('   TP = '+ str(TP) +' TN = '+ str(TN) +' FP = '+ str(FP) +' FN = '+ str(FN) )
    print( ' Accuracy =  '+ str((TP+TN)/(TP+TN+FP+FN)))
    print (' Precison = '+ str(TP/(TP+FP)))
    print(' Recall = ' + str(TP / (TP + FN)))
    print(' F1 SCORE =  '+ str((2*TP)/(2*TP+FP+FN)))
def P(str1,DS1):
    if(str1.count('='))==1:
        col = int(str1[:str1.index('=')].replace('X','').replace('x',''))
        val = int(str1[str1.index('=')+1:].replace('.0',''))
        lenght = len(DS1)
        t = []
        for l in DS1:
            if l[col]==val : t.append(l)
        DS1 = t
        #DS1 = DS1[np.array(DS1)[:, col] == val]
        if lenght == 0 :return 1
        return (len(DS1)/lenght)
    else:
        if ',' in str1 : cut = str1.rindex(',')
        else: cut = str1.index('|')
        str2 = str1[cut+1:]
        col = int(str2[:str2.index('=')].replace('X','').replace('x',''))
        val = int(str2[str2.index('=')+1:].replace('.0',''))
        lenght = len(DS1)
        t = []
        for l in DS1:
            if l[col]==val : t.append(l)
        DS1 = t
        #DS1 = DS1[DS1[:, col] == val]
        return (P(str1[:cut],DS1))
def BayesianNet_Test(TrainDs,TestDs,Mat):
    print(Mat)
    TestDs = np.array(TestDs)
    Lable = TestDs[:,0]
    TestDs = np.delete(TestDs , 0, axis=1)
    TP = 0 ; TN = 0 ; FP = 0 ; FN = 0
    for i in range(TestDs.shape[0]) :
        Prob_ = 'X1 = 1 '
        if False:
            for t1 in range(Mat.shape[0]):
                for t2 in range(Mat.shape[1]):
                    if  Mat[t1][t2] == 1 :
                        if '|' not in Prob_ : Prob_ + '| X'+str(t2)+' = '+str(TrainDs[i][t2])
        Prob1 = P('X0=1',TrainDs)* \
                P('X1 = '+str(TestDs[i][1]),TrainDs)* \
                P('X2 = '+str(TestDs[i][2]),TrainDs)* \
                P('X3 = '+str(TestDs[i][3])+'| X1 = '+str(TestDs[i][1])+ ','+ 'X2 = '+str(TestDs[i][2]),TrainDs)* \
                P('X4 = ' + str(TestDs[i][4]) + '| X1 = ' + str(TestDs[i][1]) + ',' + 'X2 = ' + str(TestDs[i][2])+' , X0=1',TrainDs)* \
                P('X5 = ' + str(TestDs[i][5]) + '| X3 = ' + str(TestDs[i][3]) + ',' + 'X4 = ' + str(TestDs[i][4]) + ' , X0=1',TrainDs) * \
                P('X6 = ' + str(TestDs[i][4]) + '| X4 = ' + str(TestDs[i][4]) + ',' + 'X5 = ' + str(TestDs[i][5]) + ' , X0=1',TrainDs)

        Prob2 = P('X0=2',TrainDs) * \
                P('X1 = ' + str(TestDs[i][1]),TrainDs) * \
                P('X2 = ' + str(TestDs[i][2]),TrainDs) * \
                P('X3 = ' + str(TestDs[i][3]) + '| X1 = ' + str(TestDs[i][1]) + ',' + 'X2 = ' + str(TestDs[i][2]),TrainDs) * \
                P('X4 = ' + str(TestDs[i][4]) + '| X1 = ' + str(TestDs[i][1]) + ',' + 'X2 = ' + str(TestDs[i][2]) + ' , X0=2',TrainDs) * \
                P('X5 = ' + str(TestDs[i][5]) + '| X3 = ' + str(TestDs[i][3]) + ',' + 'X4 = ' + str(TestDs[i][4]) + ' , X0=2',TrainDs) * \
                P('X6 = ' + str(TestDs[i][4]) + '| X4 = ' + str(TestDs[i][4]) + ',' + 'X5 = ' + str(TestDs[i][5]) + ' , X0=2',TrainDs)

        pred = 0
        if Prob1 > Prob2 : pred = 1 ;
        else : pred=2
        #print('1#'+str(Prob1)+' 2#'+str(Prob2)+ '  pred#'+str(pred)+ '  lable#'+str(Lable[i]))
        if pred ==1 and Lable[i] == 1 : FP +=1
        if pred == 2 and Lable[i] == 1: TN += 1
        if pred == 1 and Lable[i] == 2: FN += 1
        if pred == 2 and Lable[i] == 2: TP += 1
    print('================================== Scope ==============================================')
    print('   TP = '+ str(TP) +' TN = '+ str(TN) +' FP = '+ str(FP) +' FN = '+ str(FN) )
    print( ' Accuracy =  '+ str((TP+TN)/(TP+TN+FP+FN)))
    print (' Precison = '+ str(TP/(TP+FP)))
    print(' Recall = ' + str(TP / (TP + FN)))
    print(' F1 SCORE =  '+ str((2*TP)/(2*TP+FP+FN)))
    return TP ,TN ,FP , FN




Dataset = read_dataset("data1.txt")
Dataset2 = read_dataset("data2.txt")

a=P('X0 = 2|X1=4',Dataset2)

#a=P('X0 = 2|X1=4',Dataset2)
print(a)
print(len(Dataset2))

CV(5,Dataset2)





