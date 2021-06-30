# def add_device(device):
#     import pandas as pd
#     import numpy as np
    
#     #calling .csv file
#     df=pd.read_csv('Fake_weibul.csv')   #load .csv file
#     x=df['Fluence']           #x=column A
#     Fluence=np.array(x)
#     np.max(x)
    
#     x0=df['LET']        #x=column b
#     LET=np.array(x0)
    
#     x00=df['XS']         #x00=column C, label times
#     x00
#     XS=np.array(x00)
#     np.max(XS)
#     XS

def LV2PY (LET) :

    #import matplotlib.pyplot as plt
    #from matplotlib.backends.backend_agg import FigureCanvasAgg
    #from matplotlib.figure import Figure
    #import cvlib as cv
    #import sys
    #import cv2
    #plt.ioff()
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    
    #plt.pyplot.ioff()
    
    
    # Name = str(Name) +'.csv'
    # print(str(Name))
    
    #     #calling .csv file
    # df=pd.read_csv(str(Name))   #load .csv file
    # x=df['LET']           #x=column A
    # LET=np.array(x)
    # XS=np.max(x)
    
    # df=pd.read_csv(str(Name))   #load .csv file
    # x0=df['XS']           #x0=column B
    # XS=np.array(x0)
    # np.max(x0)
    
    # df=pd.read_csv(Name + '.csv')   #load .csv file
    # x00=df['Weibull_3']           #x=column C
    # Weibull3=np.array(x00)
    # np.max(x00)
    
    # df=pd.read_csv(Name + '.csv')   #load .csv file
    # x000=df['Weibull_4']           #x=column D
    # Weibull4=np.array(x000)
    # np.max(x000)
    
    def weibull(x, A, B, W, S):
        g = -1*((x-B/W))**S
        h = np.exp(g)
        i = 1-h
        Weib=A*i
        return (Weib)
    
    vecWeibull = np.vectorize(weibull)
    
    importedLET=LET
    a = 1/importedLET
    b = 0.5
    w = 20
    s = 1
    
    #X = np.arange(0.1,30,.1)
    X=1
    YExac = weibull(X,a,b,w,s)
    
    #Offset = 0.01
    
    # A = stats.norm.rvs(loc=a,scale=a*Offset, size = len(X))
    # B = stats.norm.rvs(loc=b,scale=b*Offset, size = len(X))
    # W = stats.norm.rvs(loc=w,scale=w*Offset, size = len(X))
    # S = stats.norm.rvs(loc=s,scale=s*Offset, size = len(X))
    
    # # plt.figure('A')
    # # plt.hist(A)
    # # plt.figure('B')
    # # plt.hist(B)
    # # plt.figure('W')
    # # plt.hist(W)
    # # plt.figure('S')
    # # plt.hist(S)
    
    
    # YExp = vecWeibull(X,A,B,W,S)
    #if importedLET>= 1:
    #    return a
    
    # plt.figure('Weibull')
    # plt.scatter(X, YExp, color = 'black', label = 'Experiment')
    # plt.plot(X, YExac, color = 'red', label = 'Exact')
    # plt.legend()
    
    return YExac


