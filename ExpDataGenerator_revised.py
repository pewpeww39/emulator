"""
LABVIEW uses functions to call the python script (MUST BE SET UP LIKE THIS).
It is going to export the LET and the Name (from the add device feature) in 
order to more efficiently describe the device.

"""


def LV2PY (LET, Name) :
# %%
    #import matplotlib.pyplot as plt
    #from matplotlib.backends.backend_agg import FigureCanvasAgg
    #from matplotlib.figure import Figure
    import numpy as np
    import scipy.stats as stats
    import pandas as pd
    #import csv
    
    #plt.pyplot.ioff()
    # %%
    """
    The name is imported as a string and the .csv is added to the Name variable
    and used to import the .csv file containing the data for the test.
       
    Then arrays are created to organize the data by columns 
    """
    Name = str(Name) +'.csv'        #a file compatible string for the imported name 
    print(str(Name))

    results = pd.read_csv(Name)     #read .csv file
    type(results)
    data=np.array(results[0:113])   #creating array from .csv file
    weib_array=data[0:113, :2]      #Rows=113   first 2 colums
    XS=(data[0:113, 0])             #Rows=113   first column
    Eff_LET=data[0:113, 1]          #rows=113   second column
    limiting_XS=(data[:1 , 3])      #row=1      Third column    
    onset_LET=data[:1 , 4]          #row=1      Fourth column
    width=data[:1,5]                #row=1      Fifth column
    shape=data[:1, 6]               #row=1      Sixth column
    
    LETT=float(LET)
    print(LETT)
    print (weib_array)
    
    """
                                (work in progress)
    This is where I'm trying to extract the cross-section values corresponding 
    to the LET. 
    
    Once the data is "sorted" for the choosen LET we can use the sub-array to 
    find the Standard Deviation.
    
    Example: 
            If the exported LET from LABVIEW = the LET imported from the data,
            give the XS for the ordered pair.
            
            for the values in column[0] find the standard deviation using
            numpy.std
            
            
            
    """
    print(LET)
    x2=[]
    with np.nditer(weib_array, flags=['multi_index'], op_flags=['writeonly']) as it:
        for x in it:
            if x == LET:
                x2=[it.multi_index[0],it.multi_index[1]]
                # x2= weib_array[x1]
                # x1 = weib_array[x2]
                print("<%s>" % ([x2]), end='\n ')
                
            

    # %%
    """
         The Weibull equation for the expected average value of the cross-section VS. LET
         Will use this value as the mean of the Weibull distribution.  
         
         
    """

    def weibull(x, A, B, W, S):
        return (A*(1 - np.exp(-1*((np.log(x) - B)/W)**S)))
    
    # vecWeibull = np.vectorize(weibull)

    # print(limiting_XS)
    
    importedLET = LET
    XS_avg = weibull(importedLET,limiting_XS,onset_LET,width,shape)
    
    #%%
    """
    finding the sample error gives us the std of the mean of sub-arrays
    
    This is where the data sets for points not tested for will be created based
    off the mean and standard deviations (std).
    
    after selecting the data set for the exported LET from labview and the XS,
    we can use a random number generator(RNG) to determine what our probability is.
    
    next compare probability of RNG to probability of normal distribution to 
    select the XS to be exported to LABVIEW
    
    """

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
    
    return XS_avg


