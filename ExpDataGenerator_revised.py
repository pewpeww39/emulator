"""
LABVIEW uses functions to call the python script (MUST BE SET UP LIKE THIS).
It is going to export the LET and the Name (from the add device feature) in 
order to more efficiently describe the device.

"""


def LV2PY (LET, Name) :
# %%
    import numpy as np
    # import scipy.stats as stats
    import pandas as pd
    
    # %%    Creating DataFrame
    """
    The name is imported as a string and the .csv is added to the Name 
    variable and used to import the .csv file containing the data for 
    the test.
       
    Then DataFrames are created to organize the data by columns 
    """
    Name = str(Name) +'.csv'        #a file compatible string
                                    #                   for the imported name 
    
    
         
    XS= pd.DataFrame(pd.read_csv(Name, index_col=[0]),  #creating Dataframe 
                     columns=['XS', 'Eff. LET'])
    XS=XS.sort_values(by = ['Eff. LET'], ascending = True)
    Weibull_parameters=pd.DataFrame(pd.read_csv(Name, index_col=[0]),
                                    index=[0], columns=['Limiting XS',
                                                        'Onset LET','Width',
                                                        'Shape'])
    # print(XS)
    # %%        Weibull Equation
    """
          The Weibull equation for the expected average value of the
          cross-section VS. LET 
         
          Will use this value as the mean of the normal distribution for 
          the generated data set.  
         
    """

    def weibull(x, A, B, W, S):
        return (A*(1 - np.exp(-1*((np.log(x) - B)/W)**S)))

    XS_mean = weibull(LET,
                      Weibull_parameters['Limiting XS'],
                      Weibull_parameters['Onset LET'],
                      Weibull_parameters['Width'],
                      Weibull_parameters['Shape'])
    # print(XS_mean)

    #%%     sub Dataframe for different LET dataframe locations
    
    LET_value_changes = XS["Eff. LET"].shift() != XS["Eff. LET"]
    counter=0
    counterB=0
    Pairs = {'Start':counterB,
            'Stop': counter}
    row_index=pd.DataFrame(Pairs , index=[], columns= ['Start', 'Stop'])
    for LET_value_changes[1] in LET_value_changes:          #Vectorizing inclusion values
        if (LET_value_changes[1]==bool(True)): 
            row_index.loc[len(row_index.index)]= [counterB, counter] 
            counterB=counter
        
        counter+=1
    
    columns=0           #columns=8 is for LET 5.7
    XS_iter= []
    STD_subArr=pd.DataFrame(XS_iter, index=[], columns=['STD'], dtype=float )
    LET_subArr=pd.DataFrame(XS_iter, index=[], columns=['LET'], dtype=float )
    
    for rows in range(row_index.shape[0]):         #vectorizing Eff. LET
        
        XS_sub=row_index.at[columns,'Start']
        STD_sub=row_index.at[columns,'Stop']      
        RI= XS.iloc[XS_sub: STD_sub]
        RI1=RI['XS']
        RI2=RI['Eff. LET'].mean()
        STD_subArr.loc[len(STD_subArr.index)]=[RI1.std()]
        LET_subArr.loc[len(LET_subArr.index)]=RI2
        columns+=1
    STD_subArr=STD_subArr.dropna()
    LET_subArr=LET_subArr.dropna()
    Linear_fitArr=STD_subArr.join(LET_subArr, how='left')
    print(Linear_fitArr)
    



    #%%
    """
    Using a linear equation of LET vs. STD to determine the STD
    Can index the columns using: Linear_fitArr['STD'] or Linear_fitArr['LET'])
    """
    
    
    
    #%%
    """
    This is where the data sets for points not tested for will be created 
    based off the mean from the weibull and standard deviations (std).
    
    after generating the new data set for the importedLET, we noramlly 
    distribute it.
    
    we can use a random number generator(RNG) to determine what our
    z-score is and calculate a XS.
    """

    # #Offset = 0.01
    
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
    

 
    # """
    #                             (work in progress)
    # This is where I'm trying to extract the cross-section values corresponding 
    # to the LET. 
    
    # Once the data is "sorted" for the choosen LET we can use the sub-array to 
    # find the Standard Deviation.
    
    # Example: 
    #         If the exported LET from LABVIEW = the LET imported from the data,
    #         give the XS for the ordered pair.
            
    #         for the values in column[0] find the standard deviation using
    #         numpy.std

    # """
    # print(LET)
    # x2=[]
    # with np.nditer(weib_array, flags=['multi_index'],\
    #                 op_flags=['writeonly']) as it:
    #     for x in it:
    #         if x == LET:
    #             x2=[it.multi_index[0],it.multi_index[1]]
    #             # x2= weib_array[x1]
    #             # x1 = weib_array[x2]
    #             print("<%s>" % ([x2]), end='\n ')
    
    
    
    # #%%
    # """
    # finding the sample error gives us the std of the mean of sub-arrays
    
    # This is where the data sets for points not tested for will be created 
    # based off the mean and standard deviations (std).
    
    # after selecting the data set for the exported LET from labview and the XS,
    # we can use a random number generator(RNG) to determine what our
    # probability is.
    
    # next compare probability of RNG to probability of normal distribution to 
    # select the XS to be exported to LABVIEW
    
    # """

    # #Offset = 0.01
    
    # # A = stats.norm.rvs(loc=a,scale=a*Offset, size = len(X))
    # # B = stats.norm.rvs(loc=b,scale=b*Offset, size = len(X))
    # # W = stats.norm.rvs(loc=w,scale=w*Offset, size = len(X))
    # # S = stats.norm.rvs(loc=s,scale=s*Offset, size = len(X))
    
    # # # plt.figure('A')
    # # # plt.hist(A)
    # # # plt.figure('B')
    # # # plt.hist(B)
    # # # plt.figure('W')
    # # # plt.hist(W)
    # # # plt.figure('S')
    # # # plt.hist(S)
    
    
    # # YExp = vecWeibull(X,A,B,W,S)
    # #if importedLET>= 1:
    # #    return a
    
    # # plt.figure('Weibull')
    # # plt.scatter(X, YExp, color = 'black', label = 'Experiment')
    # # plt.plot(X, YExac, color = 'red', label = 'Exact')
    # # plt.legend()
    
    # return XS_avg




