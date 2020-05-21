#%% Packages importing
from MT_PEU_Linear import EstimacaoLinear
from numpy import log, array

#%% Starting the MT_PEU main object
Estimation = EstimacaoLinear(symbols_x=[r'X'], symbols_y=[r'Y'], symbols_param=['A','B'],  folder='FormamideLinear')

#%% Defining observed data
# Input data
T = [297.1,298.2,299.3,301.2,304.2,307.2,310.2,314.1,316.2,317.8,318.2,320.2,
     323.1,326.2,329.1,331.2,334.2,337.1,340.2,343.2,346.2,349.1,352.2]
# Input data uncertainty
uT = [0.1]*len(T)

# Output data
P = [2.93,3.21,3.49,4.22,5.60,7.31,9.12,13.07,14.98,17.63,18.02,22.08,26.95,34.61,
     40.93,50.17,63.36,78.93,93.65,115.11,140.27,171.89,208.00]
# Output data uncertainty
uP = [0.08,0.09,0.09,0.11,0.17,0.21,0.25,0.35,0.40,0.47,0.48,0.58,
      0.70,0.89,1.05,1.28,1.61,2.00,2.37,2.90,3.53,4.32,5.23]

#%%Definition of new variables
#Input observed data
X = [1 / float(i) for i in T]
#Output observed data
Y = log(P) - [(-68.2 / 8.31446)*log(i/298.15) for i in T]

#Propagation of uncertainty
#uncertainty of input observed data
uX = (((-1/array(T))**2*array(uT)**2)**0.5).tolist()

#uncertainty of output observed data
uY = (((1/array(P))**2*array(uP)**2 + (-(-68.2/(8.31446*298.15*array(T))))**2*array(uT)**2)**0.5).tolist()

#%% Setting the observed data set
# inputs
Estimation.setDados(0,(X,uX))
# outputs
Estimation.setDados(1,(Y,uY))

# Defining the previous data set to be used to parameter estimation
Estimation.setConjunto(type='estimacao')

#%% Optimization - estimating the parameters
Estimation.optimize()

#%% Evaluating the parameters uncertainty and coverage region
Estimation.parametersUncertainty(objectiveFunctionMapping=True)

#%% Evaluating model predictions
Estimation.prediction()

#%% Evaluating residuals and quality index
Estimation.residualAnalysis()

#%% Plotting the main results
Estimation.plots()

#%% Calculation of parameters based on the original model

A = Estimation.parametros.estimativa[1]*8.31446
B = Estimation.parametros.estimativa[0]*8.31446
a=1