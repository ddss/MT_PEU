#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from casadi import exp

#%% Model definition
def Model(param, x, args):
    T = x[:, 0]
    A, B, C = param[0], param[1], param[2]

    return exp(A - (B / (T + C)))  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
Estimation = EstimacaoNaoLinear(Model, symbols_x=[r'T'], symbols_y=[r'y'], symbols_param=['A','B','C'],  Folder='VapourPressuresEX3' )

#%% Defining observed data
# Input data
T = [297.1,298.2,299.3,301.2,304.2,307.2,310.2,314.1,316.2,317.8,318.2,320.2,
     323.1,326.2,329.1,331.2,334.2,337.1,340.2,343.2,346.2,349.1,352.2]
# Input data uncertainty
uxT = [0.1]*len(T)

# Output data
P = [2.93,3.21,3.49,4.22,5.60,7.31,9.12,13.07,14.98,17.63,18.02,22.08,26.95,34.61,
     40.93,50.17,63.36,78.93,93.65,115.11,140.27,171.89,208.00]
# Output data uncertainty
uP = [0.08,0.09,0.09,0.11,0.17,0.21,0.25,0.35,0.40,0.47,0.48,0.58,
      0.70,0.89,1.05,1.28,1.61,2.00,2.37,2.90,3.53,4.32,5.23]

#%% Setting the observed data set
# inputs
Estimation.setDados(0,(T,uxT))
# outputs
Estimation.setDados(1,(P,uP))

# Defining the previous data set to be used to parameter estimation
Estimation.setConjunto(dataType='estimacao')

#%% Optimization - estimating the parameters
Estimation.optimize(initial_estimative = [1, 1.5, 0.009],algorithm='bonmin')

#%% Evaluating the parameters uncertainty and coverage region
Estimation.parametersUncertainty(uncertaintyMethod='SensibilidadeModelo', objectiveFunctionMapping=True, parametersReport = True, iteracoes =500000)

#%% Evaluating model predictions
Estimation.prediction(export_y=True)

#%% Evaluating residuals and quality index
Estimation.residualAnalysis()

#%% Plotting the main results
Estimation.plots()

