#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from casadi import exp

#%% Model definition
def Model(param, x, args):
    T = x[:, 0]
    A, B, C = param[0], param[1], param[2]

    return exp(A - (B / (T + C)))  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
Estimation = EstimacaoNaoLinear(Model, symbols_x=[r'T'], symbols_y=[r'y'], symbols_param=['TAO','tao','C'],  Folder='Example5' )

#%% Defining observed data
# Input data
T = [288.8, 292.2, 295.1, 300.5, 300.7, 303.7, 305.7, 310.5, 315.7, 320.8, 323.2,
     325.8, 330.9, 335.9, 338.5, 340.9, 346, 348.5, 350.9, 356.1]
# Input data uncertainty
uxT = [1]*len(T)

# Output data
P = [75.1, 96.1, 117.3, 167.7, 169, 211, 248.2, 326.5, 464.6, 627.4, 709.4,
     853.4, 1093, 1478.8, 1732.8, 1947.5, 2460.1, 2893.3, 3208.9, 4163.7]
# Output data uncertainty
uP  = [1.9, 2.4, 3.0, 4.2, 4.3, 5.3, 6.2, 8.2, 11.6, 15.7, 17.8, 21.4, 27.4, 37.0, 43.3, 48.7, 61.5, 72.4, 80.2, 104.1]

#%% Setting the observed data set
# inputs
Estimation.setDados(0,(T,uxT))
# outputs
Estimation.setDados(1,(P,uP))

# Defining the previous data set to be used to parameter estimation
Estimation.setConjunto(type='estimacao')

#%% Optimization - estimating the parameters
Estimation.optimize(initial_estimative = [1, 1.5, 0.009])

#%% Evaluating the parameters uncertainty and coverage region
Estimation.parametersUncertainty(uncertaintyMethod='SensibilidadeModelo', objectiveFunctionMapping=True)

#%% Evaluating model predictions
Estimation.prediction()

#%% Evaluating residuals and quality index
Estimation.residualAnalysis()

#%% Plotting the main results
Estimation.plots()

