#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
def Model(param,x,*args):

    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-(ko*10**17)*time*exp(-E/T))

#%% Starting the MT_PEU main object
Estime = EstimacaoNaoLinear(Model, symbols_x=['t','Tao'], symbols_y=['y'], symbols_param=['ko','E'], Folder='KineticModelEX1')

#%% Defining observed data
# Output data
y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]
# output uncertainty
uy = [1]*41
# input 1
time = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]
# input 1 uncertainty
uxtime = [1]*41
# input 2
temperature = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]
# input 2 uncertainty
uxtemperature = [1]*41

#%% Setting the observed data set
# inputs
Estime.setDados(0,(time,uxtime),(temperature,uxtemperature))
# outputs
Estime.setDados(1,(y,uy))

# Defining the previous data set to be used to parameter estimation
Estime.setConjunto(type='estimacao')

#%% Optimization - estimating the parameters
Estime.optimize(initial_estimative=[0.5,25000], algorithm='ipopt')

#%% Evaluating the parameters uncertainty and coverage region
Estime.parametersUncertainty(uncertaintyMethod='Geral')

#%% Evaluating model predictions
Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.