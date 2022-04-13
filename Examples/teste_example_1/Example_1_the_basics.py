#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp,array
#%% Model definition
# def Model: The subroutine that specifies the equations with their respective parameters.
def Model(param, x, *args):

    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-(ko*10**17)*time*exp(-E/T))

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: list of symbols for quantity x;
# symbols_ux: list of symbols for uncertainty x;
# symbols_y: list of symbols for quantity y;
# symbols_uy: list of symbols for uncertainty y;
# symbols_param: list of symbols for the parameters to be estimated;
# Folder: string with the name of the folder where reports and charts will be saved;
Estime = EstimacaoNaoLinear(Model, symbols_x=['Time','Temperature'],symbols_ux=['UxTime','Uxtemperature']\
,symbols_y=['Y'] ,symbols_uy=['uY'], symbols_param=['ko','E'], Folder='Example1')


#%% Defining the observed data set
y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]
# uncertainty of dependent variables
uy = [1]*41
# Observed data of independent variable (input 1)
time = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]
# input 1 uncertainty
uxtime = [1]*41
# Observed data of independent variable (input 2)
temperature = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]
# input 2 uncertainty
uxtemperature = [1]*41

#Setting manual data entry

#Estime.setDados(data={'Time':time,'UxTime':uxtime,'Temperature':temperature,'Uxtemperature':uxtemperature,'Y':y,'uY':uy})

#Data entry using .XLSX

Estime.setDados(data=["data_exa1"])

#Data entry using .CSV
#Estime.setDados(data=["data_exa1_independent","data_exa1_dependent"])

#%% Optimization - estimating the parameters
# initial_estimate: list containing initial estimate for optimization algorithm
Estime.optimize(initial_estimative=[0.5,25000])

#%% Evaluating the parameters uncertainty and coverage region
# using solely default options
Estime.parametersUncertainty()

#%% Evaluating model predictions
# using solely default options
Estime.prediction()

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
# using solely default options
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros.
# Rio de Janeiro: e-papers, 2007.
#%%