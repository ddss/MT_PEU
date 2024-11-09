#%% Packages importing
from sys import path
path.append("../../modules")#A list of strings that specifies the search path for modules
from modules.MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The subroutine that specifies the equations with their respective parameters.
def model(param, y, x, *args):

    ko, E = param[0], param[1]
    reacfrac, time, T = y[0], y[1], y[2]

    return [reacfrac - exp(-(ko*10**17)*time*exp(-E/T))]

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_y: list of symbols for quantity y;
# symbols_uy: list of symbols for uncertainty y;
# symbols_param: list of symbols for the parameters to be estimated;
# Folder: string with the name of the folder where reports and charts will be saved;
Estime = EstimacaoNaoLinear(model, symbols_y=['frac', 'time', 'temperature'],symbols_uy=['ufrac', 'utime', 'utemperature'],
                            symbols_param=['ko','E'], Folder='resultadoimplicito')

#%% Defining the observed data set
Frac = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]
# uncertainty of dependent variables
ufrac = [1]*41
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

#Data entry manual
Estime.setDados(data={'time':time,'utime':uxtime,'temperature':temperature,
                      'utemperature':uxtemperature,'frac':Frac,'ufrac':ufrac})


#%% Optimization - estimating the parameters
# initial_estimate: list containing initial estimate for optimization algorithm
Estime.optimize(initial_estimative=[0.5,25000]+Frac+time+temperature,
                lower_bound=[0,20000]+[0]*41+[0]*41+[500]*41,
                upper_bound=[1,30000]+[1]*41+[200]*41+[700]*41)

#%% Evaluating the parameters uncertainty and coverage region
# using solely default options
Estime.uncertainty()

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