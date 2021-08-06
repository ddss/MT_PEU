#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model (param,x, *args):

    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-time*exp(ko-E/T))

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of symbols for parameters written in LaTex;
# units_y: List of units of measurement for independent quantities;
# units_x: List of units of measurement of dependent quantities;
# units_param: List of units of measurement of the parameters;
# Folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(Model, symbols_x=['t','Tao'], units_x=['s','K'],
                            symbols_y=[r'y'], units_y=['adm'],
                            symbols_param=['ko','E'], units_param=['adm','K'], Folder='Example2')

#%% Defining observed data
# Input data
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

# Output data
y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]
# output uncertainty
uy = [1]*41

#%% Setting the observed data set

# inputs
Estime.setDados(0,(time,uxtime),(temperature,uxtemperature))
# outputs
Estime.setDados(1,(y,uy))

# Defining the previous data set to be used to parameter estimation
Estime.setConjunto()

#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
Estime.optimize(initial_estimative=[18, 20000.000],optimizationReport = False, algorithm='ipopt')

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
Estime.parametersUncertainty(uncertaintyMethod='2InvHessiana',objectiveFunctionMapping=True)

#%%Running the charts without prediction.
# using solely default options
#Estime.plots()

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
Estime.prediction(export_y=True, export_y_xls=True, export_cov_y=True, )

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
#Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.