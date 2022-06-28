#%% Packages importing
from sys import path #para buscar os arquivos em um diretório
path.append("../../modules")#passando o diretório da pasta raíz
from MT_PEU import EstimacaoNaoLinear
from casadi import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model(param, x, args):
    T = x[:, 0]
    A, B, C = param[0], param[1], param[2]

    return exp(A - (B / (T + C)))  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of Symbols for the parameters to be estimated;
# Folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoNaoLinear(Model, symbols_x=['T'],symbols_ux=['uT'], symbols_y=['P'],symbols_uy=['uP'], symbols_param=['A','B','C'],  Folder='Example3' )

#%% Defining observed data manually
# Input data
T = [297.1,298.2,299.3,301.2,304.2,307.2,310.2,314.1,316.2,317.8,318.2,320.2,
     323.1,326.2,329.1,331.2,334.2,337.1,340.2,343.2,346.2,349.1,352.2]
# Input data uncertainty
uxT = [0.1]*len(T)

#%% Setting the observed data set using manual format and import
Estimation.setDados(data=["data_example3",{'T':T,'uT':uxT}])

#%% Optimization - estimating the parameters,
# initial_estimative: List with the initial estimates for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# parametersReport: Informs whether the parameters report should be created (True or False).
Estimation.optimize(initial_estimative = [1, 1.5, 0.009],algorithm='bonmin', optimizationReport = True, parametersReport = False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters: 2InvHessian, Geral, SensibilidadeModelo
# Geral obtains the parameters uncertainty matrix without approximations (most accurate), while 2InvHessian and SensibilidadeModelo involves
# some approximations.
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# parametersReport: Informs whether the parameters report should be created (True or False).
# iterations: Number of iterations to perform the mapping of the objective function. The higher the better mapping, but it
# increases the execution time
Estimation.parametersUncertainty(uncertaintyMethod='Geral', objectiveFunctionMapping=True, iterations=5000, parametersReport = True)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
Estimation.prediction(export_y=True, export_y_xls=True, export_cov_y=True, export_x=True)

#%% Evaluating residuals and quality index
# using solely default options
Estimation.residualAnalysis()

#%% Plotting the main results
# using solely default options
Estimation.plots()

