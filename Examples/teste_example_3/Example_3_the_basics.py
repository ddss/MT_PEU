#%% Packages importing
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
# symbols_x: List of ymbols for quantity x;
# symbols_y: List of ymbols for quantity y;
# symbols_param: List of Symbols for the parameters to be estimated;
# Folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoNaoLinear(Model, symbols_x=['T'],symbols_ux=['uT'], symbols_y=['P'],symbols_uy=['uP'], symbols_param=['A','B','C'],  Folder='Example3' )

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
#Estimation.setDados(data={0:[(T,uxT)],1:[(P,uP)]})

Estimation.setDados(data="data_exa3.1.xlsx")

#Estimation.setDados(data=["data_exa3_independent","data_exa3_dependent.csv"])


# Defining the previous data set to be used to parameter estimation.
# dataType: Defines the purpose of the informed data set: estimacao, predicao.


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

