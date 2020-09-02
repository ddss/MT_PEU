#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from casadi import exp, log

#%% Model definition non-liear
# def Model: The def model specifies the equations with their respective parameters.
def Model(param, x, args):
    T = x[:, 0]
    A, B = param[0], param[1]

    return exp(A / 8.31446 + B / (8.31446 * T) - (68.2 / 8.31446) * log(T / 298.15))  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# Folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoNaoLinear(Model, symbols_x=[r'T'], symbols_y=[r'P'], symbols_param=['A','B'],  Folder='VapourpressuresNonLinearEX8' )

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

#%% Setting the observed data set
# inputs
Estimation.setDados(0,(T,uT))
# outputs
Estimation.setDados(1,(P,uP))

# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;
Estimation.setConjunto(dataType='estimacao', glx=[], gly=[])

#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# parametersReport: Informs whether the parameters report should be created (True or False).
Estimation.optimize(initial_estimative = [200, -80680.1], algorithm='ipopt', optimizationReport = True, parametersReport = False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# parametersReport: Informs whether the parameters report should be created.
Estimation.parametersUncertainty(uncertaintyMethod='SensibilidadeModelo', objectiveFunctionMapping=True, parametersReport = True)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
Estimation.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
Estimation.residualAnalysis(report=True)

#%% Plotting the main results
Estimation.plots()

#%% Model definition liear

#%% Packages importing
from MT_PEU_Linear import EstimacaoLinear
from numpy import log, array

#%% Starting the MT_PEU main object
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# Folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoLinear(symbols_x=[r'X1'], symbols_y=[r'Y1'], symbols_param=['A1','B1'],  folder='VapourpressuresLinearEX8')
#Input observed data
X = 1./array(T)
#Output observed data
Y = log(P) - (-68.2 / 8.31446)*log(array(T)/298.15)

#Propagation of uncertainty
#uncertainty of input observed data
uX = ((-1/(array(T)**2))**2*array(uT)**2)**0.5

#uncertainty of output observed data
uY = ((1/array(P))**2*array(uP)**2 + (-1*-68.2/(8.31446*array(T)))**2*array(uT)**2)**0.5

#%% Setting the observed data set
# inputs
Estimation.setDados(0, (X.tolist(), uX.tolist()))
# outputs
Estimation.setDados(1, (Y.tolist(), uY.tolist()))

# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;
Estimation.setConjunto(dataType='estimacao', glx=[], gly=[])

#%% Optimization - estimating the parameters
# parametersReport: Informs whether the parameters report should be created (True or False).
Estimation.optimize(parametersReport=True)

#%% Evaluating the parameters uncertainty and coverage region
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
Estimation.parametersUncertainty(objectiveFunctionMapping=True)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
Estimation.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
Estimation.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
Estimation.plots()
