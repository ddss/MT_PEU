#%% Packages importing
from sys import path #para buscar os arquivos em um diretório
path.append("../../modules")#passando o diretório da pasta raíz
from modules.MT_PEU_Linear import EstimacaoNaoLinear
from casadi import exp, log

#%% Model definition non-liear
# def Model: The def model specifies the equations with their respective parameters.
def Model(param, z, gamma):
    P, T = z[0], z[1]
    A, B = param[0], param[1]

    return [P - exp(A / 8.31446 + B / (8.31446 * T) - (68.2 / 8.31446) * log(T / 298.15))]  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_gamma: List of symbols for quantity gamma;
# symbols_z: List of symbols for quantity z;
# symbols_param: List of symbols for the parameters to be estimated;
# folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoNaoLinear(Model, symbols_z=['P','T'], symbols_uz=['uP','uT'], symbols_param=['A', 'B'], folder='result')

#%% Setting the observed data set

Estimation.setData(data="data_exa8.xlsx")
# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: observed, predicao.
# glx: Degrees of freedom of quantity gamma;
# gly: Degrees of freedom of quantity z;
#%% Plotting the main results
Estimation.plots()
#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# report: Informs whether the parameters report should be created (True or False).
Estimation.optimize(initial_estimative = [200, -80680.1]+Estimation.z.observed['estimation'].lista_estimativa,
                    lower_bound=[0, -1e6]+[0*data for data in Estimation.z.observed['estimation'].lista_estimativa],
                    upper_bound=[1000, 1e6]+[10*data for data in Estimation.z.observed['estimation'].lista_estimativa],
                    algorithm='ipopt',
                    optimizationReport = True,
                    report= False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# report: Informs whether the parameters report should be created.
Estimation.setupSolveModel(['P'], ['T'])

Estimation.uncertainty(objectiveFunctionMapping=True, iterations=100,  searchLimitFactor=1/10, compresscov=5e3)

#%% Evaluating model predictions
# export_y: Exports the calculated data of z, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of z, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of z (True or False);
# export_x: Exports the calculated data of gamma, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of gamma (True or False).
Estimation.prediction()

#%% Evaluating residuals and quality index
Estimation.residualAnalysis(report=True)

#%% Plotting the main results
Estimation.plots()

#%% Packages importing
# from modules.MT_PEU_Linear import EstimacaoLinear
# from numpy import log
#
# #%% Starting the MT_PEU main object
# # symbols_gamma: List of symbols for quantity gamma;
# # symbols_z: List of symbols for quantity z;
# # symbols_param: List of symbols for the parameters to be estimated;
# # folder: Defines the name of the folder where the results will be saved.
# ER = EstimacaoLinear(symbols_gamma=[r'X1'], symbols_ux=[r'uX1'], symbols_z=[r'Y1'], symbols_uz=[r'uY1'], symbols_param=['A1', 'B1'], folder='Ex8Linear')
# #%% Defining observed data
# # Input data
# T =Estimation.gamma.observed.matriz_estimativa.transpose()[0]
# # Input data uncertainty
# uT =Estimation.gamma.observed.matriz_incerteza.transpose()[0]
#
# # Output data
# P =Estimation.z.observed.matriz_estimativa.transpose()[0]
# # Output data uncertainty
# uP =Estimation.z.observed.matriz_incerteza.transpose()[0]
#
#
# #Input observed data
# X = 1./T
# #Output observed data
# Y = log(P) - (-68.2 / 8.31446)*log(T/298.15)
#
# #Propagation of uncertainty
# #uncertainty of input observed data
# uX = ((-1/(T**2))**2*uT**2)**0.5
#
# #uncertainty of output observed data
# uY = ((1/P)**2*uP**2 + (-1*-68.2/(8.31446*T))**2*uT**2)**0.5
#
# #%% Setting the observed data set
#
# # Defining the previous data set to be used to parameter estimation
# # dataType: Defines the purpose of the informed data set: observed, predicao.
# # glx: Degrees of freedom of quantity gamma;
# # gly: Degrees of freedom of quantity z;
# ER.setData(data={'X1':X, 'uX1':uX, 'Y1':Y, 'uY1':uY}, glx=[], gly=[])
#
# #%% Optimization - estimating the parameters
# # report: Informs whether the parameters report should be created (True or False).
# ER.optimize(report=True)
#
# #%% Evaluating the parameters uncertainty and coverage region
# # objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# ER.parametersUncertainty(objectiveFunctionMapping=True)
#
# #%% Evaluating model predictions
# # export_y: Exports the calculated data of z, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# # export_y_xls: Exports the calculated data of z, its uncertainty, and degrees of freedom in a xls (True or False);
# # export_cov_y: Exports the covariance matrix of z (True or False);
# # export_x: Exports the calculated data of gamma, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# # export_cov_x: Exports the covariance matrix of gamma (True or False).
# ER.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)
#
# #%% Evaluating residuals and quality index
# ER.residualAnalysis(report=True)
#
# #%% Plotting the main results
# # using solely default options
# ER.plots()
