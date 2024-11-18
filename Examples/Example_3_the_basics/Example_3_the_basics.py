#%% Packages importing
from sys import path #para buscar os arquivos em um diretório
path.append("../../modules")#passando o diretório da pasta raíz
from modules.MT_PEU import EstimacaoNaoLinear
from casadi import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model(param, y, gamma):
    P, T = y[0], y[1]
    A, B, C = param[0], param[1], param[2]

    return [P-exp(A - (B / (T + C)))]  # Pvp calculation - vectorized

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_gamma: List of symbols for quantity gamma;
# symbols_z: List of symbols for quantity z;
# symbols_param: List of Symbols for the parameters to be estimated;
# folder: Defines the name of the folder where the results will be saved.
Estimation = EstimacaoNaoLinear(Model, symbols_z=['P', 'T'],
                                symbols_uz=['uP', 'uT'],
                                symbols_param=['A','B','C'],
                                folder='resultado')

#%% Defining observed data manually
# Input data
T = [297.1,298.2,299.3,301.2,304.2,307.2,310.2,314.1,316.2,317.8,318.2,320.2,
     323.1,326.2,329.1,331.2,334.2,337.1,340.2,343.2,346.2,349.1,352.2]
# Input data uncertainty
uxT = [0.01]*len(T)

#%% Setting the observed data set using manual format and import
Estimation.setData(data=["data_example3", {'T':T, 'uT':uxT}])

#%% Optimization - estimating the parameters,
# initial_estimative: List with the initial estimates for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# report: Informs whether the parameters report should be created (True or False).
Estimation.optimize(initial_estimative = [1, 1.5, 0.009]+Estimation.z.observed['estimation'].lista_estimativa,
                    lower_bound=[-50,-1e4,-50]+[0]*len(T)+[200]*len(T),
                    upper_bound=[50,1e4,50]+[300]*len(T)+[400]*len(T),
                    algorithm='ipopt',
                    optimizationReport = True,
                    report= False)

#%% Evaluating the parameters uncertainty and coverage region
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# report: Informs whether the parameters report should be created (True or False).
# iterations: Number of iterations to perform the mapping of the objective function. The higher the better mapping, but it
# increases the execution time
Estimation.setupSolveModel(['P'], ['T'])

Estimation.uncertainty(objectiveFunctionMapping=True, iterations=20000,  searchLimitFactor=1/10, compresscov=5e3)

#%% Evaluating residuals and quality index
# using solely default options
Estimation.residualAnalysis()

#%% Plotting the main results
# using solely default options
Estimation.plots()

