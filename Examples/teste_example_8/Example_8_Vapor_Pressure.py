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
Estimation = EstimacaoNaoLinear(Model, symbols_x=[r'T'],symbols_ux=[r'uT'] ,symbols_y=[r'P'],symbols_uy=['uP'], symbols_param=['A','B'],  Folder='Example8' )

#%% Setting the observed data set
#Estimation.setDados(data={0:[(T,uT)],1:[(P,uP)]},glx=[],gly=[])
Estimation.setDados(data="data_exa8.xlsx",glx=[], gly=[])
# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;
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

#%% Model definition linear

#%% Packages importing
from MT_PEU_Linear import EstimacaoLinear
from numpy import log, array

#%% Starting the MT_PEU main object
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# Folder: Defines the name of the folder where the results will be saved.
ER = EstimacaoLinear(symbols_x=[r'X1'],symbols_ux=[r'uX1'], symbols_y=[r'Y1'],symbols_uy=[r'uY1'] ,symbols_param=['A1','B1'],  folder='Example8Linear')
#%% Defining observed data
# Input data
T =Estimation.x.estimacao.matriz_estimativa.transpose()[0]
# Input data uncertainty
uT =Estimation.x.estimacao.matriz_incerteza.transpose()[0]

# Output data
P =Estimation.y.estimacao.matriz_estimativa.transpose()[0]
# Output data uncertainty
uP =Estimation.y.estimacao.matriz_incerteza.transpose()[0]


#Input observed data
X = 1./T
#Output observed data
Y = log(P) - (-68.2 / 8.31446)*log(T/298.15)

#Propagation of uncertainty
#uncertainty of input observed data
uX = ((-1/(T**2))**2*uT**2)**0.5

#uncertainty of output observed data
uY = ((1/P)**2*uP**2 + (-1*-68.2/(8.31446*T))**2*uT**2)**0.5

#%% Setting the observed data set


# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;
ER.setDados(data={'X1':X,'uX1':uX,'Y1':Y,'uY1':uY},glx=[],gly=[])

#%% Optimization - estimating the parameters
# parametersReport: Informs whether the parameters report should be created (True or False).
ER.optimize(parametersReport=True)

#%% Evaluating the parameters uncertainty and coverage region
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
ER.parametersUncertainty(objectiveFunctionMapping=True)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
ER.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
ER.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
ER.plots()
