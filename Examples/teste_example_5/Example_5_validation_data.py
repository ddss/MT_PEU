#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model(param, x, *args):
    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-ko * time * exp(-E * (1 / T - 1. / 630.)))

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: List of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of symbols for parameters written in LaTex;
# label_latex_x: List of symbols for quantities x written in LaTex
# units_y: List of units of measurement for independent quantities;
# units_param: List of units of measurement of the parameters;
# units_x: List of units of measurement of dependent quantities;
# Folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(Model, symbols_x=['Time', 'Temperature'], symbols_ux=['UxTime','Uxtemperature'], units_x=['s', 'K'], label_latex_x=[r'$t$', '$T$'],
                            symbols_y=[r'Y'],symbols_uy=[r'uY'], units_y=['adm'],
                            symbols_param=['ko', 'E'], units_param=['adm','K'],
                            label_latex_param=[r'$k_o$', r'$E$'],
                            Folder='Example5')


#%% Setting the observed data set
#Estime.setDados(data=["data_exa5_independent.csv","data_exa5_dependent.csv"])
Estime.setDados(data="data_exa5",glx=[], gly=[])


# Defining the previous data set to be used to parameter estimation
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;


#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# upper_bound: List with the upper bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# parametersReport: Informs whether the parameters report should be created (True or False).
Estime.optimize(initial_estimative=[0.005, 20000.000], algorithm='ipopt', lower_bound=[0.006,15000], upper_bound=[100,30000],
                optimizationReport = True, parametersReport = False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# limite_inferior: Lower limit of parameters;
# limite_superior: Upper limit of the parameters;
# iterations: Number of iterations to perform the mapping of the objective function. The higher the better mapping, but it
# increases the execution time
# parametersReport: Informs whether the parameters report should be created.
Estime.parametersUncertainty(uncertaintyMethod='Geral',objectiveFunctionMapping=True, lower_bound=[7.2e-3,26400], upper_bound=[7.7e-3,28600],
                             parametersReport = True, iterations=200)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
Estime.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
Estime.residualAnalysis(report=True)

#%% Plotting the main results
Estime.plots()

# =================================================================================
# OPTIONAL: VALIDATION
# =================================================================================

u"""
If the user wishes, it is possible to do the same analysis as before with the prediction data. 
The procedure to be followed is similar to the one previously carried out. The only difference is in the argument inserted in the setConjunto method.
Instead of "type = estimacao" it becomes "type = predicao". It is necessary to enter at least 4 data for each prediction variable. 
"""

#%% Setting the validation data set
#inputs

# input 1
time = [60.0,120.0,60.0,60.0,60.0,60.0,60.0,30.0,30.0,90.0,60.0,30.0]
# input 1 uncertainty
uxtime = [0.2]*12
# input 2
temperature = [600.0,612.0,612.0,620.0,620.0,620.0,620.0,639.0,639.0,620.0,620.0,631.0]
# input 2 uncertainty
uxtemperature = [0.2]*12

# output
y = [0.949,0.785,0.890,0.782,0.800,0.802,0.799,0.655,0.638,0.712,0.794,0.717]
# output uncertainty
uy1 = [0.2]*12

#%% Setting the observed data set
# inputs
Estime.setDados(data={'Time':time,'UxTime':uxtime,'Temperature':temperature,'Uxtemperature':uxtemperature,'Y':y,'uY':uy1})

# Defining the previous data set to be used to validation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;


#%% Evaluating model predictions for the validation data
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
Estime.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
Estime.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.

