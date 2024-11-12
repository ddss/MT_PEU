#%% Packages importing
from sys import path #para buscar os arquivos em um diretório
path.append("../../modules")#passando o diretório da pasta raíz
from modules.MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def model(param, z, gamma):
    ko, E = param[0], param[1]
    frac, time, T = z[0], z[1], z[2]

    return [frac - exp(-ko * time * exp(-E * (1 / T - 1. / 630.)))]

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_gamma: List of symbols for quantity gamma;
# symbols_z: List of symbols for quantity z;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of symbols for parameters written in LaTex;
# label_latex_x: List of symbols for quantities gamma written in LaTex
# units_y: List of units of measurement for independent quantities;
# units_param: List of units of measurement of the parameters;
# units_x: List of units of measurement of dependent quantities;
# folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(model, symbols_z=['frac', 'time', 'temperature'], symbols_uz=['ufrac', 'utime', 'utemperature'],
                            symbols_param=['ko','E'], folder='resultado')


#%% Setting the observed data set
Estime.setData(data="data_exa5")

#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# upper_bound: List with the upper bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# report: Informs whether the parameters report should be created (True or False).
NE = Estime.z.observed['estimation'].NE
Estime.optimize(initial_estimative=[0.005, 20000.000]+Estime.z.observed['estimation'].lista_estimativa,
                lower_bound=[0.006,15000]+[0]*NE+[0]*NE+[500]*NE,
                upper_bound=[100,30000]+[1]*NE+[200]*NE+[700]*NE)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# limite_inferior: Lower limit of parameters;
# limite_superior: Upper limit of the parameters;
# iterations: Number of iterations to perform the mapping of the objective function. The higher the better mapping, but it
# increases the execution time
# report: Informs whether the parameters report should be created.

Estime.uncertainty()

#%% Evaluating model predictions
# export_y: Exports the calculated data of z, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of z, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of z (True or False);
# export_x: Exports the calculated data of gamma, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of gamma (True or False).
Estime.setupSolveModel(['frac'],['temperature','time'])

Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis(report=True)

#%% Plotting the main results
#Estime.plots()

# =================================================================================
# OPTIONAL: VALIDATION
# =================================================================================

u"""
If the user wishes, it is possible to do the same analysis as before with the prediction data. 
The procedure to be followed is similar to the one previously carried out. The only difference is in the argument inserted in the setConjunto method.
Instead of "type = observed" it becomes "type = predicao". It is necessary to enter at least 4 data for each prediction variable. 
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
Estime.setData(data={'time':time, 'utime':uxtime, 'temperature':temperature, 'utemperature':uxtemperature, 'frac':y, 'ufrac':uy1})

# Defining the previous data set to be used to validation
# dataType: Defines the purpose of the informed data set: observed, predicao.
# glx: Degrees of freedom of quantity gamma;
# gly: Degrees of freedom of quantity z;


#%% Evaluating model predictions for the validation data
# export_y: Exports the calculated data of z, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of z, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of z (True or False);
# export_x: Exports the calculated data of gamma, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of gamma (True or False).
Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.

