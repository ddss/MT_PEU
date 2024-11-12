#%% Packages importing
from sys import path
path.append("../../modules")#A list of strings that specifies the search path for modules
from modules.MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def model(param, y, x, *args):

    ko, E = param[0], param[1]
    reacfrac, time, T = y[0], y[1], y[2]

    return [reacfrac - exp(-time*exp(ko-E/T))]

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_gamma: List of symbols for quantity gamma;
# symbols_z: List of symbols for quantity z;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of symbols for parameters written in LaTex;
# units_y: List of units of measurement for independent quantities;
# units_x: List of units of measurement of dependent quantities;
# units_param: List of units of measurement of the parameters;
# folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(model, symbols_z=['frac', 'time', 'temperature'], symbols_uz=['ufrac', 'utime', 'utemperature'],
                            symbols_param=['ko','E'], folder='resultadoimplicito')

#%% Setting the observed data set
#Data entry using  .xlsx
Estime.setData(data="data_example2-2")

#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
Estime.optimize(initial_estimative=[18,20000]+Estime.z.observed.lista_estimativa,
                lower_bound=[0,10000]+[0]*41+[0]*41+[500]*41,
                upper_bound=[100,30000]+[1]*41+[200]*41+[700]*41)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
Estime.uncertainty(objectiveFunctionMapping=False)

#%%Running the charts without prediction.
# using solely default options
Estime.plots()

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.