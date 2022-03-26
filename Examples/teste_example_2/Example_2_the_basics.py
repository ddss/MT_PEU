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
Estime = EstimacaoNaoLinear(Model, symbols_x=['Time','Temperature'],symbols_ux=['UxTime','Uxtemperature'], units_x=['s','K'],
                            symbols_y=[r'Y'],symbols_uy=['uY'], units_y=['adm'],
                            symbols_param=['ko','E'], units_param=['adm','K'], Folder='Example2')

#%% Setting the observed data set
#Data entry using standard excel template
#Estime.setDados(data="data_exa1.xlsx")
#Data entry using .CSV
Estime.setDados(data=["data_exa1_independent.csv","data_exa1_dependent.csv"])
#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
Estime.optimize(initial_estimative=[18, 20000.000],optimizationReport=False, algorithm='ipopt')

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
Estime.parametersUncertainty(uncertaintyMethod='2InvHessiana',objectiveFunctionMapping=True)

#%%Running the charts without prediction.
# using solely default options
Estime.plots()

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
Estime.prediction(export_y=True, export_y_xls=True, export_cov_y=True, )

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.