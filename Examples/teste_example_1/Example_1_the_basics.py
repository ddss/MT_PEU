#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp,array
import pandas as pd #importação da biblioteca pandas para possibilitar trabalhar com o xlsx

#%% Model definition
# def Model: The subroutine that specifies the equations with their respective parameters.
def Model(param, x, *args):

    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-(ko*10**17)*time*exp(-E/T))

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: list of symbols for quantity x;
# symbols_y: list of symbols for quantity y;
# symbols_param: list of symbols for the parameters to be estimated;
# Folder: string with the name of the folder where reports and charts will be saved;
Estime = EstimacaoNaoLinear(Model, symbols_x=['t','Tao'], symbols_y=['y'], symbols_param=['ko','E'], Folder='Example1')
#%% Importando dados do arquivo xlsx,quando o arquivo se encontra na pasta do exemplo.py passa apenas o nome do arquivo
#quando não estiver na mesma pasta tem que usar o diretório.

data_independent_variable = pd.read_excel("data_exa1.xlsx", sheet_name="independent variable")
data_dependent_variable = pd.read_excel("data_exa1.xlsx", sheet_name="dependent variable")

#%% Defining observed data
# Observed data of independent variable
X  =data_independent_variable[{data_independent_variable.columns[i] for i in range(0,data_independent_variable.shape[1],2)}].to_numpy()
uX =data_independent_variable[{data_independent_variable.columns[i] for i in range(1,data_independent_variable.shape[1],2)}].to_numpy()
# Observed data of dependent variable
Y =data_dependent_variable[{data_dependent_variable.columns[i]for i in range(0,data_dependent_variable.shape[1],2)}].to_numpy()
uY = data_dependent_variable[{data_dependent_variable.columns[i]for i in range(1,data_dependent_variable.shape[1],2)}].to_numpy()

Estime.setDados(X,uX,Y,uY)







# Defining the previous data set to be used to parameter estimation
Estime.setConjunto()

#%% Optimization - estimating the parameters
# initial_estimate: list containing initial estimate for optimization algorithm
Estime.optimize(initial_estimative=[0.5,25000])

#%% Evaluating the parameters uncertainty and coverage region
# using solely default options
Estime.parametersUncertainty()

#%% Evaluating model predictions
# using solely default options
Estime.prediction()

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
# using solely default options
#Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros.
# Rio de Janeiro: e-papers, 2007.
#%%