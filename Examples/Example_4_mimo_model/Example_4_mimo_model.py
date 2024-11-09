#%% Packages importing
from sys import path #para buscar os arquivos em um diretório
path.append("../../modules")#passando o diretório da pasta raíz
from modules.MT_PEU import EstimacaoNaoLinear
from casadi import vertcat

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model(param,y, x,*args):

    a1, b1, a2, b2 = param[0], param[1], param[2], param[3]
    y1, y2, x1, x2 = y[0], y[1], y[2], y[3]

    return [y1 - a1*x1/(1+b1*x1), y2 - a2*(x2**b2)]

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of symbols for parameters written in LaTex;
# units_y: List of units of measurement for independent quantities;
# Folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(Model,symbols_y=['y1','y2','x1','x2'],symbols_uy=['uy1','uy2','ux1','ux2'],symbols_param=['alpha1','alpha2', 'beta1', 'beta2'],
                          label_latex_param=[r'$\alpha_1$',r'$\alpha_2$',r'$\beta_1$',r'$\beta_2$'],Folder='resultadoimplicit')


#%% Setting the observed data set

Estime.setDados(data=["data_exa4_independent.xlsx",
                      "data_exa4_dependent.csv"])

# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: observado, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;


#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# upper_bound: List with the upper bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# report: Informs whether the parameters report should be created (True or False).
Estime.optimize(initial_estimative=[3,0.1,5,0.4]+Estime.y.observado.lista_estimativa,
                algorithm='ipopt',
                lower_bound=[0.2,0.09,3.1,0.3]+[1e-3]*Estime.y.NV*Estime.y.observado.NE,
                upper_bound=[3.6,0.3,5.6,0.6]+[100]*Estime.y.NV*Estime.y.observado.NE,
                optimizationReport = True,
                report= False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# lower_bound: Lower limit of parameters;
# upper_bound: Upper limit of the parameters.
# report: Informs whether the parameters report should be created.
Estime.uncertainty()

#%% Evaluating residuals and quality index
Estime.residualAnalysis()

#%% Plotting the main results
# using solely default options
Estime.plots()
Estime.reports()