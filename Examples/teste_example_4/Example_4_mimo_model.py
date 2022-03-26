#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from casadi import vertcat

#%% Model definition
# def Model: The def model specifies the equations with their respective parameters.
def Model(param,x,*args):

    a1, b1, a2, b2 = param[0], param[1], param[2], param[3]
    x1, x2 = x[:,0], x[:,1]

    return vertcat(a1*x1/(1+b1*x1), a2*(x2**b2))

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: Lis of symbols for quantity x;
# symbols_y: List of symbols for quantity y;
# symbols_param: List of symbols for the parameters to be estimated;
# label_latex_param: List of ymbols for parameters written in LaTex;
# units_y: List of units of measurement for independent quantities;
# Folder: Defines the name of the folder where the results will be saved.
Estime = EstimacaoNaoLinear(Model,symbols_x=['x1','x2'],symbols_ux=['ux1','ux2'],symbols_y=['y1','y2'],symbols_uy=['uy1','uy2'],symbols_param=['alpha1','alpha2', 'beta1', 'beta2'],
                          label_latex_param=[r'$\alpha_1$',r'$\alpha_2$',r'$\beta_1$',r'$\beta_2$'],units_y=['kg','kg'],Folder='Example4')

#%% Defining observed data
# Input data
# input 1
x1 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
# input 1 uncertainty
ux1 = [1]*10
# input 2
x2 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
# input 2 ucertainty
ux2 = [1]*10

# Output data
# output 1
y1 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]
# output 1 uncertainty
uy1 = [1]*10
# output 2
y2 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]
# output 2 uncertainty
uy2 = [1]*10

#%% Setting the observed data set
#Estime.setDados(data={0:[(x1, ux1),(x2, ux2)],1:[(y1,uy1),(y2,uy2)]},glx=[], gly=[])
#Estime.setDados(data="data_exa4.xlsx",glx=[], gly=[])
Estime.setDados(data=["data_exa4_independent","data_exa4_dependent"])

# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;


#%% Optimization - estimating the parameters
# initial_estimative: List with the initial estimates for the parameters;
# lower_bound: List with the lower bounds for the parameters;
# upper_bound: List with the upper bounds for the parameters;
# algorithm: Informs the optimization algorithm that will be used. Each algorithm has its own keywords;
# optimizationReport: Informs whether the optimization report should be created (True or False);
# parametersReport: Informs whether the parameters report should be created (True or False).
Estime.optimize(initial_estimative=[3,0.1,5,0.4], algorithm='ipopt', lower_bound=[0.2,0.09,3.1,0.3], upper_bound=[3.6,0.3,5.6,0.6],
                optimizationReport = True, parametersReport = False)

#%% Evaluating the parameters uncertainty and coverage region
# uncertaintyMethod: method for calculating the covariance matrix of the parameters;
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# lower_bound: Lower limit of parameters;
# upper_bound: Upper limit of the parameters.
# parametersReport: Informs whether the parameters report should be created.
Estime.parametersUncertainty(uncertaintyMethod='2InvHessiana', objectiveFunctionMapping=True, lower_bound=[1,0.04,1.75,0.175], upper_bound=[4.5,0.16,6.75,1],
                             parametersReport = True)

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
# using solely default options
Estime.plots()
Estime.reports()