#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from casadi import vertcat

#%% Model definition
def Model(param,x,*args):

    a1, b1, a2, b2 = param[0], param[1], param[2], param[3]
    x1, x2 = x[:,0], x[:,1]

    return vertcat(a1*x1/(1+b1*x1), a2*(x2**b2))

#%% Starting the MT_PEU main object
Estime = EstimacaoNaoLinear(Model,symbols_x=['x1','x2'],symbols_y=['y1','y2'],symbols_param=['alpha1','alpha2', 'beta1', 'beta2'],
                          label_latex_param=[r'$\alpha_1$',r'$\alpha_2$',r'$\beta_1$',r'$\beta_2$'],units_y=['kg','kg'],Folder='Example 4')


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

# inputs
Estime.setDados(0,(x1,ux1),(x2,ux2))
# outputs
Estime.setDados(1, (y1, uy1), (y2, uy2))

# Defining the previous data set to be used to parameter estimation
Estime.setConjunto(type='estimacao')

#%% Optimization - estimating the parameters
Estime.optimize(initial_estimative=[3,0.1,5,0.4])

#%% Evaluating the parameters uncertainty and coverage region
Estime.parametersUncertainty(uncertaintyMethod='2InvHessiana')

#%% Evaluating model predictions
Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()