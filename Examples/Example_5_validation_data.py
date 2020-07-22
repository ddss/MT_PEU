#%% Packages importing
from MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
def Model(param, x, *args):
    ko, E = param[0], param[1]
    time, T = x[:,0], x[:,1]

    return exp(-ko * time * exp(-E * (1 / T - 1. / 630.)))

#%% Starting the MT_PEU main object
Estime = EstimacaoNaoLinear(Model, symbols_x=[r't', 'Tau'], units_x=['s', 'K'], label_latex_x=[r'$t$', '$T$'],
                            symbols_y=[r'y'], units_y=['adm'],
                            symbols_param=['ko', 'E'], units_param=['adm','K'],
                            label_latex_param=[r'$k_o$', r'$E$'],
                            Folder='KineticModelEX5')

#%% Defining observed data
# Input data
# input 1
time = [120.0, 60.0, 120.0, 60.0, 30.0, 15.0, 45.1, 90.0, 150.0, 60.0, 60.0, 30.0, 150.0, 90.4, 120.0, 60.0, 60.0,
         60.0, 30.0,
         45.1, 30.0, 45.0, 15.0, 90.0, 25.0, 60.1, 60.0, 30.0, 60.0]
# input 1 uncertainty
uxtime = [0.001] * 29

# input 2
temperature = [600.0, 612.0, 612.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0,
               620.0, 620.0,
               620.0, 631.0, 631.0, 631.0, 631.0, 639.0, 639.0, 639.0, 639.0, 639.0, 639.0, 639.0]
# input 2 uncertainty
uxtemperature = [0.05] * 29

# Output data
y = [0.9, 0.886, 0.791, 0.787, 0.877, 0.938, 0.827, 0.696, 0.582, 0.795, 0.790, 0.883, 0.576, 0.715, 0.673, 0.802,
     0.804, 0.804, 0.764,
     0.688, 0.802, 0.695, 0.808, 0.309, 0.689, 0.437, 0.425, 0.659, 0.449]
# output uncertainty
uy1 = [1] * 29

#%% Setting the observed data set

# inputs
Estime.setDados(0, (time, uxtime), (temperature, uxtemperature))
# output
Estime.setDados(1, (y, uy1))

# Defining the previous data set to be used to parameter estimation
Estime.setConjunto(dataType='estimacao')

#%% Optimization - estimating the parameters
Estime.optimize(initial_estimative=[0.005, 20000.000])

#%% Evaluating the parameters uncertainty and coverage region
Estime.parametersUncertainty(uncertaintyMethod='Geral',objectiveFunctionMapping=False)

#%% Evaluating model predictions
Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()

# =================================================================================
# IX - OPTIONAL: VALIDATION
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
Estime.setDados(0,(time,uxtime),(temperature,uxtemperature))
# outputs
Estime.setDados(1,(y,uy1))

# Defining the previous data set to be used to validation
Estime.setConjunto(dataType='predicao')

#%% Evaluating model predictions for the validation data
Estime.prediction()

#%% Evaluating residuals and quality index
Estime.residualAnalysis()

#%% Plotting the main results
Estime.plots()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.

