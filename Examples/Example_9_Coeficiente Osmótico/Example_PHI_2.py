#%% Packages importing
from modules.MT_PEU import EstimacaoNaoLinear
from casadi import exp, log
#%% Model definition
# def Model: The subroutine that specifies the equations with their respective parameters.
def Model(param, x, args):
    T = x[:, 0]
    M = x[:, 1]
    B01,B02,B03,B11,B12,B13,C1,C2,C3 = param[0], param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]
    PHI = 1+ (1*1*(-(-61.44534*exp((T-273.15)/273.15)+2.864468*exp(2*((T-273.15)/273.15))+183.5379*log(T/273.15)-0.6820223*(T-273.15)+7.875695e-4*((T)**2-273.15**2)+58.95788*(273.15/T))*(((1*1/2.)*M*2)**0.5)/(1+1.2*((1*1/2)*M*2)**0.5))+M*2*1*1/2*((B01+1e-4*B02*(T-365.40)+1e-6*B03*(T-365.40)**2)+(B11+1e-4*B12*(T-365.40)+1e-6*B13*((T-365.40)**2))*(exp(-1*((1*1/2.)*M*2)**0.5)))+(M**2*((2*(1*1)**1.5)/2.)*(C1+1e-5*C2*(T-365.40)+1e-7*C3*(T-365.40)**2)))
    return PHI
#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_x: list of symbols for quantity x;
# symbols_ux: list of symbols for uncertainty x;
# symbols_y: list of symbols for quantity y;
# symbols_uy: list of symbols for uncertainty y;
# symbols_param: list of symbols for the parameters to be estimated;
# Folder: string with the name of the folder where reports and charts will be saved;
Estimation = EstimacaoNaoLinear(Model, symbols_x=['T','M'],symbols_ux=['uT','uM'] ,symbols_y=['PHI'],symbols_uy=['uPHI'],
                                symbols_param=['B01','B02','B03','B11','B12','B13','C1','C2','C3'], Folder='CoefOsmotico' )
#%% Data entry using .CSV
Estimation.setDados(data='Dados',separador= ';',decimal=',')

#%% Optimization - estimating the parameters
# initial_estimate: list containing initial estimate for optimization algorithm
Estimation.optimize(initial_estimative = [1,1,1,1,1,1,1,1,1])

#%% Evaluating the parameters uncertainty and coverage region
# using solely default options
Estimation.parametersUncertainty(uncertaintyMethod='Geral', objectiveFunctionMapping=False,iterations=5000,
                                 lower_bound=[0.085,0,-5.5,0.02,-4,2,-6e-3,-8,2],upper_bound=[0.115,4,-2.5,0.12,8,13,-2e-3,-1,7])
#%% Evaluating model predictions
# using solely default options
Estimation.prediction()

#%% Evaluating residuals and quality index
# using solely default options
Estimation.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
#Estimation.plots()

