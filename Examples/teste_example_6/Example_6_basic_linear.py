#%% Packages importing
from MT_PEU_Linear import EstimacaoLinear

#%% Initialization of the class that performs the estimation.
ER = EstimacaoLinear(['y'],['uy'],['x'],['ux'],['p1','p2'],folder='Exemple6')

#%% Setting the observed data set to using Manual data entry


# input 1
x = [0,1,2,3,4,5]
# input 1 uncertainty
ux = [1,1,1,1,1,1]
# Output data
# output 1
y = [.1,.9,2.2,3.2,3.9,4.8]
# output 1 uncertainty
uy = [1,1,1,1,1,1]
#Manual data entry

#ER.setDados(data={'x':x,'ux':ux,'y':y ,'uy':uy},glx=[],gly=[])

#%% Setting the observed data set
# inputs
#ER.setDados(data="data_exa6.xlsx")

ER.setDados(data=["data_exa6_independent.csv","data_exa6_dependent.csv"])

# Defining the previous data set to be used to parameter estimation
# dataType: Defines the purpose of the informed data set: estimacao, predicao.
# glx: Degrees of freedom of quantity x;
# gly: Degrees of freedom of quantity y;

#%% Optimization - estimating the parameters
# parametersReport: Informs whether the parameters report should be created (True or False).
ER.optimize(parametersReport=False)

#%% Evaluating the parameters uncertainty and coverage region
# objectiveFunctionMapping: Deals with mapping the objective function (True or False);
# parametersReport: Informs whether the parameters report should be created.
ER.parametersUncertainty(objectiveFunctionMapping=True, parametersReport=True)

#%% Evaluating model predictions
# export_y: Exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation (True or False);
# export_y_xls: Exports the calculated data of y, its uncertainty, and degrees of freedom in a xls (True or False);
# export_cov_y: Exports the covariance matrix of y (True or False);
# export_x: Exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation(True or False);
# export_cov_x: Exports the covariance matrix of x (True or False).
ER.prediction(export_y=True,export_y_xls=True, export_cov_y=True, export_x=True, export_cov_x=True)

#%% Evaluating residuals and quality index
ER.residualAnalysis(report=True)

#%% Plotting the main results
# using solely default options
ER.plots()
ER.reports()

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.
# Avaliação de dados de medição — Guia para a expressão de incerteza de medição http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf


