# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""
 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from MT_PEU_Linear import EstimacaoLinear


##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE I - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================
#
# # #Sem o cálculo do termo independente
ER = EstimacaoLinear(['y'],['x'],['p1'],projeto='LINEARsemB')
x = [0,1,2,3,4,5]
ux = [1,1,1,1,1,1]

y = [.1,.9,2.2,3.2,3.9,4.8]
uy = [1,1,1,1,1,1]

ER.setDados(0,(x,ux))
ER.setDados(1,(y,uy))

ER.setConjunto()

# Com o cálculo do termo independente
ER = EstimacaoLinear(['y'],['x'],['p1','p2'],projeto='LINEARcomB1')
x = [0,1,2,3,4,5]
ux = [1,1,1,1,1,1]
#
y = [.1,.9,2.2,3.2,3.9,4.8]
uy = [1,1,1,1,1,1]
#
ER.setDados(0,(x,ux))
ER.setDados(1,(y,uy))
#
ER.setConjunto()

# =================================================================================
# PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

#ER.otimiza()
#ER.SETparametro([0.9571428571428567, 0.12380952380952503])
#ER.incertezaParametros(iteracoes=1000,metodoPreenchimento='MonteCarlo')
#ER.predicao(delta=1e-6)
#ER.analiseResiduos()
#ER.graficos(['analiseResiduos','regiaoAbrangencia', 'grandezas-entrada', 'predicao','grandezas-calculadas','otimizacao'])
#ER.relatorio()
