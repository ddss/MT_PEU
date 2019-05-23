# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo, Regiane
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# =================================================================================
# PARTE I - INCLUSÃO DAS BIBLIOTECAS
# ==============================================================================

""" 
Abaixo estão representadas as bibliotecas necessárias para a execução:

"""

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from numpy import array
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

ER.otimiza()
ER.incertezaParametros(metodoPreenchimento='MonteCarlo')
ER.predicao(delta=1e-6)
ER.analiseResiduos()
ER.graficos(['analiseResiduos','regiaoAbrangencia', 'grandezas-entrada', 'predicao','grandezas-calculadas','otimizacao'])
ER.relatorio()
