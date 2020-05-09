# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo, Regiane
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA

EXEMPLO (4.12) - Retirado do livro Schwaab e Pinto (2007, p. 230) que trata sobre transferencia de calor:

SUMÁRIO:

I - INCLUSÃO DAS BIBLIOTECAS
II - INICIALIZAÇÃO DA CLASSE
III - INCLUSÃO DE DADOS
IV - OTIMIZAÇÃO
V - INCERTEZA E ANALISE DE RESIDUOS
VI - GERAÇÃO DE GRAFICOS E RELATÓRIOS

"""

# =================================================================================
# PARTE I - INCLUSÃO DAS BIBLIOTECAS
# ==============================================================================

u""" 
Abaixo estão representadas as bibliotecas necessárias para a execução:

"""

from numpy import array
from MT_PEU_Linear import EstimacaoLinear

# =================================================================================
# II - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""

Nessa etapa é inicializada a classe que realiza a estimação. 
Aqui é possível renomear a pasta onde são apresentados os resultados.  

"""
ER = EstimacaoLinear(['q'],['x'],['k'],folder='LINEAR')

# =================================================================================
# III - INCLUSÃO DE DADOS
# =================================================================================

x = [10, 20, 30, 40]
ux = [1, 1, 1, 1]

q = [1050, 2000, 2950, 4000]
uq= [1, 1, 1, 1]

ER.setDados(0,(x,ux))
ER.setDados(1,(q,uq))
#
ER.setConjunto()

# =================================================================================
# IV - OTIMIZAÇÃO
# =================================================================================

u"""
Método para obtenção da estimativa dos parâmetros e sua matriz de covariância.

"""

ER.optimize()

# =================================================================================
# V - INCERTEZA E ANALISE DE RESIDUOS
# =================================================================================
u"""

 Na estimação linear, o metodo disponivel para avaliar a incerteza é o MonteCarlo.
 As analises são feitas, utilizando dados de predição.
 Em analise de residuos é possível vericar possíveis relações de dependencia e/ou tendencia entre as variaveis. 

"""
ER.parametersUncertainty(objectiveFunctionMapping='MonteCarlo')
ER.prediction()
ER.residualAnalysis()

# =================================================================================
# VI - GERAÇÃO DE GRAFICOS E RELATÓRIOS
# =================================================================================

u"""

 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos

"""

ER.plots()
ER.reports()



u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 


"""