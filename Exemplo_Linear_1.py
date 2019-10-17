# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo, Regiane
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA

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

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from numpy import array
from MT_PEU_Linear import EstimacaoLinear

# =================================================================================
# II - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""

Nessa etapa é inicializada a classe que realiza a estimação. 
Também é possível renomear a pasta onde são apresentados os resultados.  

"""

# Com o cálculo do termo independente

ER = EstimacaoLinear(['y'],['x'],['p1','p2'],projeto='LINEARcomB1')

# =================================================================================
# III - INCLUSÃO DE DADOS
# =================================================================================

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
# IV - OTIMIZAÇÃO
# =================================================================================

u"""
Método para obtenção da estimativa dos parâmetros e sua matriz de covariância.

"""

ER.otimiza(parametersReport=False)

# =================================================================================
# V - INCERTEZA E ANALISE DE RESIDUOS
# =================================================================================

u"""

 Na estimação linear, o metodo disponivel para avaliar a incerteza é o MonteCarlo.
 As analises são feitas, utilizando dados de predição.
 Em analise de residuos é possível vericar possíveis relações de dependencia e/ou tendencia entre as variaveis. 

"""

ER.incertezaParametros(metodoPreenchimento='MonteCarlo')
ER.predicao()
ER.analiseResiduos()

# =================================================================================
# VI - GERAÇÃO DE GRAFICOS E RELATÓRIOS
# =================================================================================

u"""

 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos

"""

ER.graficos(['analiseResiduos','regiaoAbrangencia', 'grandezas-entrada', 'predicao','grandezas-calculadas','otimizacao'])
#ER.relatorio()



u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 


"""