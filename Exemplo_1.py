# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo, Regiane.
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA


Exemplo básico de uso do MT_PEU, onde são apresentadas as configurações mínimas para o funcionamento do software.

EXEMPLO (5.11) - Retirado do livro Schwaab e Pinto (2007, p. 361) que trata sobre a estimação de parâmetros do modelo cinético:

SUMÁRIO:

I - INCLUSÃO DAS BIBLIOTECAS
II - CRIAÇÃO DO MODELO
III - INICIALIZAÇÃO DA CLASSE
IV - INCLUSÃO DE DADOS
V - OTIMIZAÇÃO
VI - INCERTEZA
VII - PREDIÇÃO E ANALISE DE RESIDUOS
VIII - GERAÇÃO DE GRÁFICOS E RELATÓRIO

"""

# =================================================================================
# I - INCLUSÃO DAS BIBLIOTECAS.
# =================================================================================
u""" 
Abaixo estão representadas as bibliotecas necessárias para a execução:

"""
from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import exp

# =================================================================================
# II - CRIAÇÃO DO MODELO.
# =================================================================================

u"""
O modelo é definido na forma de uma subrotina ((def) do python) e representa a equação abaixo,onde: 
y é a fração que resta do reagente, T é a temperatura, e por fim k0 e E são os parâmetros a serem estimados.

"""

def Modelo (param, x, args):

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y = exp(-(ko*10**17)*tempo*exp(-E/T))


    return y


# =================================================================================
# III - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""

Nessa etapa é inicializada a classe que realiza a estimação. Por padrão, informações como os simbolos das variáveis,
são obrigatorias e passadas nesta etapa.  

"""
#Cria o objeto que realiza a estimação

Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], simbolos_y=[r'y'], simbolos_param=['ko','E'])

# =================================================================================
# IV - INCLUSÃO DE DADOS
# =================================================================================

u"""

Os dados experimentais da variável dependente (y) e das variáveis independentes (tempo e Temperatura)
são disponibilizados em Schwaab e Pinto (2007, p.324), e apresentados abaixo na forma de listas:

"""
y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]

temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

u"""

Como entrada obrigatória, a plataforma MT_PEU necessita da incerteza dos dados experimentais (ux1, ux2, uy1).
Neste exemplo, foram adotados o valor 1 para as incertezas.

"""

uy = [1]*41

uxtempo = [1]*41

uxtemperatura = [1]*41

u"""

Inclusão de dados experimentais na estimação:
Inclui os dados experimentais nesse objeto (setDados), onde a opção 0 é para a grandeza dependente,
e a opção 1 é para a grandeza independente.

"""

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))

Estime.setDados(1,(y,uy))


u"""

Define que os dados experimentais previamente inseridos serão utilizados como um conjunto de dados para o qual os 
parâmetros serão estimados:

"""

Estime.setConjunto(tipo='estimacao')

# =================================================================================
# V - OTIMIZAÇÃO
# =================================================================================

u"""

A otimização será realizada utilizando o algoritmo default (Nelder-Mead), se faz necessário informar a estimativa inicial.

"""
Estime.otimiza(estimativa_inicial= [0.5, 25000.000])

# =================================================================================
# VI - INCERTEZA
# =================================================================================

u"""
 "Associada a toda medida existe uma incerteza." 
 Este método calcula as incertezas associadas aos parâmetros (neste exemplo k0 e E). 
 
"""


# =================================================================================
# VII - PREDIÇÃO E ANALISE DE RESIDUOS
# =================================================================================

u"""
 No método predição, é feita a avaliação da grandeza dependente com base nos parametros fornecidos. 
 A covariância é avaliada, e consequentemente a eficiencia do modelo. 
 Em analise de residuos é possível vericar possíveis relações de dependencia e/ou tendencia entre as variaveis. 
 Testes estatisticos como o de homocedasticidade, chi quadrado, dentre outros são realizados
 nesta etapa. A analise de residuos é feita prioritariamente com os dados de validação.

"""

Estime.predicao()
Estime.analiseResiduos()

# =================================================================================
# VIII - GERAÇÃO DE GRÁFICOS E RELATÓRIO
# =================================================================================

u"""
 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos. 
 Os gráficos são gerados de acordo com as etapas que foram realizadas. No relátorio contém informações a respeito
 dos testes estatisticos, função objetivo, matriz de covariância, status da otimização, dentre outros.

"""
Estime.relatorio()
etapas = ['otimizacao', 'grandezas-entrada', 'predicao', 'grandezas-calculadas', 'analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)

#Estime.incertezaParametros()
Estime.predicao()
Estime.graficos(etapas)


u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 


"""