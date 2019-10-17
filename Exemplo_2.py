# -*- coding: utf-8 -*-
"""
Exemplo de uso do MT_PEU

EXEMPLO (5.12-1) - Retirado do livro Schwaab e Pinto (2007, p. 364) que trata sobre a estimação de parâmetros do modelo cinético:

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
# I - INCLUSÃO DAS BIBLIOTECAS
# =================================================================================
u""" 
Abaixo estão representadas as bibliotecas necessárias para o correto funcionamento do programa:

"""

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import exp

# =================================================================================
# II - CRIAÇÃO DO MODELO
# =================================================================================

u"""
Similar ao Exemplo_1, k0 e E são os parametros a serem estimados, diferindo apenas a equação do modelo. 

"""

def Modelo (param, x, args):

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y = exp(-tempo*exp(ko-E/T))

    return y

# =================================================================================
# III - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""
Neste exemplo apresentamos alumas possíveis entradas opcionais, como unidades, assim como a opção de renomear a pasta 
onde são gerados os aquivos com os resultados.
No exemplo a baixo, onde o nome da pasta foi alterado para 'Teste1'.

"""

Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], unidades_x=['s','K'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko','E'], unidades_param=['adm','K'], projeto='Teste1')

# =================================================================================
# IV - INCLUSÃO DE DADOS
# =================================================================================

u"""
Os dados experimentais da variável dependente (y) e das variáveis independentes (t e T)
são disponibilizados em Schwaab e Pinto (2007, p.326), e apresentados abaixo na forma de listas:

"""

#Tempo
tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]

#Temperatura
temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

u"""
Como entrada obrigatória, a plataforma MT_PEU necessita da incerteza dos dados experimentais (ux1, ux2, uy1).
Neste exemplo, foram adotados o valor 1 para as incertezas.

"""

uxtempo = [1]*41
uxtemperatura = [1]*41
uy = [1]*41

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

Estime.setConjunto()

# =================================================================================
# PARTE V - OTIMIZAÇÃO
# =================================================================================

u"""
Neste exemplo, o usuário tem a opção de escolha do algoritmo a ser utilizado na otimização. 
Disponiveis: 'Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'CG'. 
Caso opte por não escolher, será utilizado o algoritmo default: Nelder-Mead, com estimativa inicial em [0.03 , 20000.00].

"""

Estime.otimiza(estimativa_inicial= [0.03, 20000.000],algoritmo='Nelder-Mead')
#Estime.SETparametro([3.900e+01, 2.764e+04])
# =================================================================================
# VI - INCERTEZA
# =================================================================================

u"""
 Neste exemplo é possível escolher o método útilizado para avaliar a incerteza. 
 Métodos disponíveis: 2InvHessiana, Geral, SensibilidadeModelo. 
 Por definição o preenchimeto da região de verossimilhança é 'True', caso necessário esta opção pode ser alterada.

"""

Estime.incertezaParametros(delta=1e-5,metodoIncerteza='2InvHessiana',preencherregiao=True)

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
# VIII - GRÁFICOS E RELATÓRIO
# =================================================================================

u"""
 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos. 
 Os gráficos são gerados de acordo com as etapas que foram realizadas. No relátorio contém informações a respeito
 dos testes estatisticos, função objetivo, matriz de covariância, status da otimização, dentre outros.

"""

etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)
Estime.relatorio()


u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 


"""