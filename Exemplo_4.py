# -*- coding: utf-8 -*-
"""
Exemplo de uso do MT_PEU

EXERCÍCIO SUGERIDO (2) - Retirado do livro Schwaab e Pinto (2007, p. 375) que trata sobre a estimação de parâmetros.
A resolução pode ser encontrada no cap. 6 do mesmo livro, p. 428.

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

U""" 
Abaixo estão representadas as bibliotecas necessárias para o correto funcionamento do programa:

"""

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import concatenate

# =================================================================================
# II - CRIAÇÃO DO MODELO
# =================================================================================

u"""
O modelo é definido na forma de uma subrotina ((def) do python). Neste exemplo serão comparados dois modelos,
 y1 e y2, para que possa ser feita a analise de qual é o melhor modelo. Alpha1, beta1, alpha2, beta2 
são os parâmetros a serem estimados.

"""

def Modelo1(param,x,args):
    x1 = x[:,0:1]
    x2 = x[:,1:]

    alpha1 = param[0]
    beta1  = param[1]
    alpha2 = param[2]
    beta2  = param[3]

    y1 = alpha1*x1/(1+beta1*x1)
    y2 = alpha2*(x2**beta2)

    y = concatenate((y1,y2),axis=1)

    return y

# =================================================================================
# III - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""
Nessa etapa é inicializada a classe que realiza a estimação. Por padrão, informações como os simbolos das variáveis,
são obrigatorias e passadas nesta etapa.  

"""

Estime = EstimacaoNaoLinear(Modelo1,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=['alpha1','alpha2', 'beta1', 'beta2'],
                          label_latex_param=[r'$\alpha_1$',r'$\alpha_2$',r'$\beta_1$',r'$\beta_2$'],unidades_y=['kg','kg'],projeto='projeto')

# =================================================================================
# IV - INCLUSÃO DE DADOS
# =================================================================================

u"""
Os dados experimentais da variável dependente (y1, y2) e das variável independentes (x1, x2)
são disponibilizados em Schwaab e Pinto (2007, p.375), e apresentados abaixo na forma de listas:

"""

x1 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y1 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]
x2 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y2 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]

u"""
Como entrada obrigatória, a plataforma MT_PEU necessita da incerteza dos dados experimentais (ux1, ux2, uy1).
Neste exemplo, foram adotados o valor 1 para as incertezas.

"""

ux1 = [1]*10
ux2 = [1]*10
uy1 = [1]*10
uy2 = [1]*10

u"""
Inclusão de dados experimentais na estimação:
Inclui os dados experimentais nesse objeto (setDados), onde a opção 0 é para a grandeza dependente,
e a opção 1 é para a grandeza independente.

"""

Estime.setDados(0,(x1,ux1),(x2,ux2))
Estime.setDados(1, (y1, uy1), (y2, uy2))

u"""
Define que os dados experimentais previamente inseridos serão utilizados como um conjunto de dados para o qual os 
parâmetros serão estimados:

"""

Estime.setConjunto(tipo='estimacao')

# =================================================================================
# V - OTIMIZAÇÃO
# =================================================================================

u"""
A otimização será realizada utilizando o algoritmo default (BFGA), se faz necessário informar a estimativa inicial.

"""

# Otimização
Estime.otimiza(estimativa_inicial=[3, 0.1, 5,0.4],algoritmo='BFGS')

# =================================================================================
# VI - INCERTEZA
# =================================================================================

u"""
 Metodo que calcula as incertezas dos parametros de alpha1, alpha2, beta1, beta2. 
 Nesta etapa é possível escolher o método útilizado para avaliar a incerteza. 
 Métodos disponíveis: 2InvHessiana, Geral, SensibilidadeModelo. 
 Por definição o preenchimeto da região de verossimilhança é 'True', caso necessário esta opção pode ser alterada.

"""

Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=True)
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

etapas = ['grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)
Estime.relatorio()

u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 

"""