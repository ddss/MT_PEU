# -*- coding: utf-8 -*-
"""
Exemplo de uso do MT_PEU

EXERCÍCIO SUGERIDO (2) - Retirado do livro Schwaab e Pinto (2007, p. 375) que trata sobre a estimação de parâmetros.
A resolução pode ser encontrada no cap. 6 do mesmo livro, p. 428.

"""
# =================================================================================
# PARTE I - INCLUSÃO DAS BIBLIOTECAS
# =================================================================================

""" 
Abaixo estão representadas as bibliotecas necessárias para o correto funcionamento do programa:

"""

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import concatenate

##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS NÃO LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE II - CRIAÇÃO DO MODELO
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

    y1 = concatenate((y1,y2),axis=1)

    return y1

# =================================================================================
# PARTE III - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""
Inicialização da classe que realiza estimação. Entradas opcionais como unidades, podem ser passadas nesta etapa.
Também é possível renomear a pasta onde são gerados os aquivos com os resultados.como no exemplo a baixo, onde o nome
da pasta foi alterado para 'projeto'.

"""

Estime = EstimacaoNaoLinear(Modelo1,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=[r'a%d'%i for i in xrange(4)],
                          label_latex_param=[r'$\alpha_{%d}$'%i for i in xrange(4)],unidades_y=['kg','kg'],projeto='projeto')

# =================================================================================
# PARTE IV - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
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
A plataforma MT_PEU necessita da incerteza dos dados experimentais, e quando está informação não é disponibilizada,
o programa assume valor 1 para todos os dados (ux1, ux2, uy1):

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
# PARTE V - OTIMIZAÇÃO
# =================================================================================

u"""
Otimização será realizada utilizando o algoritmo escolhido pelo usuário (disponiveis: Nelder-Mead, Powell, BFGS, L-BFGS-B, CG),
caso não seja escolhido o algoritmo a ser utilizado, por padrão a otimização será realizada utilizando o Nelder-Mead,
com estimativa inicial em [3 , 0.1, 5, 0,4], para alpha1, beta1, alpha2, beta2 respectivamente.

"""

# Otimização
Estime.otimiza(estimativa_inicial=[3, 0.1, 5,0.4],algoritmo='Nelder-Mead')

# =================================================================================
# PARTE VI - INCERTEZA
# =================================================================================

u"""
 Associada a toda medida existe uma incerteza. O método incerteza parâmetros calcula as incertezas 
 associadas aos parâmetros (neste exemplo k0 e E). Nesta etapa é possível escolher o método útilizado para
 avaliar a incerteza. Métodos disponíveis: 2InvHessiana, Geral, SensibilidadeModelo. Por definição o preenchimeto
 da região de verossimilhança é 'True', caso necessário esta opção pode ser alterada nesta etapa.

"""

Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=False)
# =================================================================================
# PARTE VII - PREDIÇÃO E ANALISE DE RESIDUOS
# =================================================================================

u"""
 No método predição, é feita a primeira analise sobre os resultados obtidos. A covariância é avaliada, 
 e consequentemente a eficiencia do modelo. Em analise de residuos é possível vericar possíveis relações de dependencia 
 e/ou tendencia entre as variaveis. Testes estatisticos como o de homocedasticidade, chi quadrado, etc são realizados
 nesta etapa. A analise de residuos é feita prioritariamente com os dados de validação.

"""

Estime.predicao()
Estime.analiseResiduos()

# =================================================================================
# PARTE VIII - GRÁFICOS E RELATÓRIO
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

"""