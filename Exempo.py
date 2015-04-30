# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

from Funcao_Objetivo import WLS
from Modelo import Modelo
from MT_PEU import EstimacaoNaoLinear
from MT_PEU_Linear import EstimacaoLinear
from numpy import ones, array, transpose, concatenate

##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS NÃO LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE I - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

# ---------------------------------------------------------------------------------
# Exemplo validação: Exemplo resolvido 5.11, 5.12, 5.13 (capítulo 5) (Análise de Dados experimentais I)
# ---------------------------------------------------------------------------------

#Tempo
x1 = transpose(array([120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,\
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,\
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,\
60.1,60.0,30.0,30.0,60.0],ndmin=2))

#Temperatura
x2 = transpose(array([600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,\
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,\
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,\
639.0,639.0,639.0,639.0,639.0,639.0,639.0],ndmin=2))

x = concatenate((x1,x2),axis=1)

ux = ones((41,2))

y = transpose(array([0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,\
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,\
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,\
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449],ndmin=2))

uy = ones((41,1))

tipo = 3

Estime = EstimacaoNaoLinear(WLS,Modelo,nomes_x = ['variavel teste x1','variavel teste 2'],simbolos_x=[r't','T'],label_latex_x=[r'$t$','$T$'],nomes_y=['y1'],simbolos_y=[r'y1'],nomes_param=['theta'+str(i) for i in xrange(2)],simbolos_param=[r'theta%d'%i for i in xrange(2)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(2)],projeto='EX_%d'%tipo)

sup=[100,30000]
inf=[0,20000]

# ---------------------------------------------------------------------------------
# Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
# ---------------------------------------------------------------------------------

# x1 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
# y1 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
# x2 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
# y2 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
#
# ux1 = ones((10,1))
# ux2 = ones((10,1))
# uy1 = ones((10,1))
# uy2 = ones((10,1))
#
# x  = concatenate((x1,x2),axis=1)
# y  = concatenate((y1,y2),axis=1)
# ux = concatenate((ux1,ux2),axis=1)
# uy = concatenate((uy1,uy2),axis=1)
#
#
# Estime = EstimacaoNaoLinear(WLS,Modelo,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=[r'theta%d'%i for i in xrange(4)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(4)])

# sup = [6.  ,.3  ,8.  ,0.7]
# inf = [1.  , 0  ,1.  ,0.]

#tipo = None

# =================================================================================
# PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

Estime.gerarEntradas(x,y,ux,uy,tipo='experimental')
grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário

# Otimização
Estime.otimiza(sup=sup,inf=inf,args=[tipo],algoritmo='PSO',itmax=500,Num_particulas=40,metodo={'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear'})
Estime.incertezaParametros(.95,1e-5,metodo='2InvHessiana')
grandeza = Estime._armazenarDicionario()

Estime.Predicao()
Estime.analiseResiduos()

etapas = ['regiaoAbrangencia', 'entrada', 'predicao','grandezas','estimacao']
Estime.graficos(etapas,0.95)


##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS NÃO LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE I - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

ER = EstimacaoLinear(['y'],['x'],['p1','p2'],projeto='LINEAR')
x = array([[0],[1],[2],[3],[4],[5]])
y = array([[.1],[.9],[2.2],[3.2],[3.9],[4.8]])
ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='experimental')
ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='validacao')

# =================================================================================
# PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

ER.otimiza()
ER.incertezaParametros(.95)
ER.Predicao(delta=1e-6)
ER.analiseResiduos()
ER.graficos(['regiaoAbrangencia', 'entrada', 'predicao','grandezas','estimacao'],0.95)