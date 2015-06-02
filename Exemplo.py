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

tipo = 1 # tipo: modelo a ser escolhido - 0 (exemplo 5.11), 1 (exemplo 5.12) ou 2 (exemplo 5.13)

Estime = EstimacaoNaoLinear(WLS,Modelo,simbolos_x=[r't','T'],label_latex_x=[r'$t$','$T$'],simbolos_y=[r'y'],simbolos_param=['ko','E'],projeto='EX_%d'%tipo,args=[tipo])

sup=[50,30000]
inf=[0 ,20000]

# ---------------------------------------------------------------------------------
# Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
# ---------------------------------------------------------------------------------
#
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
#
# sup = [6.  ,.3  ,8.  ,0.7]
# inf = [1.  , 0  ,1.  ,0.]
#
# tipo = None

# =================================================================================
# PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

Estime.gerarEntradas(x,y,ux,uy,tipo='experimental')
#print Estime._EstimacaoNaoLinear__etapas
#grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário
#Estime.gerarEntradas(x,y,ux,uy,tipo='experimental')
#Estime.gerarEntradas(x,y,ux,uy,tipo='validacao')

# Otimização
Estime.otimiza(sup=sup,inf=inf,algoritmo='PSO',itmax=100,Num_particulas=40,metodo={'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear'})
#Estime.SETparametro([38.995741810115526, 27642.600636422056],array([[  2.02734350e+03,   1.27340945e+06], [  1.27340945e+06,   8.00043729e+08]]), [[40.109979078710914, 28327.340030375271], [39.974873089082237, 28269.976466622789], [39.986868869853929, 28257.245507319007], [39.957802118600114, 28262.280430186871], [39.96796782099441, 28252.197585436141], [39.972752888990108, 28247.451519119131], [39.970243928961061, 28248.729896415203], [40.213165972226285, 28401.616852926723], [39.966798413346233, 28255.363399182246], [39.967131150063913, 28249.310328451735], [39.968894125215073, 28251.321914853899], [39.968259898251567, 28254.002072298666], [39.845248197231676, 28170.027690073686], [37.137087267928976, 26470.834878622278], [39.967980582134309, 28255.18242446626], [39.968018736414727, 28253.447171804281], [39.968035154317207, 28252.700487325488], [40.189655836527102, 28405.961873947806], [39.968007703632878, 28253.379865343017], [39.968008117335522, 28253.765566695656], [39.8324968893617, 28186.280024724736], [39.906652236738815, 28212.712307038702], [39.894059693996837, 28201.761703602224]])
Estime.incertezaParametros(.95,1e-5,metodo='2InvHessiana')
grandeza = Estime._armazenarDicionario()
Estime.predicao()
Estime.analiseResiduos(.95)
#
etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas,0.95)
Estime.relatorio(export_y=True,export_cov_y=True)

print Estime.y.estatisticas
print Estime.estatisticas
##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE I - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================
#
# #Sem o cálculo do termo independente
# ER = EstimacaoLinear(['y'],['x'],['p1'],projeto='LINEAR_semB')
# x = array([[0],[1],[2],[3],[4],[5]])
# y = array([[.1],[.9],[2.2],[3.2],[3.9],[4.8]])
# ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='experimental')
# #ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='validacao')
#
# # # Com o cálculo do termo independente
# # ER = EstimacaoLinear(['y'],['x'],['p1','p2'],projeto='LINEAR_comB')
# # x = array([[0],[1],[2],[3],[4],[5]])
# # y = array([[.1],[.9],[2.2],[3.2],[3.9],[4.8]])
# # ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='experimental')
# # ER.gerarEntradas(x,y,array([[1],[1],[1],[1],[1],[1]]),array([[1],[1],[1],[1],[1],[1]]),tipo='validacao')
# #
# # # =================================================================================
# # # PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# # # =================================================================================
# #
# ER.otimiza()
# #ER.SETparametro([0.9571428571428567, 0.12380952380952503])
# ER.incertezaParametros(.95)
# ER.predicao(delta=1e-6)
# ER.analiseResiduos(.95)
# ER.graficos(['analiseResiduos','regiaoAbrangencia', 'grandezas-entrada', 'predicao','grandezas-calculadas','otimizacao'],0.95)
# ER.relatorio()
